import os
import gc
import time
import logging
import threading
import pandas as pd
import numpy as np
import json
import zipfile
import shutil
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, session, jsonify,
)
from utils.data_analysis import perform_diagnostics
from utils.adaptive_cleaning import clean_dataset
from utils.intelligent_engine import run_intelligent_analysis
from utils.model_training import train_and_evaluate, MODEL_PATH, SCALER_PATH
from utils.report_generator import generate_text_report
from utils.dataset_expert import analyze_dataset_expertly
from utils.column_detector import detect_column_types
from utils.encoding_detector import read_csv_with_encoding
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import joblib
from functools import wraps

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('adie')

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(32).hex())

# Limit upload size to 50 MB
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DEFAULT_DATA_FOLDER = os.path.join('data', 'default')

# Max rows before we sample for heavy operations
MAX_ROWS_FULL = 10000

# ---------------------------------------------------------------------------
# User store — passwords hashed with werkzeug (pbkdf2)
# ---------------------------------------------------------------------------
USERS_FILE = os.path.join(UPLOAD_FOLDER, 'users.json')
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({"admin": generate_password_hash("password123")}, f)


# ---------------------------------------------------------------------------
# Background job tracker
# ---------------------------------------------------------------------------
_jobs = {}  # job_id -> {'status': str, 'error': str|None, 'started': float}


def _set_job(job_id, status, error=None):
    _jobs[job_id] = {
        'status': status,
        'error': error,
        'updated': time.time(),
    }


def _safe_sample(df, max_rows=MAX_ROWS_FULL):
    """Down-sample large datasets to stay within Render memory/time limits."""
    if len(df) > max_rows:
        logger.info('Sampling dataset from %d to %d rows', len(df), max_rows)
        return df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def _user_upload_dir():
    """Return a per-user upload directory, creating it if needed."""
    username = session.get('user', '_anonymous')
    user_dir = os.path.join(UPLOAD_FOLDER, f'user_{username}')
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'zip'}


def _build_cleaning_policy_from_form(form):
    mode = form.get('cleaning_mode', 'gentle')
    policy = {
        'mode': mode,
        'drop_identifier_columns': False,
        'drop_leakage_columns': False,
        'drop_high_missing_columns': False,
        'remove_duplicates': False,
        'handle_outliers': False,
        'encode_features': mode != 'gentle',
    }
    if mode == 'gentle':
        policy['encode_features'] = False
    return policy


def _quality_gates(before_df, after_df):
    warnings = []
    if len(before_df) > 0:
        r = abs(len(before_df) - len(after_df)) / len(before_df)
        if r > 0.2:
            warnings.append(f'High row change detected ({r:.1%}).')
    if len(before_df.columns) > 0:
        c = abs(len(before_df.columns) - len(after_df.columns)) / len(before_df.columns)
        if c > 0.1:
            warnings.append(f'High column change detected ({c:.1%}).')
    return warnings


def _distribution_drift_report(before_df, after_df):
    report = {'numeric_mean_shift': {}, 'numeric_std_shift': {}}
    common = [c for c in before_df.select_dtypes(include=np.number).columns
              if c in after_df.columns]
    for col in common:
        bm = float(before_df[col].mean()) if pd.notna(before_df[col].mean()) else 0.0
        am = float(after_df[col].mean()) if pd.notna(after_df[col].mean()) else 0.0
        bs = float(before_df[col].std()) if pd.notna(before_df[col].std()) else 0.0
        a_s = float(after_df[col].std()) if pd.notna(after_df[col].std()) else 0.0
        report['numeric_mean_shift'][col] = round(am - bm, 6)
        report['numeric_std_shift'][col] = round(a_s - bs, 6)
    return report


# ===================================================================
# ROUTES — Auth
# ===================================================================

@app.route('/health')
def health():
    return {'status': 'ok'}, 200


@app.route('/')
def splash():
    return render_template('splash.html')


@app.route('/login')
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('auth.html', signup=False)


@app.route('/signup')
def signup():
    return render_template('auth.html', signup=True)


@app.route('/login_post', methods=['POST'])
def login_post():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    if not username or not password:
        flash('Username and password are required.')
        return redirect(url_for('login'))
    try:
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        logger.error('Failed to read users file')
        flash('Authentication service error. Please try again.')
        return redirect(url_for('login'))
    stored = users.get(username)
    if stored and check_password_hash(stored, password):
        session['user'] = username
        logger.info('User %s logged in', username)
        flash(f'Welcome back, {username}!')
        return redirect(url_for('dashboard'))
    logger.warning('Failed login attempt for user: %s', username)
    flash('Invalid credentials. Please try again.')
    return redirect(url_for('login'))


@app.route('/signup_post', methods=['POST'])
def signup_post():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    if not username or not password:
        flash('Username and password are required.')
        return redirect(url_for('signup'))
    if len(password) < 6:
        flash('Password must be at least 6 characters.')
        return redirect(url_for('signup'))
    try:
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        users = {}
    if username in users:
        flash('Username already exists. Please choose another.')
        return redirect(url_for('signup'))
    users[username] = generate_password_hash(password)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)
    session['user'] = username
    logger.info('New user registered: %s', username)
    flash('Account created successfully! Welcome to ADIE.')
    return redirect(url_for('dashboard'))


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Successfully logged out.')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    user_dir = _user_upload_dir()
    default_datasets = []
    if os.path.exists(DEFAULT_DATA_FOLDER):
        default_datasets = [f for f in os.listdir(DEFAULT_DATA_FOLDER)
                            if f.endswith(('.csv', '.zip'))]
    has_model = os.path.exists(os.path.join(user_dir, 'best_model.pkl'))
    return render_template('index.html',
                           default_datasets=default_datasets,
                           has_model=has_model)


# ===================================================================
# ROUTES — Pipeline (analyze, clean, train) with background processing
# ===================================================================

# ---------- /analyze_default ----------
@app.route('/analyze_default', methods=['POST'])
@login_required
def analyze_default():
    try:
        user_dir = _user_upload_dir()
        selected_file = request.form.get('default_file')
        if not selected_file:
            flash('No default file selected')
            return redirect(url_for('dashboard'))
        source_path = os.path.join(DEFAULT_DATA_FOLDER, selected_file)
        if not os.path.exists(source_path):
            flash('Selected default file not found')
            return redirect(url_for('dashboard'))

        target_path = os.path.join(user_dir, 'current_dataset.csv')
        if selected_file.endswith('.zip'):
            with zipfile.ZipFile(source_path, 'r') as zf:
                csvs = [f for f in zf.namelist() if f.endswith('.csv')]
                if not csvs:
                    flash('No CSV file found inside the ZIP')
                    return redirect(url_for('dashboard'))
                zf.extract(csvs[0], user_dir)
                tmp = os.path.join(user_dir, csvs[0])
                if os.path.exists(target_path):
                    os.remove(target_path)
                os.rename(tmp, target_path)
        else:
            shutil.copy(source_path, target_path)

        t0 = time.time()
        df = read_csv_with_encoding(target_path)
        df_diag = _safe_sample(df)
        target_col = df.columns[-1]
        col_types = detect_column_types(df, target_col)

        metadata = {
            "filename": selected_file,
            "size_kb": round(os.path.getsize(target_path) / 1024, 2),
            "rows": df.shape[0], "columns": df.shape[1],
            "column_names": df.columns.tolist(),
            "types": df.dtypes.astype(str).to_dict(),
            "column_types": {
                "identifiers": col_types['identifiers'],
                "datetime_cols": col_types['datetime_cols'],
                "numerical_cols": col_types['numerical_cols'],
                "nominal_categorical": col_types['nominal_categorical'],
                "ordinal_categorical": col_types['ordinal_categorical'],
            },
        }
        with open(os.path.join(user_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        diagnostics = perform_diagnostics(df_diag)
        expert_report = analyze_dataset_expertly(df_diag, diagnostics)

        with open(os.path.join(user_dir, 'diagnostics.json'), 'w') as f:
            json.dump(diagnostics, f)
        with open(os.path.join(user_dir, 'expert_report.json'), 'w') as f:
            json.dump(expert_report, f)

        logger.info('analyze_default completed in %.1fs for user %s',
                     time.time() - t0, session.get('user'))
        gc.collect()
        return render_template('result.html', diagnostics=diagnostics,
                               expert_report=expert_report, metadata=metadata,
                               filename=selected_file)
    except Exception as e:
        logger.exception('analyze_default failed: %s', e)
        flash(f'Failed to analyze default dataset: {str(e)}')
        return redirect(url_for('dashboard'))


# ---------- /analyze ----------
@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('dashboard'))
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        flash('Please upload a valid CSV or ZIP file')
        return redirect(url_for('dashboard'))
    try:
        user_dir = _user_upload_dir()
        filename = secure_filename(file.filename)
        filepath = os.path.join(user_dir, filename)
        file.save(filepath)

        target_path = os.path.join(user_dir, 'current_dataset.csv')
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zf:
                csvs = [f for f in zf.namelist() if f.endswith('.csv')]
                if not csvs:
                    flash('No CSV file found inside the ZIP')
                    return redirect(url_for('dashboard'))
                zf.extract(csvs[0], user_dir)
                csv_path = os.path.join(user_dir, csvs[0])
                if os.path.exists(target_path):
                    os.remove(target_path)
                os.rename(csv_path, target_path)
        else:
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(filepath, target_path)

        t0 = time.time()
        df = read_csv_with_encoding(target_path)
        df.columns = df.columns.str.strip()
        df_diag = _safe_sample(df)
        target_col = df.columns[-1]
        col_types = detect_column_types(df, target_col)

        metadata = {
            "filename": filename,
            "size_kb": round(os.path.getsize(target_path) / 1024, 2),
            "rows": df.shape[0], "columns": df.shape[1],
            "column_names": df.columns.tolist(),
            "types": df.dtypes.astype(str).to_dict(),
            "column_types": {
                "identifiers": col_types['identifiers'],
                "datetime_cols": col_types['datetime_cols'],
                "numerical_cols": col_types['numerical_cols'],
                "nominal_categorical": col_types['nominal_categorical'],
                "ordinal_categorical": col_types['ordinal_categorical'],
            },
        }
        with open(os.path.join(user_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        diagnostics = perform_diagnostics(df_diag)
        expert_report = analyze_dataset_expertly(df_diag, diagnostics)

        with open(os.path.join(user_dir, 'diagnostics.json'), 'w') as f:
            json.dump(diagnostics, f)
        with open(os.path.join(user_dir, 'expert_report.json'), 'w') as f:
            json.dump(expert_report, f)

        logger.info('analyze completed in %.1fs for user %s',
                     time.time() - t0, session.get('user'))
        gc.collect()
        return render_template('result.html', diagnostics=diagnostics,
                               expert_report=expert_report, metadata=metadata,
                               filename=filename)
    except Exception as e:
        logger.exception('analyze failed: %s', e)
        flash(f'Failed to analyze dataset: {str(e)}')
        return redirect(url_for('dashboard'))


# ---------- /clean (background thread) ----------

def _run_clean_job(job_id, user_dir, df, target_col, leakage_cols,
                   cleaning_policy, orig_diagnostics):
    """Heavy cleaning work executed in a background thread."""
    try:
        _set_job(job_id, 'running')
        t0 = time.time()

        strategy = run_intelligent_analysis(df, target_col)

        version_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = os.path.join(user_dir, f'version_{version_ts}')
        os.makedirs(version_dir, exist_ok=True)
        df.to_csv(os.path.join(version_dir, 'before_clean.csv'), index=False)

        # Human-readable cleaned output
        cleaned_df, cleaning_report = clean_dataset(
            df, leakage_cols=leakage_cols, target_col=target_col,
            preserve_structure=True, cleaning_policy=cleaning_policy,
            return_report=True, strategy=strategy,
        )
        cleaned_path = os.path.join(user_dir, 'cleaned_dataset.csv')
        cleaned_df.to_csv(cleaned_path, index=False)
        shutil.copy(cleaned_path, os.path.join(version_dir, 'after_clean.csv'))

        # ML-ready output
        ml_policy = {
            'mode': 'balanced', 'drop_identifier_columns': False,
            'drop_leakage_columns': False, 'drop_high_missing_columns': False,
            'remove_duplicates': True, 'handle_outliers': True,
            'encode_features': True,
        }
        ml_df = clean_dataset(
            df, leakage_cols=leakage_cols, target_col=target_col,
            preserve_structure=False, cleaning_policy=ml_policy,
            strategy=strategy,
        )
        ml_path = os.path.join(user_dir, 'ml_ready_dataset.csv')
        ml_df.to_csv(ml_path, index=False)
        shutil.copy(ml_path, os.path.join(version_dir, 'ml_ready_dataset.csv'))

        # Post-clean diagnostics
        diagnostics = perform_diagnostics(_safe_sample(cleaned_df))
        expert_report = analyze_dataset_expertly(
            _safe_sample(cleaned_df), diagnostics, is_repaired=True)

        version_info = {
            'timestamp': version_ts, 'random_seed': 42,
            'original_rows': len(df), 'cleaned_rows': len(cleaned_df),
            'original_columns': len(df.columns),
            'cleaned_columns': len(cleaned_df.columns),
            'original_issues': len(orig_diagnostics.get('identified_issues', [])),
            'cleaned_issues': len(diagnostics.get('identified_issues', [])),
            'cleaning_policy': cleaning_policy, 'improvements': {},
        }
        if 'identified_issues' in orig_diagnostics and 'identified_issues' in diagnostics:
            orig_types = {i['type'] for i in orig_diagnostics['identified_issues']}
            clean_types = {i['type'] for i in diagnostics['identified_issues']}
            resolved = orig_types - clean_types
            version_info['improvements'] = {
                'issues_resolved': len(resolved),
                'resolved_list': list(resolved),
                'rows_removed': len(df) - len(cleaned_df),
                'columns_removed': len(df.columns) - len(cleaned_df.columns),
            }

        # Persist all artifacts
        for name, obj in [
            ('version_info.json', version_info),
            ('cleaning_log.json', cleaning_report),
            ('drift_report.json', _distribution_drift_report(df, cleaned_df)),
        ]:
            with open(os.path.join(version_dir, name), 'w') as f:
                json.dump(obj, f)

        for name, obj in [
            ('cleaning_log.json', cleaning_report),
            ('original_diagnostics.json', orig_diagnostics),
            ('diagnostics.json', diagnostics),
            ('expert_report.json', expert_report),
        ]:
            with open(os.path.join(user_dir, name), 'w') as f:
                json.dump(obj, f)

        # Update metadata
        meta_path = os.path.join(user_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            if 'column_types' in diagnostics:
                metadata['column_types'] = diagnostics['column_types']
                metadata['cleaning_mode'] = cleaning_policy.get('mode', 'gentle')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)

        logger.info('clean job %s completed in %.1fs', job_id, time.time() - t0)
        _set_job(job_id, 'done')
    except Exception as e:
        logger.exception('clean job %s failed: %s', job_id, e)
        _set_job(job_id, 'error', str(e))
    finally:
        gc.collect()


@app.route('/clean', methods=['POST'])
@login_required
def clean():
    user_dir = _user_upload_dir()
    filepath = os.path.join(user_dir, 'current_dataset.csv')
    diag_path = os.path.join(user_dir, 'diagnostics.json')

    if not os.path.exists(filepath):
        flash('No dataset found to clean')
        return redirect(url_for('dashboard'))

    leakage_cols = None
    if os.path.exists(diag_path):
        with open(diag_path, 'r') as f:
            leakage_cols = json.load(f).get('leakage_risk')

    try:
        df = read_csv_with_encoding(filepath)
        df.columns = df.columns.str.strip()
        target_col = df.columns[-1]
        cleaning_policy = _build_cleaning_policy_from_form(request.form)

        if leakage_cols and target_col in leakage_cols:
            leakage_cols = [c for c in leakage_cols if c != target_col]
            if not leakage_cols:
                leakage_cols = None

        orig_diagnostics = perform_diagnostics(_safe_sample(df))

        # Launch heavy work in background thread
        job_id = f'clean_{session.get("user")}_{int(time.time())}'
        _set_job(job_id, 'started')
        session['clean_job'] = job_id

        t = threading.Thread(
            target=_run_clean_job,
            args=(job_id, user_dir, df, target_col, leakage_cols,
                  cleaning_policy, orig_diagnostics),
            daemon=True,
        )
        t.start()

        flash('ADIE Pipeline started — processing your dataset. '
              'This page will show results when ready.')
        return redirect(url_for('clean_status'))
    except Exception as e:
        logger.exception('clean route failed: %s', e)
        flash(f'Failed to start cleaning: {str(e)}')
        return redirect(url_for('dashboard'))


@app.route('/clean_status')
@login_required
def clean_status():
    """Poll endpoint: redirects to results when background clean finishes."""
    job_id = session.get('clean_job')
    if not job_id or job_id not in _jobs:
        flash('No cleaning job found. Please run the pipeline first.')
        return redirect(url_for('dashboard'))

    job = _jobs[job_id]
    if job['status'] == 'done':
        # Load results from disk and render
        user_dir = _user_upload_dir()
        diagnostics = expert_report = metadata = orig_diagnostics = None
        for name, var in [('diagnostics.json', 'diagnostics'),
                          ('expert_report.json', 'expert_report'),
                          ('metadata.json', 'metadata'),
                          ('original_diagnostics.json', 'orig_diagnostics')]:
            p = os.path.join(user_dir, name)
            if os.path.exists(p):
                with open(p, 'r') as f:
                    locals()[var] = json.load(f)

        # Re-read from disk to be safe
        diagnostics = json.load(open(os.path.join(user_dir, 'diagnostics.json'))) if os.path.exists(os.path.join(user_dir, 'diagnostics.json')) else {}
        expert_report = json.load(open(os.path.join(user_dir, 'expert_report.json'))) if os.path.exists(os.path.join(user_dir, 'expert_report.json')) else {}
        metadata = json.load(open(os.path.join(user_dir, 'metadata.json'))) if os.path.exists(os.path.join(user_dir, 'metadata.json')) else {}
        orig_diagnostics = json.load(open(os.path.join(user_dir, 'original_diagnostics.json'))) if os.path.exists(os.path.join(user_dir, 'original_diagnostics.json')) else {}

        flash('ADIE Pipeline: Dataset successfully repaired and optimized!')
        return render_template('result.html',
                               diagnostics=diagnostics,
                               expert_report=expert_report,
                               metadata=metadata,
                               filename=metadata.get('filename', 'dataset.csv') if metadata else 'dataset.csv',
                               cleaned=True,
                               orig_diagnostics=orig_diagnostics)

    if job['status'] == 'error':
        flash(f'Pipeline failed: {job.get("error", "Unknown error")}')
        return redirect(url_for('dashboard'))

    # Still running — show a waiting page that auto-refreshes
    return render_template('processing.html', job_id=job_id, stage='Cleaning')


@app.route('/job_status/<job_id>')
@login_required
def job_status(job_id):
    """JSON endpoint for AJAX polling."""
    job = _jobs.get(job_id, {'status': 'unknown'})
    return jsonify(job)


# ---------- /train (background thread) ----------

def _run_train_job(job_id, user_dir, df_orig, target_col,
                   quarantined_columns, selected_algo, ml_train_policy):
    """Heavy training work executed in a background thread."""
    try:
        _set_job(job_id, 'running')
        t0 = time.time()

        strategy = run_intelligent_analysis(df_orig, target_col)
        df_sampled = _safe_sample(df_orig)

        # Train on original
        try:
            df_orig_proc = clean_dataset(
                df_sampled, target_col=target_col,
                preserve_structure=False, cleaning_policy=ml_train_policy,
                strategy=strategy,
            )
            orig_results, task_type = train_and_evaluate(
                df_orig_proc, target_col, selected_algo,
                quarantined_columns=quarantined_columns,
            )
        except Exception as e:
            logger.exception('Original training failed: %s', e)
            orig_results = {selected_algo: {'error': str(e)}}
            task_type = 'classification'

        # Train on cleaned
        cleaned_path = os.path.join(user_dir, 'cleaned_dataset.csv')
        ml_ready_path = os.path.join(user_dir, 'ml_ready_dataset.csv')
        cleaned_results = {}
        if os.path.exists(cleaned_path):
            try:
                if os.path.exists(ml_ready_path):
                    df_cl = read_csv_with_encoding(ml_ready_path)
                    df_cl.columns = df_cl.columns.str.strip()
                else:
                    df_raw = read_csv_with_encoding(cleaned_path)
                    df_raw.columns = df_raw.columns.str.strip()
                    df_cl = clean_dataset(
                        _safe_sample(df_raw), target_col=target_col,
                        preserve_structure=False, cleaning_policy=ml_train_policy,
                        strategy=strategy,
                    )
                df_cl = _safe_sample(df_cl)
                cleaned_results, _ = train_and_evaluate(
                    df_cl, target_col, selected_algo,
                    quarantined_columns=quarantined_columns,
                )
            except Exception as e:
                logger.exception('Cleaned training failed: %s', e)
                cleaned_results = {selected_algo: {'error': str(e)}}

        # Persist results
        results_data = {
            'orig_results': orig_results or {},
            'cleaned_results': cleaned_results or {},
            'task_type': task_type,
            'selected_algo': selected_algo,
        }
        with open(os.path.join(user_dir, 'ml_results.json'), 'w') as f:
            json.dump(results_data, f)

        logger.info('train job %s completed in %.1fs', job_id, time.time() - t0)
        _set_job(job_id, 'done')
    except Exception as e:
        logger.exception('train job %s failed: %s', job_id, e)
        _set_job(job_id, 'error', str(e))
    finally:
        gc.collect()


@app.route('/train', methods=['POST'])
@login_required
def train():
    user_dir = _user_upload_dir()
    orig_filepath = os.path.join(user_dir, 'current_dataset.csv')
    selected_algo = request.form.get('algorithm', 'All Algorithms')

    if not os.path.exists(orig_filepath):
        flash('Original dataset not found')
        return redirect(url_for('dashboard'))

    df_orig = read_csv_with_encoding(orig_filepath)
    df_orig.columns = df_orig.columns.str.strip()
    target_col = df_orig.columns[-1]

    quarantined_columns = []
    log_path = os.path.join(user_dir, 'cleaning_log.json')
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                quarantined_columns = json.load(f).get('quarantined_columns', [])
        except Exception:
            logger.warning('Could not read cleaning log')

    ml_policy = {
        'mode': 'balanced', 'drop_identifier_columns': False,
        'drop_leakage_columns': False, 'drop_high_missing_columns': False,
        'remove_duplicates': True, 'handle_outliers': True,
        'encode_features': True,
    }

    job_id = f'train_{session.get("user")}_{int(time.time())}'
    _set_job(job_id, 'started')
    session['train_job'] = job_id

    t = threading.Thread(
        target=_run_train_job,
        args=(job_id, user_dir, df_orig, target_col,
              quarantined_columns, selected_algo, ml_policy),
        daemon=True,
    )
    t.start()

    flash('Model training started — this may take a moment.')
    return redirect(url_for('train_status'))


@app.route('/train_status')
@login_required
def train_status():
    """Poll endpoint: redirects to results when background training finishes."""
    job_id = session.get('train_job')
    if not job_id or job_id not in _jobs:
        flash('No training job found. Please run training first.')
        return redirect(url_for('dashboard'))

    job = _jobs[job_id]
    if job['status'] == 'done':
        user_dir = _user_upload_dir()
        res_path = os.path.join(user_dir, 'ml_results.json')
        if not os.path.exists(res_path):
            flash('Training results not found.')
            return redirect(url_for('dashboard'))

        with open(res_path, 'r') as f:
            results_data = json.load(f)

        orig_results = results_data.get('orig_results', {})
        cleaned_results = results_data.get('cleaned_results', {})
        task_type = results_data.get('task_type', 'classification')
        selected_algo = results_data.get('selected_algo', 'All Algorithms')

        selected_metrics = None
        if selected_algo != 'All Algorithms':
            selected_metrics = cleaned_results.get(selected_algo)
            if selected_metrics is None:
                selected_metrics = {'error': f'"{selected_algo}" not available for {task_type}.'}
                cleaned_results[selected_algo] = selected_metrics

        expert_report = metadata = diagnostics = None
        for name in ['expert_report.json', 'metadata.json', 'diagnostics.json']:
            p = os.path.join(user_dir, name)
            if os.path.exists(p):
                with open(p, 'r') as f:
                    val = json.load(f)
                if name == 'expert_report.json':
                    expert_report = val
                elif name == 'metadata.json':
                    metadata = val
                else:
                    diagnostics = val

        logger.info('Training results served for user %s', session.get('user'))
        return render_template('result.html',
                               orig_results=orig_results,
                               cleaned_results=cleaned_results,
                               task_type=task_type,
                               selected_algo=selected_algo,
                               selected_metrics=selected_metrics,
                               expert_report=expert_report,
                               metadata=metadata,
                               diagnostics=diagnostics,
                               filename=metadata.get('filename', 'dataset.csv') if metadata else 'dataset.csv',
                               trained=True)

    if job['status'] == 'error':
        flash(f'Training failed: {job.get("error", "Unknown error")}')
        return redirect(url_for('dashboard'))

    return render_template('processing.html', job_id=job_id, stage='Training')


# ===================================================================
# ROUTES — Downloads
# ===================================================================

@app.route('/download_cleaned')
@login_required
def download_cleaned():
    user_dir = _user_upload_dir()
    fp = os.path.join(user_dir, 'cleaned_dataset.csv')
    if os.path.exists(fp):
        return send_file(fp, as_attachment=True, download_name='cleaned_dataset.csv')
    flash('Cleaned dataset not found')
    return redirect(url_for('dashboard'))


@app.route('/download_report')
@login_required
def download_report():
    user_dir = _user_upload_dir()
    diag_p = os.path.join(user_dir, 'diagnostics.json')
    res_p = os.path.join(user_dir, 'ml_results.json')
    exp_p = os.path.join(user_dir, 'expert_report.json')

    if os.path.exists(diag_p) and os.path.exists(res_p) and os.path.exists(exp_p):
        with open(diag_p) as f:
            diagnostics = json.load(f)
        with open(res_p) as f:
            results_data = json.load(f)
        with open(exp_p) as f:
            expert_report = json.load(f)

        report_text = generate_text_report(
            diagnostics, expert_report,
            results_data.get('orig_results', {}),
            results_data['cleaned_results'],
            results_data['task_type'],
            results_data['selected_algo'],
        )
        report_path = os.path.join(user_dir, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        return send_file(report_path, as_attachment=True,
                         download_name='ML_Analysis_Report.txt')

    flash('Please perform analysis and train models before downloading the report.')
    return redirect(url_for('dashboard'))


# ===================================================================
# Startup
# ===================================================================
if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    app.run(debug=True)
