import os
import pandas as pd
import numpy as np
import json
import zipfile
import shutil
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
from utils.data_analysis import perform_diagnostics
from utils.data_cleaning import clean_dataset
from utils.model_training import train_and_evaluate, MODEL_PATH, SCALER_PATH
from utils.report_generator import generate_text_report
from utils.dataset_expert import analyze_dataset_expertly
from utils.column_detector import detect_column_types
from utils.encoding_detector import read_csv_with_encoding
import joblib
from functools import wraps

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DEFAULT_DATA_FOLDER = os.path.join('data', 'default')

# Mock database for authentication
USERS_FILE = os.path.join(UPLOAD_FOLDER, 'users.json')
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({"admin": "password123"}, f)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

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
    username = request.form.get('username')
    password = request.form.get('password')
    
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    
    if username in users and users[username] == password:
        session['user'] = username
        flash(f'Welcome back, {username}!')
        return redirect(url_for('dashboard'))
    
    flash('Invalid credentials. Please try again.')
    return redirect(url_for('login'))

@app.route('/signup_post', methods=['POST'])
def signup_post():
    username = request.form.get('username')
    password = request.form.get('password')
    
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    
    if username in users:
        flash('Username already exists. Please choose another.')
        return redirect(url_for('signup'))
    
    users[username] = password
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)
    
    session['user'] = username
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
    # List default datasets
    default_datasets = []
    if os.path.exists(DEFAULT_DATA_FOLDER):
        default_datasets = [f for f in os.listdir(DEFAULT_DATA_FOLDER) if f.endswith(('.csv', '.zip'))]
    has_model = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'best_model.pkl'))
    return render_template('index.html', default_datasets=default_datasets, has_model=has_model)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'zip'}

def _build_cleaning_policy_from_form(form):
    mode = form.get('cleaning_mode', 'gentle')
    policy = {
        'mode': mode,
        'drop_identifier_columns': False,
        'drop_leakage_columns': False,
        'drop_high_missing_columns': False,
        # Enforced safety: never remove or replace correct data
        'remove_duplicates': False,
        'handle_outliers': False,
        'encode_features': mode != 'gentle'
    }
    # Hard safety rule requested: never drop columns in any mode.
    if mode == 'gentle':
        policy['encode_features'] = False
    return policy

def _quality_gates(before_df, after_df):
    warnings = []
    if len(before_df) > 0:
        row_change_ratio = abs(len(before_df) - len(after_df)) / len(before_df)
        if row_change_ratio > 0.2:
            warnings.append(f'High row change detected ({row_change_ratio:.1%}).')
    if len(before_df.columns) > 0:
        col_change_ratio = abs(len(before_df.columns) - len(after_df.columns)) / len(before_df.columns)
        if col_change_ratio > 0.1:
            warnings.append(f'High column change detected ({col_change_ratio:.1%}).')
    return warnings

def _distribution_drift_report(before_df, after_df):
    report = {'numeric_mean_shift': {}, 'numeric_std_shift': {}}
    common_numeric = [c for c in before_df.select_dtypes(include=np.number).columns if c in after_df.columns]
    for col in common_numeric:
        b_mean = float(before_df[col].mean()) if pd.notna(before_df[col].mean()) else 0.0
        a_mean = float(after_df[col].mean()) if pd.notna(after_df[col].mean()) else 0.0
        b_std = float(before_df[col].std()) if pd.notna(before_df[col].std()) else 0.0
        a_std = float(after_df[col].std()) if pd.notna(after_df[col].std()) else 0.0
        report['numeric_mean_shift'][col] = round(a_mean - b_mean, 6)
        report['numeric_std_shift'][col] = round(a_std - b_std, 6)
    return report

@app.route('/analyze_default', methods=['POST'])
@login_required
def analyze_default():
    try:
        selected_file = request.form.get('default_file')
        if not selected_file:
            flash('No default file selected')
            return redirect(url_for('dashboard'))
        
        source_path = os.path.join(DEFAULT_DATA_FOLDER, selected_file)
        if not os.path.exists(source_path):
            flash('Selected default file not found')
            return redirect(url_for('dashboard'))
        
        # Copy to uploads for processing (following the pipeline flow)
        target_name = 'current_dataset.csv'
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_name)
        
        if selected_file.endswith('.zip'):
            with zipfile.ZipFile(source_path, 'r') as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if not csv_files:
                    flash('No CSV file found inside the ZIP')
                    return redirect(url_for('dashboard'))
                zip_ref.extract(csv_files[0], app.config['UPLOAD_FOLDER'])
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_files[0])
                if os.path.exists(target_path): os.remove(target_path)
                os.rename(temp_path, target_path)
        else:
            import shutil
            shutil.copy(source_path, target_path)
        
        # Now trigger the ADIE Pipeline just like a regular upload
        df = read_csv_with_encoding(target_path)
        target_col = df.columns[-1]
        
        # Detect column types for metadata
        col_types = detect_column_types(df, target_col)
        
        # Metadata Extraction
        metadata = {
            "filename": selected_file,
            "size_kb": round(os.path.getsize(target_path) / 1024, 2),
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns.tolist(),
            "types": df.dtypes.astype(str).to_dict(),
            "column_types": {
                "identifiers": col_types['identifiers'],
                "datetime_cols": col_types['datetime_cols'],
                "numerical_cols": col_types['numerical_cols'],
                "nominal_categorical": col_types['nominal_categorical'],
                "ordinal_categorical": col_types['ordinal_categorical']
            }
        }
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        diagnostics = perform_diagnostics(df)
        expert_report = analyze_dataset_expertly(df, diagnostics)
        
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'diagnostics.json'), 'w') as f:
            json.dump(diagnostics, f)
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'expert_report.json'), 'w') as f:
            json.dump(expert_report, f)
            
        return render_template('result.html', diagnostics=diagnostics, expert_report=expert_report, metadata=metadata, filename=selected_file)
    except Exception as e:
        flash(f'Failed to analyze default dataset: {str(e)}')
        return redirect(url_for('dashboard'))

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    # 1. User -> Web Interface (Frontend) -> POST Request
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('dashboard'))
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        flash('Please upload a valid CSV or ZIP file')
        return redirect(url_for('dashboard'))
    
    try:
        # 2. Backend API (Python) -> Validation
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        csv_path = None
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if not csv_files:
                    flash('No CSV file found inside the ZIP')
                    return redirect(url_for('dashboard'))
                zip_ref.extract(csv_files[0], app.config['UPLOAD_FOLDER'])
                csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_files[0])
                target_path = os.path.join(app.config['UPLOAD_FOLDER'], 'current_dataset.csv')
                if os.path.exists(target_path):
                    os.remove(target_path)
                os.rename(csv_path, target_path)
                csv_path = target_path
        else:
            target_path = os.path.join(app.config['UPLOAD_FOLDER'], 'current_dataset.csv')
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(filepath, target_path)
            csv_path = target_path

        # 3. Dataset Storage (Local Hybrid) -> Trigger Processing (ADIE Pipeline)
        df = read_csv_with_encoding(csv_path)
        df.columns = df.columns.str.strip()
        target_col = df.columns[-1]
        col_types = detect_column_types(df, target_col)
        
        metadata = {
            "filename": filename,
            "size_kb": round(os.path.getsize(csv_path) / 1024, 2),
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns.tolist(),
            "types": df.dtypes.astype(str).to_dict(),
            "column_types": {
                "identifiers": col_types['identifiers'],
                "datetime_cols": col_types['datetime_cols'],
                "numerical_cols": col_types['numerical_cols'],
                "nominal_categorical": col_types['nominal_categorical'],
                "ordinal_categorical": col_types['ordinal_categorical']
            }
        }
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        diagnostics = perform_diagnostics(df)
        expert_report = analyze_dataset_expertly(df, diagnostics)
        
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'diagnostics.json'), 'w') as f:
            json.dump(diagnostics, f)
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'expert_report.json'), 'w') as f:
            json.dump(expert_report, f)
        
        return render_template('result.html', diagnostics=diagnostics, expert_report=expert_report, metadata=metadata, filename=filename)
    except Exception as e:
        flash(f'Failed to analyze dataset: {str(e)}')
        return redirect(url_for('dashboard'))

@app.route('/clean', methods=['POST'])
@login_required
def clean():
    # ADIE Pipeline Stage 2: Data Cleaning (Repair)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'current_dataset.csv')
    diag_path = os.path.join(app.config['UPLOAD_FOLDER'], 'diagnostics.json')
    
    if not os.path.exists(filepath):
        flash('No dataset found to clean')
        return redirect(url_for('dashboard'))
    
    # Load leakage info from diagnostics if it exists
    leakage_cols = None
    if os.path.exists(diag_path):
        with open(diag_path, 'r') as f:
            old_diag = json.load(f)
            leakage_cols = old_diag.get('leakage_risk')
    
    try:
        df = read_csv_with_encoding(filepath)
        df.columns = df.columns.str.strip()
        target_col = df.columns[-1]
        cleaning_policy = _build_cleaning_policy_from_form(request.form)
        
        if leakage_cols and target_col in leakage_cols:
            leakage_cols = [c for c in leakage_cols if c != target_col]
            if not leakage_cols:
                leakage_cols = None
        
        orig_diagnostics = perform_diagnostics(df)
        version_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'version_{version_timestamp}')
        os.makedirs(version_dir, exist_ok=True)
        shutil.copy(filepath, os.path.join(version_dir, 'before_clean.csv'))
        
        cleaned_df, cleaning_report = clean_dataset(
            df,
            leakage_cols=leakage_cols,
            target_col=target_col,
            preserve_structure=True,
            cleaning_policy=cleaning_policy,
            return_report=True
        )
        
        cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_dataset.csv')
        cleaned_df.to_csv(cleaned_filepath, index=False)
        shutil.copy(cleaned_filepath, os.path.join(version_dir, 'after_clean.csv'))

        # Generate ML-ready dataset separately to keep user export human-readable.
        ml_policy = {
            'mode': 'balanced',
            'drop_identifier_columns': False,
            'drop_leakage_columns': False,
            'drop_high_missing_columns': False,
            'remove_duplicates': True,
            'handle_outliers': True,
            'encode_features': True
        }
        ml_ready_df = clean_dataset(
            df,
            leakage_cols=leakage_cols,
            target_col=target_col,
            preserve_structure=False,
            cleaning_policy=ml_policy
        )
        ml_ready_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ml_ready_dataset.csv')
        ml_ready_df.to_csv(ml_ready_path, index=False)
        shutil.copy(ml_ready_path, os.path.join(version_dir, 'ml_ready_dataset.csv'))
        
        version_info = {
            'timestamp': version_timestamp,
            'random_seed': 42,
            'original_rows': len(df),
            'cleaned_rows': len(cleaned_df),
            'original_columns': len(df.columns),
            'cleaned_columns': len(cleaned_df.columns),
            'original_issues': len(orig_diagnostics.get('identified_issues', [])),
            'cleaned_issues': 0,
            'cleaning_policy': cleaning_policy,
            'improvements': {}
        }
        
        diagnostics = perform_diagnostics(cleaned_df)
        expert_report = analyze_dataset_expertly(cleaned_df, diagnostics, is_repaired=True)
        version_info['cleaned_issues'] = len(diagnostics.get('identified_issues', []))
        
        if 'identified_issues' in orig_diagnostics and 'identified_issues' in diagnostics:
            orig_issue_types = {i['type'] for i in orig_diagnostics['identified_issues']}
            cleaned_issue_types = {i['type'] for i in diagnostics['identified_issues']}
            resolved_issues = orig_issue_types - cleaned_issue_types
            version_info['improvements'] = {
                'issues_resolved': len(resolved_issues),
                'resolved_list': list(resolved_issues),
                'rows_removed': len(df) - len(cleaned_df),
                'columns_removed': len(df.columns) - len(cleaned_df.columns)
            }
        
        with open(os.path.join(version_dir, 'version_info.json'), 'w') as f:
            json.dump(version_info, f)
        with open(os.path.join(version_dir, 'cleaning_log.json'), 'w') as f:
            json.dump(cleaning_report, f)
        with open(os.path.join(version_dir, 'drift_report.json'), 'w') as f:
            json.dump(_distribution_drift_report(df, cleaned_df), f)
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'cleaning_log.json'), 'w') as f:
            json.dump(cleaning_report, f)
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'original_diagnostics.json'), 'w') as f:
            json.dump(orig_diagnostics, f)
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'diagnostics.json'), 'w') as f:
            json.dump(diagnostics, f)
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'expert_report.json'), 'w') as f:
            json.dump(expert_report, f)
        
        meta_path = os.path.join(app.config['UPLOAD_FOLDER'], 'metadata.json')
        metadata = None
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            if 'column_types' in diagnostics:
                metadata['column_types'] = diagnostics['column_types']
                metadata['cleaning_mode'] = cleaning_policy.get('mode', 'gentle')
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)

        for warning in _quality_gates(df, cleaned_df):
            flash(f'Quality gate warning: {warning}')
        flash('ADIE Pipeline: Dataset successfully repaired and optimized!')
        
        return render_template('result.html', 
                               diagnostics=diagnostics, 
                               expert_report=expert_report, 
                               metadata=metadata, 
                               filename=metadata.get('filename') if metadata else 'dataset.csv', 
                               cleaned=True,
                               orig_diagnostics=orig_diagnostics)
    except Exception as e:
        flash(f'Failed to clean dataset: {str(e)}')
        return redirect(url_for('dashboard'))

@app.route('/train', methods=['POST'])
@login_required
def train():
    orig_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'current_dataset.csv')
    cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_dataset.csv')
    selected_algo = request.form.get('algorithm', 'All Algorithms')
    
    if not os.path.exists(orig_filepath):
        flash('Original dataset not found')
        return redirect(url_for('dashboard'))
        
    df_orig = read_csv_with_encoding(orig_filepath)
    
    # FIX: Sanitize column names
    df_orig.columns = df_orig.columns.str.strip()
    
    target_col = df_orig.columns[-1]
    
    quarantined_columns = []
    cleaning_log_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaning_log.json')
    if os.path.exists(cleaning_log_path):
        try:
            with open(cleaning_log_path, 'r') as f:
                cleaning_report = json.load(f)
                quarantined_columns = cleaning_report.get('quarantined_columns', [])
        except Exception:
            pass
    
    ml_ready_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ml_ready_dataset.csv')
    ml_train_policy = {
        'mode': 'balanced',
        'drop_identifier_columns': False,
        'drop_leakage_columns': False,
        'drop_high_missing_columns': False,
        'remove_duplicates': True,
        'handle_outliers': True,
        'encode_features': True
    }

    try:
        df_orig_processed = clean_dataset(
            df_orig,
            target_col=target_col,
            preserve_structure=False,
            cleaning_policy=ml_train_policy
        )
        orig_results, task_type = train_and_evaluate(
            df_orig_processed, 
            target_col, 
            selected_algo, 
            quarantined_columns=quarantined_columns
        )
    except Exception as e:
        orig_results = {selected_algo: {'error': f'Original dataset training failed: {str(e)}'}}
        task_type = 'classification'
    
    if os.path.exists(cleaned_filepath):
        df_cleaned = read_csv_with_encoding(cleaned_filepath)
        
        # FIX: Sanitize column names
        df_cleaned.columns = df_cleaned.columns.str.strip()
        
        try:
            if os.path.exists(ml_ready_path):
                df_cleaned_processed = read_csv_with_encoding(ml_ready_path)
                df_cleaned_processed.columns = df_cleaned_processed.columns.str.strip()
            else:
                df_cleaned_processed = clean_dataset(
                    df_cleaned,
                    target_col=target_col,
                    preserve_structure=False,
                    cleaning_policy=ml_train_policy
                )
            cleaned_results, _ = train_and_evaluate(
                df_cleaned_processed, 
                target_col, 
                selected_algo, 
                quarantined_columns=quarantined_columns
            )
        except Exception as e:
            cleaned_results = {selected_algo: {'error': f'Cleaned dataset training failed: {str(e)}'}}
    else:
        cleaned_results = {}
    
    if cleaned_results is None:
        cleaned_results = {}
    if orig_results is None:
        orig_results = {}
    
    selected_metrics = None
    if selected_algo != 'All Algorithms':
        selected_metrics = cleaned_results.get(selected_algo)
        if selected_metrics is None:
            # Ensure template can always render a deterministic status.
            selected_metrics = {'error': f'"{selected_algo}" is not available for detected task type ({task_type}).'}
            cleaned_results[selected_algo] = selected_metrics
    
    # Get expert report, diagnostics, and metadata from disk
    expert_report = None
    metadata = None
    diagnostics = None
    expert_path = os.path.join(app.config['UPLOAD_FOLDER'], 'expert_report.json')
    meta_path = os.path.join(app.config['UPLOAD_FOLDER'], 'metadata.json')
    diag_path = os.path.join(app.config['UPLOAD_FOLDER'], 'diagnostics.json')
    
    if os.path.exists(expert_path):
        with open(expert_path, 'r') as f:
            expert_report = json.load(f)
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    if os.path.exists(diag_path):
        with open(diag_path, 'r') as f:
            diagnostics = json.load(f)

    # Store results for report generation
    results_data = {
        'orig_results': orig_results,
        'cleaned_results': cleaned_results,
        'task_type': task_type,
        'selected_algo': selected_algo
    }
    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'ml_results.json'), 'w') as f:
        json.dump(results_data, f)
        
    return render_template('result.html', 
                           orig_results=orig_results, 
                           cleaned_results=cleaned_results, 
                           task_type=task_type,
                           selected_algo=selected_algo,
                           selected_metrics=selected_metrics,
                           expert_report=expert_report,
                           metadata=metadata,
                           diagnostics=diagnostics,
                           filename=metadata.get('filename') if metadata else 'dataset.csv',
                           trained=True)

@app.route('/download_cleaned')
@login_required
def download_cleaned():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_dataset.csv')
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name='cleaned_dataset.csv')
    flash('Cleaned dataset not found')
    return redirect(url_for('dashboard'))

@app.route('/download_report')
@login_required
def download_report():
    diag_path = os.path.join(app.config['UPLOAD_FOLDER'], 'diagnostics.json')
    res_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ml_results.json')
    expert_path = os.path.join(app.config['UPLOAD_FOLDER'], 'expert_report.json')
    
    if os.path.exists(diag_path) and os.path.exists(res_path) and os.path.exists(expert_path):
        with open(diag_path, 'r') as f:
            diagnostics = json.load(f)
        with open(res_path, 'r') as f:
            results_data = json.load(f)
        with open(expert_path, 'r') as f:
            expert_report = json.load(f)
            
        report_text = generate_text_report(
            diagnostics, 
            expert_report,
            results_data.get('orig_results', {}),
            results_data['cleaned_results'], 
            results_data['task_type'], 
            results_data['selected_algo']
        )
        
        report_path = os.path.join(app.config['UPLOAD_FOLDER'], 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
            
        return send_file(report_path, as_attachment=True, download_name='ML_Analysis_Report.txt')
    
    flash('Please perform analysis and train models before downloading the report.')
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    # Reduce memory usage for systems with limited RAM
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    app.run(debug=True)
