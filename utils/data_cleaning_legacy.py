raise RuntimeError(
    "Legacy cleaning module is deprecated and must not be used. "
    "Import from utils.adaptive_cleaning instead."
)

# --- Original legacy code below (kept for reference only) ---
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.utils import compute_sample_weight
from scipy import stats
import joblib
import os
from utils.column_detector import detect_column_types
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

ENCODER_PATH = os.path.join('uploads', 'encoder_mappings.pkl')


def _analyze_missing_patterns(df):
    """Analyze missing data patterns and provide insights."""
    missing_analysis = {
        'total_missing': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'column_missing': {},
        'row_missing': {},
        'missing_patterns': []
    }
    
    # Column-wise missing analysis
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        missing_analysis['column_missing'][col] = {
            'count': int(missing_count),
            'percentage': round(missing_pct, 2),
            'severity': 'High' if missing_pct > 50 else 'Medium' if missing_pct > 20 else 'Low'
        }
    
    # Row-wise missing analysis
    row_missing_counts = df.isnull().sum(axis=1)
    missing_analysis['row_missing'] = {
        'max_missing': int(row_missing_counts.max()) if len(row_missing_counts) > 0 else 0,
        'avg_missing': round(float(row_missing_counts.mean()), 2) if len(row_missing_counts) > 0 else 0.0,
        'complete_rows': int((row_missing_counts == 0).sum()),
        'partial_rows': int((row_missing_counts > 0).sum())
    }
    
    # Detect missing patterns (e.g., systematic missingness)
    if len(df) > 10:
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                # Check if missing is systematic (e.g., every nth row)
                missing_indices = df[df[col].isnull()].index
                if len(missing_indices) > 5:
                    gaps = np.diff(missing_indices)
                    if len(gaps) > 0 and len(np.unique(gaps)) == 1 and gaps[0] > 1:
                        missing_analysis['missing_patterns'].append({
                            'column': col,
                            'pattern': f'Systematic missing every {gaps[0]} rows',
                            'type': 'systematic'
                        })
    
    return missing_analysis

def _intelligent_imputation(series, strategy='auto', context_cols=None):
    """Intelligent imputation based on data characteristics."""
    if len(series.dropna()) == 0:
        return series.fillna(0 if pd.api.types.is_numeric_dtype(series) else 'Unknown')
    
    missing_pct = (series.isna().sum() / len(series)) * 100
    
    # Choose strategy based on missing percentage and data type
    if strategy == 'auto':
        if pd.api.types.is_numeric_dtype(series):
            if missing_pct > 30:
                strategy = 'median'
            elif missing_pct > 10:
                strategy = 'knn'
            else:
                strategy = 'iterative'
        else:
            if missing_pct > 50:
                strategy = 'constant'
            else:
                strategy = 'mode'
    
    try:
        if strategy == 'median':
            return series.fillna(series.median())
        elif strategy == 'mean':
            return series.fillna(series.mean())
        elif strategy == 'mode':
            mode_val = series.mode()
            return series.fillna(mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown')
        elif strategy == 'knn' and context_cols and len(context_cols) > 0:
            # Use KNN imputation with context
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=min(5, len(series.dropna())))
            if context_cols:
                # Create temporary dataframe with series and context
                temp_df = pd.concat([series] + context_cols, axis=1)
                imputed = imputer.fit_transform(temp_df)
                return pd.Series(imputed[:, 0], index=series.index)
        elif strategy == 'iterative' and context_cols and len(context_cols) > 0:
            # Use iterative imputation
            from sklearn.impute import IterativeImputer
            imputer = IterativeImputer(max_iter=10, random_state=42)
            temp_df = pd.concat([series] + context_cols, axis=1)
            imputed = imputer.fit_transform(temp_df)
            return pd.Series(imputed[:, 0], index=series.index)
        else:
            # Fallback to simple strategies
            if pd.api.types.is_numeric_dtype(series):
                return series.fillna(series.median())
            else:
                mode_val = series.mode()
                return series.fillna(mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown')
    except Exception:
        # Ultimate fallback
        return series.fillna(0 if pd.api.types.is_numeric_dtype(series) else 'Unknown')

def _detect_outliers_advanced(series, method='isolation_forest'):
    """Advanced outlier detection using multiple methods."""
    if len(series.dropna()) < 10:
        return np.array([False] * len(series))
    
    clean_series = series.dropna()
    outliers_mask = np.array([False] * len(series))
    
    try:
        if method == 'isolation_forest':
            iso = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso.fit_predict(clean_series.values.reshape(-1, 1))
            outliers_mask[clean_series.index] = outliers == -1
        elif method == 'iqr':
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_mask = (series < lower_bound) | (series > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(clean_series))
            outliers_mask[clean_series.index] = z_scores > 3
        elif method == 'local_outlier_factor':
            lof = LocalOutlierFactor(contamination=0.1)
            outliers = lof.fit_predict(clean_series.values.reshape(-1, 1))
            outliers_mask[clean_series.index] = outliers == -1
    except Exception:
        # Fallback to IQR method
        try:
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_mask = (series < lower_bound) | (series > upper_bound)
        except Exception:
            pass
    
    return outliers_mask

def _transform_data(series, transformation='auto'):
    """Apply appropriate data transformation."""
    if series.isna().all() or not pd.api.types.is_numeric_dtype(series):
        return series
    
    clean_series = series.dropna()
    if len(clean_series) < 5:
        return series
    
    # Auto-detect best transformation
    if transformation == 'auto':
        skewness = clean_series.skew()
        kurtosis = clean_series.kurtosis()
        
        if abs(skewness) > 2:
            transformation = 'log'
        elif abs(kurtosis) > 3:
            transformation = 'yeo-johnson'
        elif clean_series.min() > 0:
            transformation = 'log'
        else:
            transformation = 'none'
    
    try:
        if transformation == 'log':
            if series.min() > 0:
                return np.log1p(series)
            else:
                # Shift to make positive
                shifted = series - series.min() + 1
                return np.log1p(shifted)
        elif transformation == 'sqrt':
            if series.min() >= 0:
                return np.sqrt(series)
            else:
                shifted = series - series.min()
                return np.sqrt(shifted)
        elif transformation == 'box-cox':
            from scipy.stats import boxcox
            if series.min() > 0:
                transformed, _ = boxcox(series)
                return transformed
        elif transformation == 'yeo-johnson':
            from scipy.stats import yeojohnson
            transformed, _ = yeojohnson(series)
            return transformed
        else:
            return series
    except Exception:
        return series

def _normalize_data(series, method='standard'):
    """Normalize data using various methods."""
    if series.isna().all() or not pd.api.types.is_numeric_dtype(series):
        return series
    
    clean_series = series.dropna()
    if len(clean_series) < 2:
        return series.fillna(0)
    
    try:
        if method == 'standard':
            scaler = StandardScaler()
            scaled = scaler.fit_transform(clean_series.values.reshape(-1, 1))
            result = series.copy()
            result[clean_series.index] = scaled.flatten()
            return result.fillna(0)
        elif method == 'minmax':
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(clean_series.values.reshape(-1, 1))
            result = series.copy()
            result[clean_series.index] = scaled.flatten()
            return result.fillna(0)
        elif method == 'robust':
            scaler = RobustScaler()
            scaled = scaler.fit_transform(clean_series.values.reshape(-1, 1))
            result = series.copy()
            result[clean_series.index] = scaled.flatten()
            return result.fillna(0)
        else:
            return series
    except Exception:
        return series.fillna(0)

def _assess_data_quality(df):
    """Comprehensive data quality assessment."""
    quality_report = {
        'overall_score': 0,
        'completeness': 0,
        'consistency': 0,
        'validity': 0,
        'issues': [],
        'recommendations': []
    }
    
    # Completeness assessment
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    quality_report['completeness'] = round(((total_cells - missing_cells) / total_cells) * 100, 2)
    
    # Consistency assessment (duplicate rows, inconsistent formats)
    duplicate_rows = df.duplicated().sum()
    quality_report['consistency'] = round(((len(df) - duplicate_rows) / len(df)) * 100, 2)
    
    # Validity assessment (data type consistency, value ranges)
    validity_issues = 0
    total_checks = 0
    
    for col in df.columns:
        total_checks += 1
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check for negative values where inappropriate
            if 'age' in col.lower() and len(df[df[col] < 0]) > 0:
                validity_issues += 1
                quality_report['issues'].append(f"Negative values in {col}")
            # Check for unreasonable outliers
            if df[col].std() / df[col].mean() > 5 if df[col].mean() != 0 else False:
                validity_issues += 1
                quality_report['issues'].append(f"High variance in {col}")
        elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
            # Check for mixed case in categorical data
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) < 20:  # Likely categorical
                case_variations = set()
                for val in unique_vals:
                    if isinstance(val, str):
                        case_variations.add(val.lower())
                if len(case_variations) < len(unique_vals):
                    validity_issues += 1
                    quality_report['issues'].append(f"Case inconsistencies in {col}")
    
    quality_report['validity'] = round(((total_checks - validity_issues) / total_checks) * 100, 2)
    
    # Overall score
    quality_report['overall_score'] = round(
            (quality_report['completeness'] * 0.4 + 
             quality_report['consistency'] * 0.3 + 
             quality_report['validity'] * 0.3), 2
        )
    
    # Generate recommendations
    if quality_report['completeness'] < 80:
        quality_report['recommendations'].append("Consider advanced imputation strategies")
    if quality_report['consistency'] < 90:
        quality_report['recommendations'].append("Remove duplicate rows and standardize formats")
    if quality_report['validity'] < 85:
        quality_report['recommendations'].append("Validate data ranges and formats")
    
    return quality_report

def _safe_impute_numeric(series, strategy='median', default_value=0):
    """Safely impute numeric series with robust fallback."""
    try:
        if series.isna().all():
            # If all values are NaN, fill with default
            return series.fillna(default_value)
        
        # Handle infinity values first
        series_clean = series.copy()
        series_clean = series_clean.replace([np.inf, -np.inf], np.nan)
        
        if strategy == 'median':
            median_val = series_clean.median()
            if pd.isna(median_val):
                return series_clean.fillna(default_value)
            return series_clean.fillna(median_val)
        elif strategy == 'mean':
            mean_val = series_clean.mean()
            if pd.isna(mean_val):
                return series_clean.fillna(default_value)
            return series_clean.fillna(mean_val)
        else:
            return series_clean.fillna(default_value)
    except Exception:
        return series.fillna(default_value)

def _safe_impute_categorical(series, default_value='Unknown'):
    """Safely impute categorical series with robust fallback."""
    try:
        if series.isna().all():
            return series.fillna(default_value)
        
        # Convert empty strings to NaN first
        series_clean = series.copy()
        series_clean = series_clean.replace(['', ' ', '  ', '   ', '    '], np.nan)
        
        mode_val = series_clean.mode()
        if len(mode_val) == 0 or pd.isna(mode_val.iloc[0]):
            return series_clean.fillna(default_value)
        return series_clean.fillna(mode_val.iloc[0])
    except Exception:
        return series.fillna(default_value)

def _validate_dataframe_shape(df, operation_name="operation"):
    """Validate DataFrame shape and handle edge cases."""
    if df is None or df.empty:
        raise ValueError(f"DataFrame is empty before {operation_name}")
    
    if len(df.columns) == 0:
        raise ValueError(f"DataFrame has no columns before {operation_name}")
    
    if len(df) == 0:
        raise ValueError(f"DataFrame has no rows before {operation_name}")
    
    # Check for columns with all NaN values
    empty_cols = [col for col in df.columns if len(df[col].dropna()) == 0]
    if empty_cols:
        print(f"Warning: Columns with all NaN values detected: {empty_cols}")
    
    return True

def _safe_dataframe_creation(data, columns, index, operation_name="DataFrame creation"):
    """Safely create DataFrame with shape validation."""
    try:
        if data is None or len(data) == 0:
            raise ValueError(f"No data provided for {operation_name}")
        
        if columns is None or len(columns) == 0:
            raise ValueError(f"No columns provided for {operation_name}")
        
        # Handle numpy arrays and lists
        if hasattr(data, 'shape'):
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            elif data.shape[1] == 0:
                raise ValueError(f"Data has 0 columns for {operation_name}")
        
        # Ensure data and columns match
        if hasattr(data, 'shape') and data.shape[1] != len(columns):
            raise ValueError(f"Shape mismatch: data has {data.shape[1]} columns but {len(columns)} column names provided for {operation_name}")
        
        df = pd.DataFrame(data, columns=columns, index=index)
        _validate_dataframe_shape(df, operation_name)
        return df
        
    except Exception as e:
        raise ValueError(f"Failed to create DataFrame for {operation_name}: {str(e)}")

def _json_safe(obj):
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(obj, dict):
        return {_json_safe(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _default_cleaning_policy(mode='gentle'):
    base = {
        'mode': mode,
        'drop_identifier_columns': False,
        'drop_leakage_columns': False,
        'drop_high_missing_columns': False,
        'missing_column_threshold': 0.98,
        'remove_duplicates': False,
        'handle_outliers': False,
        'outlier_percentile': 95,           # Winsorize at real 95th percentile boundary
        'outlier_method': 'isolation_forest',  # Advanced outlier detection method
        'outlier_action': 'cap',            # 'cap' or 'remove'
        'convert_mixed_numeric_text': False,
        'encode_features': False,
        'apply_log_transform': False,       # Log-transform skewed numeric features
        'apply_transformation': False,       # Advanced data transformation
        'transformation_method': 'auto',     # auto, log, sqrt, box-cox, yeo-johnson
        'apply_normalization': False,       # Data normalization
        'normalization_method': 'standard',  # standard, minmax, robust
        'skew_threshold': 1.0,              # |skewness| above this triggers log transform
        'numeric_imputation': 'auto',       # Auto-adaptive imputation
        # No SMOTE: imbalance handled via sample_weight in training
    }
    if mode == 'balanced':
        base.update({
            'missing_column_threshold': 0.95,
            'encode_features': True,
            'handle_outliers': True,
            'apply_log_transform': True,
            'apply_transformation': True,
            'apply_normalization': True,
            'numeric_imputation': 'auto',
        })
    elif mode == 'aggressive':
        base.update({
            'missing_column_threshold': 0.90,
            'encode_features': True,
            'handle_outliers': True,
            'outlier_action': 'remove',
            'apply_log_transform': True,
            'apply_transformation': True,
            'apply_normalization': True,
            'numeric_imputation': 'auto',
        })
    return base


def clean_dataset(df, leakage_cols=None, target_col=None, fit_encoders=True,
                  preserve_structure=False, cleaning_policy=None,
                  return_report=False, is_inference=False):
    """
    Non-destructive Repair-In-Place cleaning pipeline.

    Strategy:
      - Missing numeric  → KNN Imputation (from real neighbours)
      - Missing text     → Mode imputation (most frequent real value)
      - Outliers         → Winsorization (capped at real 95th percentile boundary)
      - Skewed features  → Log transformation (reversible)
      - Imbalance        → sample_weight computed and returned (no SMOTE)
      - Label noise      → Cleanlab relabelling (row kept, only label fixed)
      - Redundant cols   → Flagged, kept (regularization handles them)

    Returns:
        cleaned_df               if return_report=False
        (cleaned_df, report)     if return_report=True
        report includes 'sample_weights' array when class imbalance found.
    """
    policy = _default_cleaning_policy('gentle' if preserve_structure else 'balanced')
    if cleaning_policy:
        policy.update(cleaning_policy)

    original_df = df.copy()
    cleaned_df = df.copy()

    # Per-cell repair log: list of {column, row_index, before, after, reason}
    cell_repairs = []

    cleaning_report = {
        'policy': policy,
        'actions': [],
        'cell_repairs': cell_repairs,      # ← granular audit trail
        'sample_weights': None,            # filled later if imbalance detected
        'before': {
            'rows': int(len(original_df)),
            'columns': int(len(original_df.columns)),
            'missing_total': int(original_df.isnull().sum().sum())
        },
        'after': {}
    }

    # --- Validate initial DataFrame ---
    _validate_dataframe_shape(cleaned_df, "data cleaning start")
    
    # --- Advanced Data Quality Assessment ---
    quality_assessment = _assess_data_quality(cleaned_df)
    cleaning_report['data_quality'] = quality_assessment
    
    # --- Advanced Missing Data Analysis ---
    missing_analysis = _analyze_missing_patterns(cleaned_df)
    cleaning_report['missing_analysis'] = missing_analysis
    
    # --- Sanitize column names ---
    cleaned_df.columns = cleaned_df.columns.str.strip()
    
    # --- Handle infinity values early ---
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Replace infinities with NaN first, then they'll be handled by imputation
            cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], np.nan)
    
    # Remove completely empty columns (but fill them instead of dropping for preserve_structure)
    empty_cols = [col for col in cleaned_df.columns if cleaned_df[col].isna().all()]
    if empty_cols:
        if preserve_structure:
            # For preserve_structure, fill empty columns instead of dropping
            for col in empty_cols:
                if cleaned_df[col].dtype in ['object', 'category', 'string']:
                    cleaned_df[col] = 'Unknown'
                else:
                    cleaned_df[col] = 0
            cleaning_report['actions'].append({
                'step': 'fill_empty_columns',
                'columns': empty_cols,
                'reason': 'All values were NaN - filled with defaults'
            })
        else:
            # For ML pipeline, drop empty columns
            cleaned_df = cleaned_df.drop(columns=empty_cols)
            cleaning_report['actions'].append({
                'step': 'drop_empty_columns',
                'columns': empty_cols,
                'reason': 'All values were NaN'
            })

    # --- Drop fully-empty rows (not a data row, just blank lines) ---
    before_rows = len(cleaned_df)
    cleaned_df.dropna(how='all', inplace=True)
    if len(cleaned_df) < before_rows:
        cleaning_report['actions'].append({
            'step': 'drop_all_null_rows',
            'rows_removed': before_rows - len(cleaned_df)
        })

    # --- Auto-detect target column ---
    if not is_inference:
        if target_col is None:
            target_col = cleaned_df.columns[-1]
        target_col = target_col.strip()
        if target_col not in cleaned_df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found. "
                f"Available: {list(cleaned_df.columns)}"
            )
    else:
        target_col = "__INFERENCE_NO_TARGET__"

    col_types = detect_column_types(cleaned_df, target_col)
    cleaning_report['quarantined_columns'] = []

    # --- BLOCK 0 & 1: FLAG (not drop) IDENTIFIERS & LEAKAGE ---
    identifiers_to_flag = [c for c in col_types.get('identifiers', []) if c != target_col]
    if identifiers_to_flag:
        cleaning_report['actions'].append({
            'step': 'flag_identifier_columns',
            'columns': identifiers_to_flag,
            'action': 'quarantined'
        })
        cleaning_report['quarantined_columns'].extend(identifiers_to_flag)

    if leakage_cols:
        leakage_to_flag = [c for c in leakage_cols
                           if c in cleaned_df.columns and c != target_col]
        if leakage_to_flag:
            cleaning_report['actions'].append({
                'step': 'flag_leakage_columns',
                'columns': leakage_to_flag,
                'action': 'quarantined'
            })
            cleaning_report['quarantined_columns'].extend(leakage_to_flag)

    cleaning_report['quarantined_columns'] = list(set(cleaning_report['quarantined_columns']))

    # --- BLOCK 2: FLAG HIGH-MISSING COLUMNS (no drop) ---
    missing_threshold = float(policy.get('missing_column_threshold', 0.95))
    high_missing_cols = []
    for col in cleaned_df.columns:
        if col == target_col:
            continue
        ratio = cleaned_df[col].isnull().sum() / max(len(cleaned_df), 1)
        if ratio > missing_threshold:
            high_missing_cols.append(col)
    if high_missing_cols:
        cleaning_report['actions'].append({
            'step': 'flag_high_missing_columns',
            'columns': high_missing_cols,
            'threshold': missing_threshold,
            'action': 'imputed_despite_high_missing'
        })

    # --- BLOCK 3: PARSE DATETIME COLUMNS ---
    for col in col_types['datetime_cols']:
        if col not in cleaned_df.columns:
            continue
        
        # Convert to datetime with error handling
        cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
        
        if preserve_structure:
            # For preserve_structure, format as string
            cleaned_df[col] = cleaned_df[col].dt.strftime('%d-%m-%Y')
        else:
            # For ML pipeline, create safe datetime components
            # Only create components if conversion was successful
            if cleaned_df[col].notna().any():
                # Create datetime components with safe handling for NaT values
                cleaned_df[f'{col}_year'] = cleaned_df[col].dt.year.fillna(0).astype(int)
                cleaned_df[f'{col}_month'] = cleaned_df[col].dt.month.fillna(1).astype(int)
                cleaned_df[f'{col}_day'] = cleaned_df[col].dt.day.fillna(1).astype(int)
                cleaned_df[f'{col}_dayofweek'] = cleaned_df[col].dt.dayofweek.fillna(0).astype(int)
            else:
                # If all values are NaT, create default columns
                cleaned_df[f'{col}_year'] = 0
                cleaned_df[f'{col}_month'] = 1
                cleaned_df[f'{col}_day'] = 1
                cleaned_df[f'{col}_dayofweek'] = 0
    if col_types['datetime_cols']:
        cleaning_report['actions'].append({
            'step': 'datetime_normalization',
            'columns': [c for c in col_types['datetime_cols'] if c in original_df.columns]
        })

    # --- BLOCK 3.5: FORMAT NAME COLUMNS ---
    for col in cleaned_df.select_dtypes(include=['object', 'string']).columns:
        if 'name' in col.lower():
            cleaned_df[col] = cleaned_df[col].str.title()

    # --- BLOCK 4: MIXED-TYPE COERCION ---
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object' and policy.get('convert_mixed_numeric_text'):
            values = cleaned_df[col].astype(str)
            numeric_mask = values.str.match(r'^-?\d*\.?\d+$')
            if numeric_mask.any() and (~numeric_mask).any():
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                cleaning_report['actions'].append({
                    'step': 'mixed_type_coercion', 'column': col
                })

    # --- BLOCK 5: IMPUTATION (Repair-In-Place) ---
    numeric_cols = cleaned_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    categorical_cols = [c for c in categorical_cols if c != target_col]
    
    # Filter out columns that were already handled as empty
    categorical_cols = [c for c in categorical_cols if c not in empty_cols and len(cleaned_df[c].dropna()) > 0]

    # 5a. Numeric → Intelligent Imputation (auto-strategy based on data characteristics)
    if numeric_cols:
        missing_before = {c: cleaned_df[c].isnull().sum() for c in numeric_cols}
        
        # Use intelligent imputation for each column
        for col in numeric_cols:
            if missing_before[col] > 0:
                # Get context columns for intelligent imputation
                context_cols = [c for c in numeric_cols if c != col and missing_before[c] == 0 and len(cleaned_df[c].dropna()) > 0]
                context_series = [cleaned_df[c] for c in context_cols[:3]]  # Limit to 3 context columns
                
                # Apply intelligent imputation
                imputed_series = _intelligent_imputation(
                    cleaned_df[col], 
                    strategy=policy.get('numeric_imputation', 'auto'),
                    context_cols=context_series
                )
                
                # Record repairs
                null_rows = cleaned_df[col][cleaned_df[col].isnull()].index
                for idx in null_rows:
                    if idx < len(cleaned_df):
                        before_val = cleaned_df[col].iloc[idx] if idx < len(cleaned_df[col]) else None
                        after_val = imputed_series.iloc[idx] if idx < len(imputed_series) else None
                        cell_repairs.append({
                            'column': col,
                            'row_index': int(idx),
                            'before': None,
                            'after': round(float(after_val), 4) if pd.api.types.is_numeric_dtype(type(after_val)) else str(after_val),
                            'reason': f'intelligent imputation (auto-strategy) with {len(context_series)} context columns'
                        })
                
                cleaned_df[col] = imputed_series

        cleaning_report['actions'].append({
            'step': 'intelligent_numeric_imputation',
            'strategy': 'auto-adaptive',
            'columns': numeric_cols,
            'cells_repaired': sum(missing_before.values())
        })

    # 5b. Categorical → Intelligent Imputation (context-aware)
    if categorical_cols:
        missing_before_cat = {c: cleaned_df[c].isnull().sum() for c in categorical_cols}
        
        # Use intelligent imputation for each categorical column
        for col in categorical_cols:
            if missing_before_cat[col] > 0:
                # Apply intelligent categorical imputation
                imputed_series = _intelligent_imputation(
                    cleaned_df[col], 
                    strategy='auto',
                    context_cols=None  # Categorical doesn't use numeric context
                )
                
                # Record repairs
                null_rows = cleaned_df[col][cleaned_df[col].isnull()].index
                for idx in null_rows:
                    if idx < len(cleaned_df):
                        before_val = cleaned_df[col].iloc[idx] if idx < len(cleaned_df[col]) else None
                        after_val = imputed_series.iloc[idx] if idx < len(imputed_series) else None
                        cell_repairs.append({
                            'column': col,
                            'row_index': int(idx),
                            'before': None,
                            'after': str(after_val) if after_val is not None and not pd.isna(after_val) else 'Unknown',
                            'reason': 'intelligent categorical imputation (auto-strategy)'
                        })
                
                cleaned_df[col] = imputed_series

        cleaning_report['actions'].append({
            'step': 'intelligent_categorical_imputation',
            'strategy': 'auto-adaptive',
            'columns': categorical_cols,
            'cells_repaired': sum(missing_before_cat.values())
        })

    # --- BLOCK 6: ADVANCED OUTLIER DETECTION AND HANDLING ---
    if policy.get('handle_outliers'):
        outlier_method = policy.get('outlier_method', 'isolation_forest')
        outlier_log = []
        
        for col in numeric_cols:
            if cleaned_df[col].notna().sum() > 10:  # Only process columns with sufficient data
                # Detect outliers using advanced methods
                outliers_mask = _detect_outliers_advanced(cleaned_df[col], method=outlier_method)
                
                if outliers_mask.any():
                    old_vals = cleaned_df[col].copy()
                    
                    # Handle outliers based on policy
                    if policy.get('outlier_action', 'cap') == 'cap':
                        # Winsorization (capping)
                        pct = float(policy.get('outlier_percentile', 95))
                        low_pct = 100 - pct
                        lower_bound = np.percentile(cleaned_df[col].dropna(), low_pct)
                        upper_bound = np.percentile(cleaned_df[col].dropna(), pct)
                        cleaned_df[col] = np.clip(cleaned_df[col], lower_bound, upper_bound)
                        
                        for idx in np.where(outliers_mask)[0]:
                            if idx < len(old_vals):  # Ensure valid index
                                before_val = old_vals.iloc[idx] if idx < len(old_vals) else None
                                after_val = cleaned_df.iloc[idx] if idx < len(cleaned_df) else None
                                if (before_val is not None and after_val is not None and 
                    not isinstance(before_val, pd.Series) and not isinstance(after_val, pd.Series) and
                    not pd.isna(before_val) and not pd.isna(after_val)):
                                    cell_repairs.append({
                                        'column': col,
                                        'row_index': int(idx),
                                        'before': round(float(before_val), 4) if pd.api.types.is_numeric_dtype(type(before_val)) else str(before_val),
                                        'after': round(float(after_val), 4) if pd.api.types.is_numeric_dtype(type(after_val)) else str(after_val),
                                        'reason': f'outlier capped: {outlier_method} detection, winsorized at {pct}th percentile'
                                    })
                    else:
                        # Remove outliers (set to median)
                        median_val = cleaned_df[col].median()
                        cleaned_df.loc[outliers_mask, col] = median_val
                        
                        for idx in np.where(outliers_mask)[0]:
                            if idx < len(old_vals):  # Ensure valid index
                                before_val = old_vals.iloc[idx] if idx < len(old_vals) else None
                                if before_val is not None and not pd.isna(before_val):
                                    cell_repairs.append({
                                        'column': col,
                                        'row_index': int(idx),
                                        'before': round(float(before_val), 4) if pd.api.types.is_numeric_dtype(type(before_val)) else str(before_val),
                                        'after': round(float(median_val), 4),
                                        'reason': f'outlier replaced: {outlier_method} detection, set to median'
                                    })
                    
                    outlier_log.append({
                        'column': col,
                        'method': outlier_method,
                        'outliers_detected': int(outliers_mask.sum()),
                        'action': policy.get('outlier_action', 'cap')
                    })
        
        if outlier_log:
            cleaning_report['actions'].append({
                'step': 'advanced_outlier_handling',
                'method': outlier_method,
                'action': policy.get('outlier_action', 'cap'),
                'columns': outlier_log
            })

    # --- BLOCK 6.5: DATA TRANSFORMATION ---
    if policy.get('apply_transformation', False):
        transformation_log = []
        for col in numeric_cols:
            # Skip datetime component columns from transformation
            if any(col.endswith(suffix) for suffix in ['_year', '_month', '_day', '_dayofweek']):
                continue
                
            if cleaned_df[col].notna().sum() > 5:  # Only transform columns with sufficient data
                old_vals = cleaned_df[col].copy()
                transformed_series = _transform_data(
                    cleaned_df[col], 
                    transformation=policy.get('transformation_method', 'auto')
                )
                
                # Check if transformation was applied
                if not transformed_series.equals(old_vals.fillna(old_vals)):
                    cleaned_df[col] = transformed_series
                    orig_skew = old_vals.dropna().skew()
                    trans_skew = transformed_series.dropna().skew()
                    transformation_log.append({
                        'column': col,
                        'method': 'auto-detected',
                        'original_skew': round(float(orig_skew), 4) if orig_skew is not None and not pd.isna(orig_skew) else 0.0,
                        'transformed_skew': round(float(trans_skew), 4) if trans_skew is not None and not pd.isna(trans_skew) else 0.0
                    })
        
        if transformation_log:
            cleaning_report['actions'].append({
                'step': 'data_transformation',
                'method': 'auto-detection',
                'columns': transformation_log
            })

    # --- BLOCK 6.6: DATA NORMALIZATION ---
    if policy.get('apply_normalization', False):
        normalization_method = policy.get('normalization_method', 'standard')
        normalization_log = []
        
        for col in numeric_cols:
            # Skip datetime component columns from normalization
            if any(col.endswith(suffix) for suffix in ['_year', '_month', '_day', '_dayofweek']):
                continue
                
            if cleaned_df[col].notna().sum() > 2:  # Only normalize columns with sufficient data
                old_vals = cleaned_df[col].copy()
                normalized_series = _normalize_data(cleaned_df[col], method=normalization_method)
                
                # Check if normalization was applied
                if not normalized_series.equals(old_vals.fillna(old_vals)):
                    cleaned_df[col] = normalized_series
                    orig_mean = old_vals.mean()
                    orig_std = old_vals.std()
                    norm_mean = normalized_series.mean()
                    norm_std = normalized_series.std()
                    normalization_log.append({
                        'column': col,
                        'method': normalization_method,
                        'original_mean': round(float(orig_mean), 4) if orig_mean is not None and not pd.isna(orig_mean) else 0.0,
                        'original_std': round(float(orig_std), 4) if orig_std is not None and not pd.isna(orig_std) else 0.0,
                        'normalized_mean': round(float(norm_mean), 4) if norm_mean is not None and not pd.isna(norm_mean) else 0.0,
                        'normalized_std': round(float(norm_std), 4) if norm_std is not None and not pd.isna(norm_std) else 0.0
                    })
        
        if normalization_log:
            cleaning_report['actions'].append({
                'step': 'data_normalization',
                'method': normalization_method,
                'columns': normalization_log
            })

    # --- BLOCK 7: LOG TRANSFORMATION (for skewed numeric features) ---
    log_transformed_cols = []
    if policy.get('apply_log_transform'):
        skew_threshold = float(policy.get('skew_threshold', 1.0))
        for col in numeric_cols:
            col_data = cleaned_df[col].dropna()
            if len(col_data) < 4:
                continue
            skewness = float(col_data.skew())
            # Only log-transform if positive skew and all values > 0
            if abs(skewness) > skew_threshold and col_data.min() > 0:
                original_vals = cleaned_df[col].copy()
                changed = original_vals != np.log1p(cleaned_df[col])
                cleaned_df[col] = np.log1p(cleaned_df[col])
                log_transformed_cols.append({
                    'column': col,
                    'original_skew': round(skewness, 4),
                    'new_skew': round(float(cleaned_df[col].skew()), 4),
                    'transform': 'log1p',
                    'reversible': True
                })
                # Sample repair entries (first 3 changed cells for readability)
                for idx in original_vals[changed].index:
                    if idx < len(cleaned_df):
                        before_val = original_vals.iloc[idx]
                        after_val = cleaned_df.at[idx, col]
                        cell_repairs.append({
                            'column': col,
                            'row_index': int(idx),
                            'before': round(float(before_val), 4) if pd.api.types.is_numeric_dtype(type(before_val)) else str(before_val),
                            'after': round(float(after_val), 4) if pd.api.types.is_numeric_dtype(type(after_val)) else str(after_val),
                            'reason': f'winsorized: capped at real {pct}th percentile boundary ({upper_bound:.2f})'
                        })
        if log_transformed_cols:
            cleaning_report['actions'].append({
                'step': 'log_transformation',
                'columns': log_transformed_cols,
                'note': 'Reverse transform: original = exp(value) - 1'
            })

    # --- BLOCK 8: EARLY RETURN for human-readable / preserve_structure output ---
    if preserve_structure or not policy.get('encode_features'):
        # Enhanced missing value handling for preserve_structure path
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype in ['object', 'category', 'string']:
                # Fill categorical/string columns with 'Unknown' 
                if cleaned_df[col].isna().any():
                    cleaned_df[col] = _safe_impute_categorical(cleaned_df[col], 'Unknown')
            else:
                # Fill numeric columns with 0
                if cleaned_df[col].isna().any():
                    cleaned_df[col] = _safe_impute_numeric(cleaned_df[col], 'median', 0)
        
        # Replace infinities and ensure no remaining NaNs
        cleaned_df.replace([np.inf, -np.inf], 0, inplace=True)
        cleaned_df.fillna(0, inplace=True)
        
        cleaning_report['after'] = {
            'rows': int(len(cleaned_df)),
            'columns': int(len(cleaned_df.columns)),
            'missing_total': int(cleaned_df.isnull().sum().sum()),
            'columns_removed': sorted(
                list(set(original_df.columns) - set(cleaned_df.columns)))
        }
        # Compute sample weights even for human-readable output
        if not is_inference and target_col in cleaned_df.columns:
            _attach_sample_weights(cleaned_df, target_col, cleaning_report)
        if return_report:
            return cleaned_df, _json_safe(cleaning_report)
        return cleaned_df

    # --- Validate before encoding ---
    _validate_dataframe_shape(cleaned_df, "encoding phase")
    
    # --- BLOCK 9: ENCODE CATEGORICAL DATA (ML-ready path) ---
    encoder_mappings = {}

    nominal_cols = [c for c in col_types['nominal_categorical']
                    if c in cleaned_df.columns and c != target_col]
    high_cardinality_cols = []
    safe_nominal_cols = []
    for col in nominal_cols:
        if cleaned_df[col].nunique() > 50:
            high_cardinality_cols.append(col)
        else:
            safe_nominal_cols.append(col)

    if safe_nominal_cols:
        try:
            if fit_encoders:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_data = ohe.fit_transform(cleaned_df[safe_nominal_cols].astype(str))
                encoder_mappings['nominal'] = {'encoder': ohe, 'columns': safe_nominal_cols}
            else:
                all_mappings = joblib.load(ENCODER_PATH)
                ohe = all_mappings['nominal']['encoder']
                encoded_data = ohe.transform(cleaned_df[safe_nominal_cols].astype(str))
            ohe_columns = ohe.get_feature_names_out(safe_nominal_cols)
            
            # Validate encoding result
            if encoded_data.shape[1] == 0:
                print(f"Warning: One-hot encoding produced no columns for {safe_nominal_cols}")
            else:
                encoded_df = _safe_dataframe_creation(
                    encoded_data, ohe_columns, cleaned_df.index,
                    "one-hot encoding"
                )
                
                # Validate before concatenation
                _validate_dataframe_shape(encoded_df, "concatenation")
                _validate_dataframe_shape(cleaned_df, "concatenation")
                
                cleaned_df = pd.concat([cleaned_df, encoded_df], axis=1)
        except Exception as e:
            print(f"Warning: One-hot encoding failed for {safe_nominal_cols}: {e}")

    if high_cardinality_cols:
        encoder_mappings['high_cardinality'] = {
            'columns': high_cardinality_cols, 'mappings': {}
        }
        for col in high_cardinality_cols:
            freq_map = cleaned_df[col].value_counts(normalize=True)
            encoder_mappings['high_cardinality']['mappings'][col] = freq_map.to_dict()
            cleaned_df[col] = cleaned_df[col].map(freq_map).fillna(0)

    ordinal_cols = [c for c in col_types['ordinal_categorical']
                    if c in cleaned_df.columns and c != target_col]
    if ordinal_cols:
        if fit_encoders:
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            cleaned_df[ordinal_cols] = oe.fit_transform(
                cleaned_df[ordinal_cols].astype(str))
            encoder_mappings['ordinal'] = {'encoder': oe, 'columns': ordinal_cols}
        else:
            all_mappings = joblib.load(ENCODER_PATH)
            oe = all_mappings['ordinal']['encoder']
            cleaned_df[ordinal_cols] = oe.transform(
                cleaned_df[ordinal_cols].astype(str))

    # Target encoding (LabelEncoder ONLY for target column)
    if not is_inference:
        if target_col not in cleaned_df.columns:
            raise ValueError(
                f"Target column '{target_col}' was removed during cleaning.")
        if (cleaned_df[target_col].dtype == 'object' or
                pd.api.types.is_categorical_dtype(cleaned_df[target_col])):
            le = LabelEncoder()
            cleaned_df[target_col] = le.fit_transform(
                cleaned_df[target_col].astype(str))
            encoder_mappings['target'] = {
                'encoder': le, 'classes': le.classes_.tolist()
            }

    # --- BLOCK 10: LABEL NOISE CORRECTION (Cleanlab) ---
    if (not is_inference and not preserve_structure
            and pd.api.types.is_numeric_dtype(cleaned_df[target_col])):
        num_classes = cleaned_df[target_col].nunique()
        if 1 < num_classes < 20:
            try:
                from cleanlab.classification import CleanLearning
                from sklearn.ensemble import RandomForestClassifier
                X_cl = cleaned_df.drop(
                    columns=[target_col] + cleaning_report['quarantined_columns'],
                    errors='ignore')
                X_cl_enc = pd.get_dummies(X_cl, dummy_na=False).select_dtypes(
                    include=np.number)
                if not X_cl_enc.empty:
                    y_cl = cleaned_df[target_col]
                    clf = RandomForestClassifier(
                        n_estimators=50, max_depth=5, random_state=42)
                    cl = CleanLearning(clf)
                    label_issues = cl.find_label_issues(
                        X_cl_enc.values, y_cl.values)
                    bad_mask = label_issues['is_label_issue']
                    if bad_mask.any():
                        corrected_y = cl.fit(
                            X_cl_enc.values, y_cl.values).predict(X_cl_enc.values)
                for i, idx in enumerate(cleaned_df.index[bad_mask]):
                    if i < len(corrected_y) and idx < len(cleaned_df):
                        before_val = cleaned_df.at[idx, target_col]
                        after_val = corrected_y[i]
                        cell_repairs.append({
                            'column': target_col,
                            'row_index': int(idx),
                            'before': int(before_val) if before_val is not None and not pd.isna(before_val) else 0,
                            'after': int(after_val) if after_val is not None and not pd.isna(after_val) else 0,
                            'reason': 'cleanlab label noise correction'
                        })
                cleaned_df.loc[bad_mask, target_col] = corrected_y[:len(bad_mask)]
                cleaning_report['actions'].append({
                    'step': 'label_noise_correction',
                    'rows_corrected': int(bad_mask.sum())
                })
            except (ImportError, Exception):
                pass

    # --- BLOCK 11: CLASS IMBALANCE → Sample Weights (no SMOTE) ---
    if not is_inference and target_col in cleaned_df.columns:
        _attach_sample_weights(cleaned_df, target_col, cleaning_report)

    # --- BLOCK 12: FLAG REDUNDANT / CORRELATED FEATURES ---
    current_numeric = cleaned_df.select_dtypes(include=np.number).columns.tolist()
    current_numeric = [c for c in current_numeric if c != target_col]
    if len(current_numeric) > 1:
        try:
            corr_matrix = cleaned_df[current_numeric].corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            redundant = [c for c in upper.columns if any(upper[c] > 0.95)]
            if redundant:
                cleaning_report['actions'].append({
                    'step': 'flag_redundant_features',
                    'columns': redundant,
                    'action': 'kept — model regularization will handle them'
                })
        except Exception:
            pass

    # --- Save encoders & imputers ---
    if fit_encoders:
        if 'imputer_num' in dir():
            encoder_mappings['imputer_num'] = imputer_num
        if 'imputer_const' in dir():
            encoder_mappings['imputer_const'] = imputer_const
        encoder_mappings_info = {
            k: {'columns': v.get('columns', []), 'classes': v.get('classes', [])}
            for k, v in encoder_mappings.items()
            if k not in ('imputer_num', 'imputer_const')
        }
        joblib.dump(encoder_mappings, ENCODER_PATH)
        joblib.dump(encoder_mappings_info, ENCODER_PATH.replace('.pkl', '_info.pkl'))

    # --- Final validation and cleanup ---
    _validate_dataframe_shape(cleaned_df, "final cleanup")
    
    # Enhanced missing value handling - fill any remaining NaNs with appropriate defaults
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype in ['object', 'category', 'string']:
            # Fill categorical/string columns with 'Unknown' or 'Missing'
            if cleaned_df[col].isna().any():
                cleaned_df[col] = _safe_impute_categorical(cleaned_df[col], 'Unknown')
        else:
            # Fill numeric columns with 0
            if cleaned_df[col].isna().any():
                cleaned_df[col] = _safe_impute_numeric(cleaned_df[col], 'median', 0)
    
    # Replace infinities and ensure no remaining NaNs
    cleaned_df.replace([np.inf, -np.inf], 0, inplace=True)
    cleaned_df.fillna(0, inplace=True)
    cleaning_report['after'] = {
        'rows': int(len(cleaned_df)),
        'columns': int(len(cleaned_df.columns)),
        'missing_total': int(cleaned_df.isnull().sum().sum()),
        'columns_removed': sorted(
            list(set(original_df.columns) - set(cleaned_df.columns)))
    }
    if return_report:
        return cleaned_df, _json_safe(cleaning_report)
    return cleaned_df


def _attach_sample_weights(cleaned_df, target_col, cleaning_report):
    """
    Compute class-balanced sample weights so the model compensates for
    imbalance without fabricating synthetic rows.
    Weight formula: n_samples / (n_classes * class_count)
    """
    try:
        y = cleaned_df[target_col]
        if not pd.api.types.is_numeric_dtype(y):
            return
        num_classes = y.nunique()
        if num_classes < 2 or num_classes > 50:
            return
        class_counts = y.value_counts()
        imbalance_ratio = class_counts.min() / class_counts.max()
        if imbalance_ratio < 0.8:   # Only apply when imbalance is meaningful
            weights = compute_sample_weight(class_weight='balanced', y=y.values)
            cleaning_report['sample_weights'] = weights.tolist()
            cleaning_report['actions'].append({
                'step': 'sample_weight_computation',
                'method': 'compute_sample_weight(balanced)',
                'imbalance_ratio': round(float(imbalance_ratio), 4),
                'class_distribution': {
                    int(k): int(v) for k, v in class_counts.items()
                },
                'note': (
                    'No synthetic rows added. Pass sample_weights to '
                    'model.fit(X, y, sample_weight=weights) during training.'
                )
            })
    except Exception:
        pass
