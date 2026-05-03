import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.column_detector import detect_column_types

try:
    from cleanlab.filter import find_label_issues
except Exception:
    find_label_issues = None

def perform_diagnostics(df):
    """
    Analyzes the dataset and returns a dictionary of diagnostics.
    This function performs a deep health check on the data to find issues.
    """
    # Initialize a dictionary to store all our findings
    diagnostics = {}
    
    # FIX: Sanitize column names
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    rows, cols = df.shape
    is_large_dataset = rows > 5000
    # We assume the last column is the target we want to predict
    target_col = df.columns[-1]
    
    # Detect column types for better analysis
    col_types = detect_column_types(df, target_col)
    
    # --- BLOCK 1: MISSING VALUES ---
    # We check each column to see if there are any empty (NaN) cells.
    # We store the total count and the count for each specific column.
    missing_values = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    diagnostics['missing_values'] = {
        'total': total_missing,
        'by_column': {k: int(v) for k, v in missing_values.items()}
    }
    
    # --- BLOCK 2: DUPLICATE ROWS ---
    # We look for rows that are exactly the same. Having many duplicates
    # can make the model "memorize" certain patterns too much (overfitting).
    duplicates = int(df.duplicated().sum())
    diagnostics['duplicates'] = duplicates
    
    # --- BLOCK 3: BASIC STATISTICS ---
    # For numerical columns, we calculate the average (mean), middle point (median),
    # and how much the data varies (std). This helps understand the data range.
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        stats = num_df.describe().T[['mean', '50%', 'std']]
        stats.columns = ['mean', 'median', 'std']
        diagnostics['statistics'] = stats.to_dict(orient='index')
    else:
        diagnostics['statistics'] = {}
        
    # --- BLOCK 4: OUTLIERS (Extreme Values) ---
    # We use the Interquartile Range (IQR) method to find values that are 
    # unusually high or low compared to the rest of the data.
    outliers_count = {}
    if not num_df.empty:
        for col in num_df.columns:
            Q1 = num_df[col].quantile(0.25)
            Q3 = num_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            count = ((num_df[col] < lower_bound) | (num_df[col] > upper_bound)).sum()
            outliers_count[col] = int(count)
    
    total_outliers = int(sum(outliers_count.values()))
    diagnostics['outliers'] = {
        'total': total_outliers,
        'by_column': outliers_count
    }
    # IsolationForest outliers (distribution-agnostic)
    isolation_outliers = 0
    if not num_df.empty and len(num_df) > 20:
        try:
            iso_X = num_df.fillna(num_df.median(numeric_only=True)).replace([np.inf, -np.inf], 0)
            if len(iso_X) > 5000:
                iso_X = iso_X.sample(n=5000, random_state=42)
            iso = IsolationForest(contamination='auto', random_state=42)
            preds = iso.fit_predict(iso_X)
            sampled_outliers = int((preds == -1).sum())
            if rows > len(iso_X):
                isolation_outliers = int(sampled_outliers * (rows / len(iso_X)))
            else:
                isolation_outliers = sampled_outliers
        except Exception:
            isolation_outliers = 0
    diagnostics['outliers']['isolation_forest'] = isolation_outliers
    
    # --- BLOCK 5: CLASS IMBALANCE ---
    # We check if one category in the target column has way more samples 
    # than others. If it does, the model might only learn the majority category.
    class_dist = df[target_col].value_counts().to_dict()
    diagnostics['class_imbalance'] = {
        'target_column': target_col,
        'distribution': {str(k): int(v) for k, v in class_dist.items()}
    }

    # --- BLOCK 6: LEAKAGE DIAGNOSTICS ---
    # Correlation + Spearman + Mutual Information.
    leakage_risk = []
    leakage_scores = {}
    if not num_df.empty and target_col in num_df.columns:
        corr = num_df.corr()[target_col].abs().sort_values(ascending=False)
        # Exclude target itself and take top 5 most related features
        top_corr = corr[1:6].to_dict()
        diagnostics['correlations'] = {k: round(v, 4) for k, v in top_corr.items()}
        
        # If correlation is over 0.95, it's a high risk of leakage
        for col, val in top_corr.items():
            if val > 0.95:
                leakage_risk.append(col)

        try:
            spearman_corr = num_df.corr(method='spearman')[target_col].abs().sort_values(ascending=False)
            top_spearman = spearman_corr[1:6].to_dict()
            diagnostics['spearman_correlations'] = {k: round(v, 4) for k, v in top_spearman.items()}
            for col, val in top_spearman.items():
                leakage_scores.setdefault(col, {})['spearman'] = float(val)
                if val > 0.95 and col not in leakage_risk:
                    leakage_risk.append(col)
        except Exception:
            diagnostics['spearman_correlations'] = {}

        try:
            X_mi = num_df.drop(columns=[target_col]).fillna(0)
            y_mi = num_df[target_col].fillna(0)
            if X_mi.shape[1] > 50:
                X_mi = X_mi.iloc[:, :50]
            if len(X_mi) > 5000:
                sample_idx = X_mi.sample(n=5000, random_state=42).index
                X_mi = X_mi.loc[sample_idx]
                y_mi = y_mi.loc[sample_idx]
            if pd.api.types.is_numeric_dtype(y_mi) and y_mi.nunique() > 20:
                mi_vals = mutual_info_regression(X_mi, y_mi, random_state=42)
            else:
                mi_vals = mutual_info_classif(X_mi, y_mi, random_state=42)
            mi_map = dict(zip(X_mi.columns.tolist(), mi_vals.tolist()))
            diagnostics['mutual_information'] = {k: round(float(v), 4) for k, v in mi_map.items()}
            for col, mi in mi_map.items():
                leakage_scores.setdefault(col, {})['mutual_info'] = float(mi)
                if mi > 0.8 and col not in leakage_risk:
                    leakage_risk.append(col)
        except Exception:
            diagnostics['mutual_information'] = {}
    else:
        diagnostics['correlations'] = {}
        diagnostics['spearman_correlations'] = {}
        diagnostics['mutual_information'] = {}
    diagnostics['leakage_scores'] = leakage_scores
    diagnostics['leakage_risk'] = leakage_risk

    # --- BLOCK 7: SEMANTIC & CONSISTENCY ANALYSIS (Label Noise) ---
    # We use a K-Nearest Neighbors (KNN) model to see if a row's label 
    # matches its most similar neighbors. If not, the label might be "noisy" (wrong).
    # IMPROVEMENT: Sample large datasets to avoid memory issues
    label_noise_count = 0
    label_noise_method = 'none'
    if rows > 10:
        try:
            # Sample large datasets for KNN analysis (max 10,000 rows)
            if rows > 3000:
                sample_size = min(3000, rows)
                df_knn = df.sample(n=sample_size, random_state=42)
            else:
                df_knn = df
            
            # We temporarily fill missing values and encode text to numbers for KNN
            temp_df = df_knn.copy().fillna(0)
            le = LabelEncoder()
            for col in temp_df.select_dtypes(include=['object']).columns:
                # IMPROVEMENT: Skip high-cardinality columns (>50 unique values)
                if temp_df[col].nunique() > 50:
                    temp_df = temp_df.drop(columns=[col])
                    continue
                temp_df[col] = le.fit_transform(temp_df[col].astype(str))
            
            X_diag = temp_df.drop(columns=[target_col])
            y_diag = temp_df[target_col]
            
            # Prefer Cleanlab when available.
            if len(y_diag.unique()) > 1:
                scaler = StandardScaler()
                X_diag_scaled = scaler.fit_transform(X_diag)
                if find_label_issues is not None:
                    knn = KNeighborsClassifier(n_neighbors=5)
                    knn.fit(X_diag_scaled, y_diag)
                    if hasattr(knn, 'predict_proba'):
                        pred_probs = knn.predict_proba(X_diag_scaled)
                        noisy_idx = find_label_issues(labels=y_diag.to_numpy(), pred_probs=pred_probs)
                        label_noise_count = int(len(noisy_idx))
                        label_noise_method = 'cleanlab_confident_learning'
                if label_noise_count == 0:
                    knn = KNeighborsClassifier(n_neighbors=5)
                    knn.fit(X_diag_scaled, y_diag)
                    y_pred = knn.predict(X_diag_scaled)
                    label_noise_count = int((y_pred != y_diag).sum())
                    label_noise_method = 'knn_fallback'
                
                # Scale back to full dataset if we sampled
                if rows > 3000:
                    noise_ratio_sample = label_noise_count / sample_size
                    label_noise_count = int(noise_ratio_sample * rows)
        except Exception:
            pass
    
    diagnostics['label_noise'] = label_noise_count
    diagnostics['label_noise_method'] = label_noise_method

    # --- BLOCK 7.5: NEAR DUPLICATE DETECTION (SEMANTIC-STYLE) ---
    near_duplicates = 0
    if rows > 2:
        try:
            near_df = df
            if rows > 1500:
                near_df = df.sample(n=1500, random_state=42)
            row_text = near_df.astype(str).agg(' | '.join, axis=1)
            tfidf = TfidfVectorizer(max_features=2000)
            emb = tfidf.fit_transform(row_text)
            nn = NearestNeighbors(metric='cosine', n_neighbors=2)
            nn.fit(emb)
            distances, _ = nn.kneighbors(emb)
            # second neighbor is nearest non-self neighbor
            sampled_near_dups = int((distances[:, 1] < 0.03).sum())
            if rows > len(near_df):
                near_duplicates = int(sampled_near_dups * (rows / len(near_df)))
            else:
                near_duplicates = sampled_near_dups
        except Exception:
            near_duplicates = 0
    diagnostics['near_duplicates'] = near_duplicates

    # --- BLOCK 7.6: FEATURE REDUNDANCY ---
    feature_redundancy = {'high_corr_pairs': []}
    if num_df.shape[1] > 1:
        corr_mat = num_df.corr().abs()
        cols_list = corr_mat.columns.tolist()
        for i in range(len(cols_list)):
            for j in range(i + 1, len(cols_list)):
                c1, c2 = cols_list[i], cols_list[j]
                v = corr_mat.iloc[i, j]
                if pd.notna(v) and float(v) > 0.95:
                    feature_redundancy['high_corr_pairs'].append([c1, c2, round(float(v), 4)])
    diagnostics['feature_redundancy'] = feature_redundancy

    # --- BLOCK 8: MIXED FIELD INCONSISTENCIES ---
    # Detect columns with mixed data types or inconsistent formats
    # Example: A column with both numbers and strings like "123", "N/A", "unknown"
    mixed_fields = {}
    for col in df.columns:
        if col == target_col:
            continue
        
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        # Check for mixed types in object columns
        if col_data.dtype == 'object':
            # Check if values have inconsistent formats
            values = col_data.astype(str)
            
            # Detect mixed numeric and non-numeric
            numeric_mask = values.str.match(r'^-?\d*\.?\d+$')
            has_numeric = numeric_mask.any()
            has_text = (~numeric_mask).any()
            
            if has_numeric and has_text:
                # Count different types
                numeric_count = values.str.match(r'^-?\d*\.?\d+$').sum()
                text_count = (~values.str.match(r'^-?\d*\.?\d+$')).sum()
                
                mixed_fields[col] = {
                    'type': 'Mixed numeric and text',
                    'numeric_count': int(numeric_count),
                    'text_count': int(text_count),
                    'sample_values': values.head(5).tolist()
                }
            
            # Detect inconsistent date formats
            elif values.str.match(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}').any():
                # Some values look like dates, others don't
                is_date = values.str.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}')
                date_count = is_date.sum()
                non_date_count = (~is_date).sum()
                
                if date_count > 0 and non_date_count > 0:
                    mixed_fields[col] = {
                        'type': 'Mixed date formats',
                        'date_count': int(date_count),
                        'non_date_count': int(non_date_count),
                        'sample_values': values.head(5).tolist()
                    }
    
    diagnostics['mixed_fields'] = mixed_fields

    # --- BLOCK 8.5: DISTRIBUTION SKEW ---
    # Detect strong skew in numeric features; this is part of FYP issue taxonomy.
    skewed_features = {}
    if not num_df.empty:
        skew_vals = num_df.skew(numeric_only=True)
        for col, skew in skew_vals.items():
            if pd.notna(skew):
                abs_skew = float(abs(skew))
                if abs_skew > 1.0:
                    skewed_features[col] = round(float(skew), 4)
    diagnostics['distribution_skew'] = {
        'skewed_features': skewed_features,
        'count': len(skewed_features)
    }

    # --- BLOCK 9: ISSUE IDENTIFICATION & SEVERITY ---
    # Based on the results above, we decide if an issue is "High" or "Medium" risk.
    issues = []
    
    # Check Missing Values ratio
    mv_ratio = total_missing / (rows * cols) if (rows * cols) > 0 else 0
    if mv_ratio > 0.05:
        issues.append({'type': 'Missing Values', 'severity': 'High' if mv_ratio > 0.2 else 'Medium', 'score': mv_ratio})
        
    # Check Class Imbalance ratio
    if len(class_dist) > 1:
        counts = list(class_dist.values())
        ratio = max(counts) / min(counts) if min(counts) > 0 else 100
        if ratio > 5:
            issues.append({'type': 'Class Imbalance', 'severity': 'High' if ratio > 20 else 'Medium', 'score': ratio})
            
    # Check Redundancy (Duplicates)
    dup_ratio = duplicates / rows if rows > 0 else 0
    if dup_ratio > 0.05 or near_duplicates > 0:
        issues.append({'type': 'Redundancy', 'severity': 'High' if dup_ratio > 0.15 else 'Medium', 'score': dup_ratio})
        
    # Check Outliers ratio
    outlier_ratio = total_outliers / (rows * len(num_df.columns)) if not num_df.empty else 0
    if outlier_ratio > 0.1 or isolation_outliers > 0:
        issues.append({'type': 'Outliers', 'severity': 'High' if outlier_ratio > 0.25 else 'Medium', 'score': outlier_ratio})

    # Check Label Noise ratio
    noise_ratio = label_noise_count / rows if rows > 0 else 0
    if noise_ratio > 0.1:
        issues.append({'type': 'Label Noise', 'severity': 'High' if noise_ratio > 0.2 else 'Medium', 'score': noise_ratio})

    # Check for Data Leakage
    if leakage_risk:
        issues.append({'type': 'Data Leakage', 'severity': 'High', 'score': len(leakage_risk)})
    
    # Check for Mixed Field Inconsistencies
    if mixed_fields:
        mixed_ratio = len(mixed_fields) / cols if cols > 0 else 0
        issues.append({'type': 'Mixed Field Inconsistencies', 'severity': 'High' if mixed_ratio > 0.2 else 'Medium', 'score': len(mixed_fields)})

    # Check for distribution skew
    if skewed_features:
        skew_ratio = len(skewed_features) / max(1, len(num_df.columns))
        issues.append({
            'type': 'Distribution Skew',
            'severity': 'High' if skew_ratio > 0.5 else 'Medium',
            'score': skew_ratio
        })

    # Final list of identified issues to be shown on the dashboard
    diagnostics['identified_issues'] = issues
    diagnostics['issue_taxonomy'] = {
        'imbalance': any(i['type'] == 'Class Imbalance' for i in issues),
        'label_noise': any(i['type'] == 'Label Noise' for i in issues),
        'redundancy': any(i['type'] == 'Redundancy' for i in issues),
        'leakage': any(i['type'] == 'Data Leakage' for i in issues),
        'distribution_skew': any(i['type'] == 'Distribution Skew' for i in issues)
    }
    
    # Store column type classification
    diagnostics['column_types'] = col_types
    
    # Add datetime column info if present
    if col_types['datetime_cols']:
        diagnostics['datetime_columns'] = col_types['datetime_cols']
    
    return diagnostics
