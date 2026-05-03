import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.encoding_detector import read_csv_with_encoding
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)
from imblearn.combine import SMOTETomek
import joblib
import os
import warnings

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

# --- BLOCK 1: SETUP STORAGE PATHS ---
# We define where to save the best performing model and the data scaler
# so we can use them later for predictions.
MODEL_PATH = os.path.join('uploads', 'best_model.pkl')
SCALER_PATH = os.path.join('uploads', 'scaler.pkl')


def _get_param_distributions(model_name, task_type):
    if task_type == 'classification':
        if model_name == 'Random Forest':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 8, 16, 24],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        if model_name == 'Decision Tree':
            return {
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        if model_name == 'Logistic Regression':
            return {
                'C': [0.01, 0.1, 1.0, 10.0, 50.0],
                'solver': ['lbfgs', 'liblinear']
            }
        if model_name == 'KNN':
            return {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance']
            }
    else:
        if model_name == 'Random Forest':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 8, 16, 24],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        if model_name == 'Decision Tree':
            return {
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        if model_name == 'KNN':
            return {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance']
            }
    return None

def detect_task_type(df, target_col):
    """
    Detects if we should do Classification (predicting categories) 
    or Regression (predicting numbers).
    IMPROVEMENT: Check all numeric dtypes, not just float64/int64
    """
    # Sanitize target_col
    target_col = target_col.strip()
    
    # Verify target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
    
    unique_vals = df[target_col].nunique()
    # If the target is decimal or has many unique values, it's likely a number prediction (Regression)
    if pd.api.types.is_numeric_dtype(df[target_col]) and unique_vals > 20:
        return 'regression'
    else:
        return 'classification'

def get_models(task_type):
    """
    Returns a dictionary of different ML algorithms to try out 
    based on whether we are doing classification or regression.
    """
    if task_type == 'classification':
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        if XGBClassifier is not None:
            models['XGBoost'] = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric='logloss'
            )
        return models
    else:
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }
        if XGBRegressor is not None:
            models['XGBoost'] = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42
            )
        return models

def train_and_evaluate(df, target_col, selected_algo='All Algorithms', quarantined_columns=None):
    """
    The main function that splits data, trains models, and calculates scores.
    """
    # Sanitize column names
    df = df.copy()
    df.columns = df.columns.str.strip()
    target_col = target_col.strip()
    
    # Verify target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}")
    
    # X is the features (data used to predict), y is the target (what we want to predict)
    cols_to_drop = [target_col]
    if quarantined_columns:
        # Only drop quarantined columns that are actually in the dataframe
        valid_quarantined = [c.strip() for c in quarantined_columns if c.strip() in df.columns]
        cols_to_drop.extend(valid_quarantined)
        
    X = df.drop(columns=cols_to_drop)
    # Training matrix must be numeric even if cleaned dataset preserves original text columns.
    X = pd.get_dummies(X, dummy_na=False)
    y = df[target_col]
    
    task_type = detect_task_type(df, target_col)
    is_large_dataset = len(df) > 5000
    
    # --- BLOCK 2: DATA SPLITTING ---
    # We split the data into a Training set (80%) and a Testing set (20%).
    # The model learns from the training set and we check its accuracy on the testing set.
    stratify_y = None
    if task_type == 'classification':
        class_counts = y.value_counts()
        if (class_counts >= 2).all():
            stratify_y = y
    
    # Robust split for very small datasets.
    test_size = 0.2
    if len(df) < 10:
        test_size = max(0.25, 1 / max(2, len(df)))
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify_y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
    
    # --- BLOCK 3: HANDLE CLASS IMBALANCE (SMOTETomek) ---
    # Combined over/under-sampling generally performs better on hard boundaries.
    smote_applied = False
    if task_type == 'classification':
        try:
            if (y_train.value_counts() >= 2).all() and len(y_train.unique()) > 1:
                sm = SMOTETomek(random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                smote_applied = True
        except Exception:
            pass

    # --- BLOCK 4: DATA SCALING ---
    # Some models work better if all numbers are in the same small range.
    # We "scale" the data and save the scaler to use it again later.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    # Decide which models to run
    available_models = get_models(task_type)
    models_to_run = {}
    if selected_algo == 'All Algorithms':
        models_to_run = available_models
    elif selected_algo in available_models:
        models_to_run = {selected_algo: available_models[selected_algo]}
    else:
        return {
            selected_algo: {
                'error': f'"{selected_algo}" is not supported for {task_type}. Available: {list(available_models.keys())}'
            }
        }, task_type

    results = {}
    best_score = -1
    best_model = None
    model_rankings = []

    # --- BLOCK 5: MODEL TRAINING & SCORING ---
    # We loop through each selected algorithm, train it, and calculate metrics.
    for name, model in models_to_run.items():
        if model is None: continue
        try:
            use_scaled = name in ['KNN', 'Logistic Regression', 'Linear Regression']
            fit_X = X_train_scaled if use_scaled else X_train
            pred_X = X_test_scaled if use_scaled else X_test
            train_size = len(y_train)

            # Keep KNN valid on smaller datasets/folds.
            if name == 'KNN':
                safe_k = max(1, min(5, train_size - 1))
                model.set_params(n_neighbors=safe_k)

            # Lightweight hyperparameter optimization for better generalization.
            param_dist = _get_param_distributions(name, task_type)
            if name == 'KNN' and param_dist:
                max_k = max(1, min(11, train_size - 1))
                param_dist['n_neighbors'] = [k for k in [1, 3, 5, 7, 9, 11] if k <= max_k]
                if not param_dist['n_neighbors']:
                    param_dist['n_neighbors'] = [1]
            if param_dist and len(y_train) >= 20 and not is_large_dataset:
                try:
                    if task_type == 'classification':
                        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                        scoring = 'f1_weighted'
                    else:
                        cv = KFold(n_splits=3, shuffle=True, random_state=42)
                        scoring = 'r2'
                    tuner = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_dist,
                        n_iter=min(5, sum(len(v) for v in param_dist.values())),
                        scoring=scoring,
                        cv=cv,
                        random_state=42,
                        n_jobs=1
                    )
                    tuner.fit(fit_X, y_train)
                    model = tuner.best_estimator_
                except Exception:
                    pass

            # Train the model
            model.fit(fit_X, y_train)
            y_pred = model.predict(pred_X)
                
            model_results = {
                'params': model.get_params(),
                'y_test': y_test.tolist()[:50],
                'y_pred': y_pred.tolist()[:50],
                'smote_applied': smote_applied
            }
            
            # Calculate metrics (Accuracy, F1 for Classification; R2, MAE for Regression)
            is_classifier = hasattr(model, 'predict_proba') or 'Classifier' in str(type(model))
            if is_classifier:
                score = accuracy_score(y_test, y_pred)
                train_pred = model.predict(fit_X)
                train_score = accuracy_score(y_train, train_pred)
                cv_metric = None
                if name != 'KNN' and not is_large_dataset:
                    try:
                        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                        cv_scores = cross_val_score(model, fit_X, y_train, cv=cv, scoring='f1_weighted', n_jobs=1)
                        cv_metric = float(np.mean(cv_scores))
                    except Exception:
                        cv_metric = None
                model_results.update({
                    'Accuracy': round(score, 4),
                    'Train Accuracy': round(train_score, 4),
                    'Generalization Gap': round(float(train_score - score), 4),
                    'CV F1 (mean)': round(cv_metric, 4) if cv_metric is not None else None,
                    'Precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                    'Recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                    'F1-Score': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                })
                # Composite score rewards performance and penalizes overfitting.
                composite_score = float(
                    0.5 * model_results['Accuracy'] +
                    0.4 * model_results['F1-Score'] -
                    0.1 * abs(model_results['Generalization Gap'])
                )
            else:
                score = r2_score(y_test, y_pred)
                train_pred = model.predict(fit_X)
                train_r2 = r2_score(y_train, train_pred)
                cv_metric = None
                if not is_large_dataset:
                    try:
                        cv = KFold(n_splits=3, shuffle=True, random_state=42)
                        cv_scores = cross_val_score(model, fit_X, y_train, cv=cv, scoring='r2', n_jobs=1)
                        cv_metric = float(np.mean(cv_scores))
                    except Exception:
                        cv_metric = None
                model_results.update({
                    'MAE': round(mean_absolute_error(y_test, y_pred), 4),
                    'MSE': round(mean_squared_error(y_test, y_pred), 4),
                    'R2 Score': round(score, 4),
                    'Train R2': round(train_r2, 4),
                    'Generalization Gap': round(float(train_r2 - score), 4),
                    'CV R2 (mean)': round(cv_metric, 4) if cv_metric is not None else None
                })
                composite_score = float(
                    0.8 * model_results['R2 Score'] -
                    0.2 * abs(model_results['Generalization Gap'])
                )
            model_results['Composite Score'] = round(composite_score, 4)
            
            # Keep track of the best model found so far
            if composite_score > best_score:
                best_score = composite_score
                best_model = model

            # Determine which features were most important for this model
            if hasattr(model, 'feature_importances_'):
                importances = dict(zip(X.columns, model.feature_importances_))
                model_results['feature_importance'] = {k: round(float(v), 4) for k, v in importances.items()}

            results[name] = model_results
            model_rankings.append((name, composite_score, model, use_scaled))
        except Exception as e:
            results[name] = {'error': str(e)}

    # Add lightweight ensemble from top-2 successful models.
    try:
        successful = [(n, s, m, us) for (n, s, m, us) in model_rankings if n in results and 'error' not in results[n]]
        successful = sorted(successful, key=lambda x: x[1], reverse=True)
        if len(successful) >= 2:
            top2 = successful[:2]
            if task_type == 'classification':
                ensemble_preds = []
                proba_supported = True
                proba_sum = None
                for _, _, mdl, use_scaled_top in top2:
                    X_pred = pred_X if use_scaled_top else X_test
                    if hasattr(mdl, 'predict_proba'):
                        probs = mdl.predict_proba(X_pred)
                        proba_sum = probs if proba_sum is None else (proba_sum + probs)
                    else:
                        proba_supported = False
                        break
                if proba_supported and proba_sum is not None:
                    avg_probs = proba_sum / len(top2)
                    ensemble_preds = np.argmax(avg_probs, axis=1)
                    ensemble_acc = accuracy_score(y_test, ensemble_preds)
                    ensemble_f1 = f1_score(y_test, ensemble_preds, average='weighted', zero_division=0)
                    results['Ensemble (Top-2)'] = {
                        'Accuracy': round(float(ensemble_acc), 4),
                        'F1-Score': round(float(ensemble_f1), 4),
                        'Composite Score': round(float(0.5 * ensemble_acc + 0.4 * ensemble_f1), 4),
                        'y_test': y_test.tolist()[:50],
                        'y_pred': ensemble_preds.tolist()[:50],
                        'members': [top2[0][0], top2[1][0]]
                    }
            else:
                pred_sum = None
                for _, _, mdl, use_scaled_top in top2:
                    X_pred = pred_X if use_scaled_top else X_test
                    p = mdl.predict(X_pred)
                    pred_sum = p if pred_sum is None else (pred_sum + p)
                avg_pred = pred_sum / len(top2)
                ensemble_r2 = r2_score(y_test, avg_pred)
                ensemble_mae = mean_absolute_error(y_test, avg_pred)
                results['Ensemble (Top-2)'] = {
                    'R2 Score': round(float(ensemble_r2), 4),
                    'MAE': round(float(ensemble_mae), 4),
                    'Composite Score': round(float(0.8 * ensemble_r2), 4),
                    'y_test': y_test.tolist()[:50],
                    'y_pred': avg_pred.tolist()[:50],
                    'members': [top2[0][0], top2[1][0]]
                }
    except Exception:
        pass
    
    # --- BLOCK 6: SAVE BEST MODEL ---
    # We save the overall best model to a file so it can be downloaded or used later.
    if best_model:
        joblib.dump(best_model, MODEL_PATH)
            
    return results, task_type

def predict_new_data(filepath, target_col_name=None, quarantined_columns=None):
    from utils.data_cleaning import clean_dataset
    from utils.data_cleaning import ENCODER_PATH
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise ValueError("Model or scaler not found. Please train a model first.")
        
    df = read_csv_with_encoding(filepath)
    df.columns = df.columns.str.strip()
    original_df = df.copy()
    
    # Clean the dataset (inference mode, don't fit new encoders/imputers)
    ml_inference_policy = {
        'mode': 'balanced',
        'drop_identifier_columns': False,
        'drop_leakage_columns': False,
        'drop_high_missing_columns': False,
        'remove_duplicates': False, # Don't drop rows in inference
        'handle_outliers': True,
        'encode_features': True,
        'apply_smote': False, # Skip SMOTE
        'apply_cleanlab': False # Skip Cleanlab
    }
    
    df_processed = clean_dataset(
        df,
        target_col=None, # no target column provided
        preserve_structure=False,
        cleaning_policy=ml_inference_policy,
        fit_encoders=False, # Use saved encoders/imputers
        is_inference=True # Bypass target detection
    )
    
    # Drop quarantined columns if they exist
    if quarantined_columns:
        valid_quarantined = [c.strip() for c in quarantined_columns if c.strip() in df_processed.columns]
        df_processed = df_processed.drop(columns=valid_quarantined)
        
    X = pd.get_dummies(df_processed, dummy_na=False)
    
    model = joblib.load(MODEL_PATH)
    if hasattr(model, 'feature_names_in_'):
        expected_cols = model.feature_names_in_
        missing_cols = set(expected_cols) - set(X.columns)
        for c in missing_cols:
            X[c] = 0
        # Ignore extra columns that weren't in training
        X = X[[c for c in expected_cols if c in X.columns]]
        # Ensure exact match
        for c in expected_cols:
            if c not in X.columns:
                 X[c] = 0
        X = X[expected_cols]
    
    scaler = joblib.load(SCALER_PATH)
    if hasattr(scaler, 'n_features_in_') and X.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Feature mismatch. Model expects {scaler.n_features_in_} features, but got {X.shape[1]}.")
        
    X_scaled = scaler.transform(X)
    
    use_scaled = type(model).__name__ in ['KNeighborsClassifier', 'KNeighborsRegressor', 'LogisticRegression', 'LinearRegression']
    X_pred = X_scaled if use_scaled else X
    
    predictions = model.predict(X_pred)
    
    try:
        mappings = joblib.load(ENCODER_PATH)
        if 'target' in mappings:
            le = mappings['target']['encoder']
            # Only inverse transform if predictions look like class indices
            if pd.api.types.is_numeric_dtype(predictions) and np.issubdtype(predictions.dtype, np.integer):
                predictions = le.inverse_transform(predictions)
    except Exception:
        pass
        
    pred_col_name = f'Predicted_{target_col_name}' if target_col_name else 'Prediction'
    original_df[pred_col_name] = predictions
    
    return original_df
