"""
ADIE Intelligent Engine — Dataset-Aware Strategy Generator.

Analyzes a dataset and produces a cleaning/feature-engineering strategy
dictionary that the adaptive_cleaning module consumes.  Every decision
is data-driven: dataset type detection, domain inference, column
importance scoring, and protection-list generation happen here.
"""

import pandas as pd
import numpy as np
import re
from utils.column_detector import detect_column_types


# ---------------------------------------------------------------------------
# 1. Dataset-type detection
# ---------------------------------------------------------------------------

def _detect_dataset_type(df, target_col, col_types):
    """
    Classify the dataset into one of:
      time_series | relational | tabular_small | tabular_large
    """
    rows = len(df)

    # Time-series heuristic: datetime column present AND data looks sequential
    if col_types.get('datetime_cols'):
        return 'time_series'

    # Check column names for temporal hints
    col_lower = [c.lower() for c in df.columns]
    time_hints = ['date', 'timestamp', 'time', 'year', 'month', 'quarter']
    if any(h in ' '.join(col_lower) for h in time_hints):
        # Verify there is a plausible ordering column
        for col in df.columns:
            if any(h in col.lower() for h in time_hints):
                if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col].head(10), errors='raise')
                        return 'time_series'
                    except Exception:
                        pass

    # Relational heuristic: identifier columns present
    if col_types.get('identifiers') and len(col_types['identifiers']) >= 1:
        return 'relational'

    # Size-based
    if rows < 500:
        return 'tabular_small'
    return 'tabular_large'


# ---------------------------------------------------------------------------
# 2. Domain inference
# ---------------------------------------------------------------------------

_DOMAIN_RULES = {
    'aviation': {
        'keywords': ['flight', 'airport', 'icao', 'iata', 'dep', 'arr',
                      'airline', 'runway', 'altitude', 'aircraft', 'terminal',
                      'passenger', 'cargo', 'aviation', 'atc'],
        'domain': 'Aviation / Air Transport',
        'industry': 'Aviation & Aerospace',
    },
    'finance': {
        'keywords': ['amount', 'price', 'loan', 'credit', 'balance',
                      'fiscal', 'interest', 'deal', 'revenue', 'profit',
                      'transaction', 'payment', 'invoice'],
        'domain': 'Finance / International Trade',
        'industry': 'Banking & Financial Services',
    },
    'healthcare': {
        'keywords': ['age', 'blood', 'patient', 'diagnosis', 'treatment',
                      'cancer', 'medical', 'hospital', 'symptom', 'drug'],
        'domain': 'Healthcare / Clinical',
        'industry': 'Medical Research',
    },
    'marketing': {
        'keywords': ['customer', 'click', 'conversion', 'sale', 'lead',
                      'campaign', 'churn', 'retention', 'engagement'],
        'domain': 'E-commerce / Marketing',
        'industry': 'Retail & Digital Marketing',
    },
    'manufacturing': {
        'keywords': ['machine', 'downtime', 'defect', 'production',
                      'assembly', 'quality', 'sensor', 'temperature',
                      'pressure', 'vibration'],
        'domain': 'Manufacturing / IoT',
        'industry': 'Industrial Manufacturing',
    },
}


def _infer_domain(df):
    """Return (domain_key, domain_label, industry) based on column names."""
    col_blob = ' '.join(c.lower() for c in df.columns)
    best_key, best_score = 'general', 0
    for key, rule in _DOMAIN_RULES.items():
        score = sum(1 for kw in rule['keywords'] if kw in col_blob)
        if score > best_score:
            best_score = score
            best_key = key
    if best_score == 0:
        return 'general', 'General Purpose', 'Cross-industry'
    r = _DOMAIN_RULES[best_key]
    return best_key, r['domain'], r['industry']


# ---------------------------------------------------------------------------
# 3. Column importance & protection list
# ---------------------------------------------------------------------------

def _build_protection_list(df, target_col, col_types, domain_key):
    """
    Return a set of column names that must NEVER be dropped or degraded.
    Includes: target, identifiers, high-importance features, domain-critical.
    """
    protected = {target_col}

    # Identifiers (needed for relational joins / display)
    for c in col_types.get('identifiers', []):
        protected.add(c)

    # Numeric columns with high variance-ratio to target (proxy for importance)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target_col in num_cols and len(num_cols) > 1:
        try:
            corr = df[num_cols].corr()[target_col].abs().sort_values(ascending=False)
            top_features = corr.index[1:6].tolist()  # top-5 correlated
            protected.update(top_features)
        except Exception:
            pass

    # Domain-critical columns (dynamic keyword match)
    if domain_key in _DOMAIN_RULES:
        kws = _DOMAIN_RULES[domain_key]['keywords']
        for col in df.columns:
            if any(kw in col.lower() for kw in kws):
                protected.add(col)

    return protected


# ---------------------------------------------------------------------------
# 4. High-cardinality detection
# ---------------------------------------------------------------------------

def _identify_high_cardinality_cols(df, target_col, threshold=50):
    """Return list of categorical columns with >threshold unique values."""
    high_card = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col == target_col:
            continue
        if df[col].nunique() > threshold:
            high_card.append(col)
    return high_card


# ---------------------------------------------------------------------------
# 5. Public entry point
# ---------------------------------------------------------------------------

def run_intelligent_analysis(df, target_col):
    """
    Analyse *df* and return a strategy dict consumed by adaptive_cleaning.

    Returns
    -------
    dict with keys:
        dataset_type        : str
        domain_key          : str
        domain_label        : str
        industry            : str
        protected_columns   : set[str]
        high_cardinality    : list[str]
        col_types           : dict  (from column_detector)
        time_series_target  : str | None
        identifier_columns  : list[str]
        is_small_dataset    : bool
        recommendations     : list[str]
    """
    col_types = detect_column_types(df, target_col)
    dataset_type = _detect_dataset_type(df, target_col, col_types)
    domain_key, domain_label, industry = _infer_domain(df)
    protected = _build_protection_list(df, target_col, col_types, domain_key)
    high_card = _identify_high_cardinality_cols(df, target_col)
    is_small = len(df) < 500

    recommendations = []
    if dataset_type == 'time_series':
        recommendations.append('Add lag features and rolling statistics')
        recommendations.append('Preserve chronological order — no random shuffle')
    if dataset_type == 'relational':
        recommendations.append('Create entity-frequency features from identifiers')
        recommendations.append('Preserve identifier columns for group-based features')
    if is_small:
        recommendations.append('Limit feature engineering to avoid overfitting')
        recommendations.append('Avoid aggressive transformations')
    if high_card:
        recommendations.append(
            f'Use frequency encoding for high-cardinality columns: {high_card}')
    if domain_key == 'aviation':
        recommendations.append('Engineer traffic_volume and dep_arr_ratio features')

    return {
        'dataset_type': dataset_type,
        'domain_key': domain_key,
        'domain_label': domain_label,
        'industry': industry,
        'protected_columns': protected,
        'high_cardinality': high_card,
        'col_types': col_types,
        'time_series_target': target_col if dataset_type == 'time_series' else None,
        'identifier_columns': col_types.get('identifiers', []),
        'is_small_dataset': is_small,
        'recommendations': recommendations,
    }
