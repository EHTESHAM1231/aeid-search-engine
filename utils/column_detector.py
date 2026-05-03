import pandas as pd
import numpy as np
import re

def detect_column_types(df, target_col=None):
    """
    Intelligently classify columns into types:
    - identifier: High cardinality, unique values (IDs, codes, names)
    - datetime: Date/time columns
    - numerical: int, float, numeric strings
    - nominal_categorical: Categories without order (colors, cities)
    - ordinal_categorical: Categories with order (low/medium/high)
    - target: The column to predict
    """
    if target_col is None:
        target_col = df.columns[-1]
    
    column_classification = {
        'identifiers': [],
        'datetime_cols': [],
        'numerical_cols': [],
        'nominal_categorical': [],
        'ordinal_categorical': [],
        'target': target_col
    }
    
    for col in df.columns:
        if col == target_col:
            continue
            
        col_data = df[col]
        unique_ratio = col_data.nunique() / len(col_data) if len(col_data) > 0 else 0
        
        # 1. Check for datetime
        if _is_datetime_column(col_data, col):
            column_classification['datetime_cols'].append(col)
            continue
        
        # 2. Check for identifier (high cardinality + unique patterns)
        if _is_identifier_column(col_data, unique_ratio, col):
            column_classification['identifiers'].append(col)
            continue
        
        # 3. Check for numerical
        if pd.api.types.is_numeric_dtype(col_data):
            column_classification['numerical_cols'].append(col)
            continue
        
        # 4. Check if categorical
        if col_data.dtype == 'object' or pd.api.types.is_categorical_dtype(col_data):
            if _is_ordinal(col_data):
                column_classification['ordinal_categorical'].append(col)
            else:
                column_classification['nominal_categorical'].append(col)
    
    return column_classification

def _is_datetime_column(col_data, col_name):
    """Detect datetime columns from strings or numeric timestamps."""
    # Already datetime type
    if pd.api.types.is_datetime64_any_dtype(col_data):
        return True
    
    # Try parsing as datetime
    if col_data.dtype == 'object':
        # Check common date patterns
        sample = col_data.dropna().head(10)
        if len(sample) == 0:
            return False
        
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # MM/DD/YYYY
            r'\d{1,2}[-/]\w+[-/]\d{4}',      # DD-Mon-YYYY
            r'\w+\s+\d{1,2},?\s+\d{4}'       # Month DD, YYYY
        ]
        
        matches = 0
        for val in sample:
            if any(re.match(p, str(val)) for p in date_patterns):
                matches += 1
        
        if matches > len(sample) * 0.7:  # 70% match threshold
            return True
        
        # Try pandas to_datetime
        try:
            pd.to_datetime(sample, errors='raise')
            return True
        except:
            return False
    
    return False

def _is_identifier_column(col_data, unique_ratio, col_name):
    """Detect identifier columns (IDs, codes, names)."""
    col_lower = col_name.lower()
    
    # If column is numeric, it's not an identifier (even with high cardinality)
    if pd.api.types.is_numeric_dtype(col_data):
        return False
    
    # Name-based detection
    id_keywords = ['id', 'code', 'number', 'no', 'name', 'icao', 'identifier', 'key']
    if any(kw in col_lower for kw in id_keywords):
        return True
    
    # High cardinality detection (>80% unique) - only for non-numeric columns
    if unique_ratio > 0.8 and len(col_data) > 50:
        return True
    
    return False

def _is_ordinal(col_data):
    """Detect ordinal categorical columns."""
    ordinal_patterns = [
        ['low', 'medium', 'high'],
        ['small', 'medium', 'large'],
        ['poor', 'fair', 'good', 'excellent'],
        ['junior', 'senior', 'lead', 'manager'],
        ['level 1', 'level 2', 'level 3']
    ]
    
    unique_vals = [str(v).lower() for v in col_data.unique()[:10]]
    
    for pattern in ordinal_patterns:
        if any(p in unique_vals for p in pattern):
            return True
    
    return False
