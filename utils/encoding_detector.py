import chardet
import pandas as pd
import warnings
import io

def detect_file_encoding(filepath, sample_size=10000):
    """
    Detect file encoding with fallback options for robust file reading.
    
    Args:
        filepath (str): Path to the file
        sample_size (int): Number of bytes to sample for encoding detection
    
    Returns:
        str: Detected encoding
    """
    try:
        with open(filepath, 'rb') as f:
            raw_data = f.read(sample_size)
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            
            # If confidence is low or encoding is None, try common encodings
            if confidence < 0.7 or encoding is None:
                return try_common_encodings(filepath)
            
            return encoding
    except Exception:
        return try_common_encodings(filepath)

def try_common_encodings(filepath):
    """
    Try common encodings in order of preference.
    
    Args:
        filepath (str): Path to the file
    
    Returns:
        str: Working encoding or 'utf-8' as default
    """
    common_encodings = [
        'utf-8',
        'utf-8-sig',  # UTF-8 with BOM
        'latin-1',    # Most permissive
        'cp1252',     # Windows default
        'iso-8859-1',
        'iso-8859-15',
        'windows-1252',
        'ascii'
    ]
    
    for encoding in common_encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                f.read(1000)  # Try to read a small sample
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # If none work, return utf-8 as last resort (will ignore errors)
    return 'utf-8'

def read_csv_with_encoding(filepath, **kwargs):
    """
    Read CSV file with automatic encoding detection and robust error handling.
    
    Args:
        filepath (str): Path to the CSV file
        **kwargs: Additional arguments for pd.read_csv
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    # Suppress warnings for encoding issues
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        encoding = detect_file_encoding(filepath)
        
        # Try different parsing strategies in order of preference
        strategies = [
            # Strategy 1: Standard parsing with detected encoding
            lambda: pd.read_csv(filepath, encoding=encoding, **kwargs),
            
            # Strategy 2: More flexible parsing parameters
            lambda: pd.read_csv(filepath, encoding=encoding, 
                              sep=None, engine='python', 
                              quoting=1, skipinitialspace=True, **kwargs),
            
            # Strategy 3: Skip bad lines
            lambda: pd.read_csv(filepath, encoding=encoding, 
                              on_bad_lines='skip', **kwargs),
            
            # Strategy 4: UTF-8 with error handling
            lambda: pd.read_csv(filepath, encoding='utf-8', errors='ignore', **kwargs),
            
            # Strategy 5: UTF-8 with flexible parsing
            lambda: pd.read_csv(filepath, encoding='utf-8', errors='ignore',
                              sep=None, engine='python', 
                              quoting=1, skipinitialspace=True, **kwargs),
            
            # Strategy 6: Latin-1 with flexible parsing
            lambda: pd.read_csv(filepath, encoding='latin-1',
                              sep=None, engine='python', 
                              quoting=1, skipinitialspace=True, **kwargs),
            
            # Strategy 7: Read as text and parse manually
            lambda: _manual_csv_parse(filepath, encoding),
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                return strategy()
            except Exception as e:
                if i == len(strategies) - 1:  # Last strategy failed
                    raise Exception(f"Failed to read CSV file after trying all strategies. Last error: {str(e)}")
                continue
        
        # This should never be reached
        raise Exception("All CSV parsing strategies failed")

def _manual_csv_parse(filepath, encoding):
    """
    Manual CSV parsing as last resort for severely malformed files.
    
    Args:
        filepath (str): Path to the CSV file
        encoding (str): File encoding
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        # Read file line by line and fix common issues
        with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
            lines = f.readlines()
        
        # Clean up lines
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Fix common CSV issues
            # Remove extra commas at end of line
            while line.endswith(','):
                line = line[:-1]
            
            # Ensure balanced quotes
            quote_count = line.count('"')
            if quote_count % 2 != 0:
                line += '"'  # Add missing quote
            
            cleaned_lines.append(line)
        
        # Join lines and parse with pandas
        csv_text = '\n'.join(cleaned_lines)
        return pd.read_csv(io.StringIO(csv_text), engine='python', on_bad_lines='skip')
        
    except Exception:
        # Final fallback - try to read with very permissive settings
        return pd.read_csv(filepath, encoding='latin-1', 
                          engine='python', sep=None, 
                          on_bad_lines='skip', skip_blank_lines=True)
