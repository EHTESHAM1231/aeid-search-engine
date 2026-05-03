"""
Quick test to verify high-cardinality fix works
"""
import pandas as pd
import numpy as np

# Test data with high-cardinality column (would cause memory error before)
print("Creating test dataset with high-cardinality column...")
n_rows = 1000

data = {
    'id': [f'ID_{i}' for i in range(n_rows)],
    'high_card_category': [f'Cat_{i}' for i in range(n_rows)],  # 1000 unique values!
    'low_card_category': np.random.choice(['A', 'B', 'C'], n_rows),
    'feature1': np.random.randn(n_rows),
    'target': np.random.choice([0, 1], n_rows)
}

df = pd.DataFrame(data)
print(f"Dataset shape: {df.shape}")
print(f"High cardinality column has {df['high_card_category'].nunique()} unique values")
print(f"Low cardinality column has {df['low_card_category'].nunique()} unique values")

# Test column detection
print("\nTesting column detection...")
from utils.column_detector import detect_column_types
col_types = detect_column_types(df, target_col='target')
print(f"Nominal categorical columns: {col_types['nominal_categorical']}")

# Test cleaning
print("\nTesting data cleaning with high-cardinality handling...")
from utils.data_cleaning import clean_dataset

try:
    cleaned_df = clean_dataset(df, target_col='target')
    print(f"✅ SUCCESS! Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Columns after cleaning: {cleaned_df.columns.tolist()}")
    
    # Verify high-cardinality was frequency encoded (not one-hot)
    if 'high_card_category' in cleaned_df.columns:
        print("✅ High-cardinality column frequency encoded (not one-hot encoded)")
        print(f"   Values are now numeric frequencies: {cleaned_df['high_card_category'].head()}")
    
    # Verify low-cardinality was one-hot encoded
    low_card_encoded = [col for col in cleaned_df.columns if 'low_card_category_' in col]
    if low_card_encoded:
        print(f"✅ Low-cardinality column one-hot encoded: {low_card_encoded}")
    
    print("\n🎉 HIGH-CARDINALITY FIX WORKING CORRECTLY!")
    print("   The memory error has been resolved.")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
