"""
validate_submission.py - Validate submission for competition requirements

This script checks:
1. Schema correctness (column names, data types)
2. Completeness (all test records have predictions)
3. Data quality (no NaN, no negative values)
4. Code correctness (all required files present)
"""

import pandas as pd
import numpy as np
import os

def validate_schema(predictions_df):
    """Validate prediction file schema"""
    print("="*60)
    print("1. SCHEMA VALIDATION")
    print("="*60)
    
    required_columns = ['dt', 'atm_id', 'predicted_withdrawn_kwd', 'predicted_withdraw_count']
    
    # Check columns exist
    missing_cols = set(required_columns) - set(predictions_df.columns)
    if missing_cols:
        print(f"❌ FAIL: Missing columns: {missing_cols}")
        return False
    else:
        print(f"✓ All required columns present: {required_columns}")
    
    # Check data types
    try:
        predictions_df['dt'] = pd.to_datetime(predictions_df['dt'])
        print(f"✓ 'dt' column is valid date format")
    except:
        print(f"❌ FAIL: 'dt' column cannot be parsed as date")
        return False
    
    if predictions_df['atm_id'].dtype == 'object':
        print(f"✓ 'atm_id' is string/object type")
    else:
        print(f"⚠ WARNING: 'atm_id' is {predictions_df['atm_id'].dtype}, expected string")
    
    if pd.api.types.is_numeric_dtype(predictions_df['predicted_withdrawn_kwd']):
        print(f"✓ 'predicted_withdrawn_kwd' is numeric")
    else:
        print(f"❌ FAIL: 'predicted_withdrawn_kwd' is not numeric")
        return False
    
    if pd.api.types.is_numeric_dtype(predictions_df['predicted_withdraw_count']):
        print(f"✓ 'predicted_withdraw_count' is numeric")
    else:
        print(f"❌ FAIL: 'predicted_withdraw_count' is not numeric")
        return False
    
    print("\n✅ Schema validation PASSED\n")
    return True

def validate_completeness(predictions_df, test_df):
    """Validate all test records have predictions"""
    print("="*60)
    print("2. COMPLETENESS VALIDATION")
    print("="*60)
    
    # Convert dates for comparison
    predictions_df['dt'] = pd.to_datetime(predictions_df['dt'])
    test_df['dt'] = pd.to_datetime(test_df['dt'])
    
    # Check row counts
    print(f"Test records: {len(test_df)}")
    print(f"Prediction records: {len(predictions_df)}")
    
    if len(predictions_df) != len(test_df):
        print(f"❌ FAIL: Row count mismatch!")
        return False
    else:
        print(f"✓ Row counts match")
    
    # Check all (dt, atm_id) pairs exist
    test_keys = set(test_df[['dt', 'atm_id']].apply(tuple, axis=1))
    pred_keys = set(predictions_df[['dt', 'atm_id']].apply(tuple, axis=1))
    
    missing_keys = test_keys - pred_keys
    extra_keys = pred_keys - test_keys
    
    if missing_keys:
        print(f"❌ FAIL: {len(missing_keys)} test records missing predictions")
        print(f"   First 5 missing: {list(missing_keys)[:5]}")
        return False
    else:
        print(f"✓ All test records have predictions")
    
    if extra_keys:
        print(f"⚠ WARNING: {len(extra_keys)} extra predictions not in test set")
        print(f"   First 5 extra: {list(extra_keys)[:5]}")
    else:
        print(f"✓ No extra predictions")
    
    # Check for duplicates (but test data may have legitimate duplicates)
    test_duplicates = test_df.duplicated(subset=['dt', 'atm_id']).sum()
    pred_duplicates = predictions_df.duplicated(subset=['dt', 'atm_id']).sum()
    
    if test_duplicates > 0:
        print(f"ℹ Test data has {test_duplicates} duplicate (dt, atm_id) pairs")
        if pred_duplicates == test_duplicates:
            print(f"✓ Predictions correctly match test data duplicates ({pred_duplicates} pairs)")
        else:
            print(f"❌ FAIL: Prediction duplicates ({pred_duplicates}) don't match test ({test_duplicates})")
            return False
    else:
        if pred_duplicates > 0:
            print(f"❌ FAIL: {pred_duplicates} duplicate (dt, atm_id) pairs found")
            return False
        else:
            print(f"✓ No duplicate predictions")
    
    print("\n✅ Completeness validation PASSED\n")
    return True

def validate_data_quality(predictions_df):
    """Validate data quality"""
    print("="*60)
    print("3. DATA QUALITY VALIDATION")
    print("="*60)
    
    # Check for missing values
    missing_withdrawn = predictions_df['predicted_withdrawn_kwd'].isna().sum()
    missing_count = predictions_df['predicted_withdraw_count'].isna().sum()
    
    if missing_withdrawn > 0:
        print(f"❌ FAIL: {missing_withdrawn} missing values in predicted_withdrawn_kwd")
        return False
    else:
        print(f"✓ No missing values in predicted_withdrawn_kwd")
    
    if missing_count > 0:
        print(f"❌ FAIL: {missing_count} missing values in predicted_withdraw_count")
        return False
    else:
        print(f"✓ No missing values in predicted_withdraw_count")
    
    # Check for negative values
    negative_withdrawn = (predictions_df['predicted_withdrawn_kwd'] < 0).sum()
    negative_count = (predictions_df['predicted_withdraw_count'] < 0).sum()
    
    if negative_withdrawn > 0:
        print(f"❌ FAIL: {negative_withdrawn} negative values in predicted_withdrawn_kwd")
        return False
    else:
        print(f"✓ No negative values in predicted_withdrawn_kwd")
    
    if negative_count > 0:
        print(f"❌ FAIL: {negative_count} negative values in predicted_withdraw_count")
        return False
    else:
        print(f"✓ No negative values in predicted_withdraw_count")
    
    # Check for infinite values
    inf_withdrawn = np.isinf(predictions_df['predicted_withdrawn_kwd']).sum()
    inf_count = np.isinf(predictions_df['predicted_withdraw_count']).sum()
    
    if inf_withdrawn > 0:
        print(f"❌ FAIL: {inf_withdrawn} infinite values in predicted_withdrawn_kwd")
        return False
    else:
        print(f"✓ No infinite values in predicted_withdrawn_kwd")
    
    if inf_count > 0:
        print(f"❌ FAIL: {inf_count} infinite values in predicted_withdraw_count")
        return False
    else:
        print(f"✓ No infinite values in predicted_withdraw_count")
    
    # Statistics
    print(f"\nPrediction Statistics:")
    print(f"  predicted_withdrawn_kwd:")
    print(f"    Mean: {predictions_df['predicted_withdrawn_kwd'].mean():.2f} KWD")
    print(f"    Median: {predictions_df['predicted_withdrawn_kwd'].median():.2f} KWD")
    print(f"    Min: {predictions_df['predicted_withdrawn_kwd'].min():.2f} KWD")
    print(f"    Max: {predictions_df['predicted_withdrawn_kwd'].max():.2f} KWD")
    print(f"  predicted_withdraw_count:")
    print(f"    Mean: {predictions_df['predicted_withdraw_count'].mean():.2f}")
    print(f"    Median: {predictions_df['predicted_withdraw_count'].median():.2f}")
    print(f"    Min: {predictions_df['predicted_withdraw_count'].min():.2f}")
    print(f"    Max: {predictions_df['predicted_withdraw_count'].max():.2f}")
    
    print("\n✅ Data quality validation PASSED\n")
    return True

def validate_code_files():
    """Validate required code files exist"""
    print("="*60)
    print("4. CODE CORRECTNESS VALIDATION")
    print("="*60)
    
    required_files = {
        'predictions.csv': 'Main submission file',
        'train.py': 'Training script',
        'predict.py': 'Prediction script',
        'requirements.txt': 'Dependencies',
        'README.md': 'Documentation',
        'challenge1.ipynb': 'Analysis notebook (optional)'
    }
    
    all_exist = True
    for file, description in required_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✓ {file}: {description} ({size:,} bytes)")
        else:
            if file == 'challenge1.ipynb':
                print(f"⚠ {file}: {description} (optional, not found)")
            else:
                print(f"❌ {file}: {description} (MISSING)")
                all_exist = False
    
    if all_exist:
        print("\n✅ Code files validation PASSED\n")
    else:
        print("\n❌ Some required files are missing\n")
    
    return all_exist

def main():
    print("\n" + "="*60)
    print("COMPETITION SUBMISSION VALIDATION")
    print("="*60 + "\n")
    
    # Load data
    try:
        predictions_df = pd.read_csv('predictions.csv')
        print("✓ predictions.csv loaded successfully")
    except Exception as e:
        print(f"❌ FAIL: Cannot load predictions.csv: {e}")
        return
    
    try:
        test_df = pd.read_csv('atm_transactions_test.csv')
        print("✓ atm_transactions_test.csv loaded successfully\n")
    except Exception as e:
        print(f"❌ FAIL: Cannot load atm_transactions_test.csv: {e}")
        return
    
    # Run validations
    schema_ok = validate_schema(predictions_df)
    completeness_ok = validate_completeness(predictions_df, test_df)
    quality_ok = validate_data_quality(predictions_df)
    code_ok = validate_code_files()
    
    # Final verdict
    print("="*60)
    print("FINAL VALIDATION RESULTS")
    print("="*60)
    
    if schema_ok and completeness_ok and quality_ok and code_ok:
        print("\n✅ ALL VALIDATIONS PASSED!")
        print("\nYour submission is ready for competition!")
        print("\nSubmit these files:")
        print("  1. predictions.csv")
        print("  2. train.py")
        print("  3. predict.py")
        print("  4. requirements.txt")
        print("  5. README.md")
        print("\n" + "="*60)
    else:
        print("\n❌ VALIDATION FAILED")
        print("\nPlease fix the issues above before submitting.")
        print("="*60)

if __name__ == "__main__":
    main()
