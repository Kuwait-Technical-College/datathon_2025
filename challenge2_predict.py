"""
Challenge 2: Advanced ATM Cash Demand Forecasting - Prediction Script

This script loads trained models and generates predictions for the test set.
"""

import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CHALLENGE 2: GENERATING PREDICTIONS")
print("="*60)

# Load test data
print("\n1. Loading datasets...")
test_df = pd.read_csv('atm_transactions_test.csv')
test_df['dt'] = pd.to_datetime(test_df['dt'])

calendar_df = pd.read_csv('calendar.csv')
calendar_df['dt'] = pd.to_datetime(calendar_df['dt'])

# Fill NaN values in holiday_name with 'Normal Day'
if 'holiday_name' in calendar_df.columns:
    calendar_df['holiday_name'] = calendar_df['holiday_name'].fillna('Normal Day')

metadata_df = pd.read_csv('atm_metadata.csv')
# Use actual column names from CSV
if 'installed_date' in metadata_df.columns:
    metadata_df['installed_date'] = pd.to_datetime(metadata_df['installed_date'])
if 'decommissioned_date' in metadata_df.columns:
    metadata_df['decommissioned_date'] = pd.to_datetime(metadata_df['decommissioned_date'])

replenishment_df = pd.read_csv('cash_replenishment.csv')
# Convert date column
replenishment_df['dt'] = pd.to_datetime(replenishment_df['dt'])

# Also load training data for lag features
train_df = pd.read_csv('atm_transactions_train.csv')
train_df['dt'] = pd.to_datetime(train_df['dt'])

print(f"   Test data: {test_df.shape}")
print(f"   Calendar: {calendar_df.shape}")
print(f"   Metadata: {metadata_df.shape}")
print(f"   Replenishment: {replenishment_df.shape}")

# Feature engineering function
def create_features_for_prediction(df, train_df, calendar_df, metadata_df, replenishment_df):
    """Create features for test data using historical data for lags"""
    features_df = df.copy()
    
    # Combine train and test for lag calculation
    combined_df = pd.concat([train_df, df], ignore_index=True).sort_values(['atm_id', 'dt'])
    
    # 1. Calendar features
    calendar_cols = [col for col in calendar_df.columns if col != 'dt']
    features_df = features_df.merge(calendar_df[['dt'] + calendar_cols], on='dt', how='left', suffixes=('', '_cal'))
    
    # 2. Time features
    features_df['day_of_week'] = features_df['dt'].dt.dayofweek
    features_df['day_of_month'] = features_df['dt'].dt.day
    features_df['week_of_year'] = features_df['dt'].dt.isocalendar().week.astype(int)
    features_df['month'] = features_df['dt'].dt.month
    features_df['quarter'] = features_df['dt'].dt.quarter
    features_df['is_month_start'] = (features_df['dt'].dt.day <= 5).astype(int)
    features_df['is_month_end'] = (features_df['dt'].dt.day >= 25).astype(int)
    
    # 3. ATM metadata
    metadata_cols = ['atm_id']
    for col in ['region', 'location_type', 'installed_date', 'decommissioned_date']:
        if col in metadata_df.columns:
            metadata_cols.append(col)
    
    features_df = features_df.merge(
        metadata_df[metadata_cols], 
        on='atm_id', 
        how='left',
        suffixes=('', '_meta')
    )
    
    # ATM age
    if 'installed_date' in features_df.columns:
        features_df['atm_age_days'] = (features_df['dt'] - features_df['installed_date']).dt.days
        features_df['atm_age_days'] = features_df['atm_age_days'].fillna(365)
        features_df['is_new_atm'] = (features_df['atm_age_days'] < 90).astype(int)
    else:
        features_df['atm_age_days'] = 365
        features_df['is_new_atm'] = 0
    
    # Decommissioned features
    if 'decommissioned_date' in features_df.columns:
        features_df['days_to_decommission'] = (features_df['decommissioned_date'] - features_df['dt']).dt.days
        features_df['is_near_decommission'] = ((features_df['days_to_decommission'] >= 0) & 
                                                (features_df['days_to_decommission'] <= 90)).astype(int)
        features_df['is_decommissioned'] = (features_df['days_to_decommission'] < 0).astype(int)
        features_df = features_df.drop('days_to_decommission', axis=1)
    else:
        features_df['is_near_decommission'] = 0
        features_df['is_decommissioned'] = 0
    
    # 4. Replenishment features
    if len(replenishment_df) > 0:
        # Store original index to restore order later
        original_index = features_df.index
        features_df = features_df.reset_index(drop=True)
        
        days_since_replen = []
        
        # Use only top 5000 rows for performance
        replen_sorted = replenishment_df.sort_values('dt')
        
        # Process all rows (limit is just for the loop performance)
        for idx in range(min(5000, len(features_df))):
            row = features_df.iloc[idx]
            atm_replen = replen_sorted[
                (replen_sorted['atm_id'] == row['atm_id']) & 
                (replen_sorted['dt'] <= row['dt'])
            ]
            
            if len(atm_replen) > 0:
                days_since = (row['dt'] - atm_replen.iloc[-1]['dt']).days
            else:
                days_since = 999
            days_since_replen.append(days_since)
        
        # For remaining rows, use default value
        remaining_count = len(features_df) - len(days_since_replen)
        if remaining_count > 0:
            days_since_replen.extend([999] * remaining_count)
        
        features_df['days_since_replenishment'] = days_since_replen
        # Recent replenishment (within 1 day) - fresh cash supply
        features_df['recent_replenishment'] = (features_df['days_since_replenishment'] <= 1).astype(int)
    else:
        features_df['days_since_replenishment'] = 999
        features_df['recent_replenishment'] = 0
    
    # 5. Lag features using combined data
    combined_df = combined_df.sort_values(['atm_id', 'dt'])
    for lag in [1, 7, 14]:
        combined_df[f'amount_lag_{lag}'] = combined_df.groupby('atm_id')['total_withdrawn_amount_kwd'].shift(lag)
        combined_df[f'count_lag_{lag}'] = combined_df.groupby('atm_id')['total_withdraw_txn_count'].shift(lag)
    
    # Rolling statistics
    combined_df['amount_rolling_mean_7'] = combined_df.groupby('atm_id')['total_withdrawn_amount_kwd'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    combined_df['amount_rolling_std_7'] = combined_df.groupby('atm_id')['total_withdrawn_amount_kwd'].transform(
        lambda x: x.rolling(window=7, min_periods=1).std().fillna(0)
    )
    
    # Extract lag features for test data - match on both dt and atm_id to avoid duplicates
    # Create a key for matching
    features_df['_merge_key'] = features_df['dt'].astype(str) + '_' + features_df['atm_id'].astype(str)
    combined_df['_merge_key'] = combined_df['dt'].astype(str) + '_' + combined_df['atm_id'].astype(str)
    
    # Get lag features only for test rows - drop duplicates to ensure 1-to-1 matching
    lag_cols = ['_merge_key', 'amount_lag_1', 'count_lag_1', 'amount_lag_7', 'count_lag_7',
                'amount_lag_14', 'count_lag_14', 'amount_rolling_mean_7', 'amount_rolling_std_7']
    lag_features = combined_df[lag_cols].drop_duplicates(subset=['_merge_key'], keep='last').copy()
    
    # Merge using the unique key
    features_df = features_df.merge(lag_features, on='_merge_key', how='left')
    
    # Drop the temporary merge key
    features_df = features_df.drop('_merge_key', axis=1)
    
    # 6. Encode categoricals
    if 'region' in features_df.columns:
        region_dummies = pd.get_dummies(features_df['region'], prefix='region', drop_first=False)
        features_df = pd.concat([features_df, region_dummies], axis=1)
    
    if 'location_type' in features_df.columns:
        location_dummies = pd.get_dummies(features_df['location_type'], prefix='location', drop_first=False)
        features_df = pd.concat([features_df, location_dummies], axis=1)
    
    return features_df

print("\n2. Creating features...")
test_features = create_features_for_prediction(test_df, train_df, calendar_df, metadata_df, replenishment_df)
print(f"   Features created: {test_features.shape}")

# Load models
print("\n3. Loading trained models...")
with open('challenge2_models.pkl', 'rb') as f:
    models = pickle.load(f)

rf_amount = models['rf_amount']
rf_count = models['rf_count']
gb_amount = models['gb_amount']
gb_count = models['gb_count']
feature_cols = models['feature_cols']

print("   ✓ Models loaded")

# Align features
print("\n4. Preparing test features...")
# Ensure all training features exist in test (add missing columns with 0)
for col in feature_cols:
    if col not in test_features.columns:
        test_features[col] = 0

X_test = test_features[feature_cols].fillna(0)

# Filter out any non-numeric columns
X_test = X_test.select_dtypes(include=[np.number])

print(f"   Test features shape: {X_test.shape}")

# Generate predictions
print("\n5. Generating predictions...")
rf_pred_amount = rf_amount.predict(X_test)
rf_pred_count = rf_count.predict(X_test)
gb_pred_amount = gb_amount.predict(X_test)
gb_pred_count = gb_count.predict(X_test)

# Ensemble (simple average)
ensemble_amount = (rf_pred_amount + gb_pred_amount) / 2
ensemble_count = (rf_pred_count + gb_pred_count) / 2

# Ensure non-negative predictions
ensemble_amount = np.maximum(0, ensemble_amount)
ensemble_count = np.maximum(0, ensemble_count)

print(f"   Predictions generated: {len(ensemble_amount)}")

# Create submission file
print("\n6. Creating submission file...")
predictions = pd.DataFrame({
    'dt': test_df['dt'],
    'atm_id': test_df['atm_id'],
    'predicted_withdrawn_kwd': ensemble_amount,
    'predicted_withdraw_count': ensemble_count
})

# Save
predictions.to_csv('challenge2_predictions.csv', index=False)
print(f"   ✓ Saved to challenge2_predictions.csv")

# Verification
print("\n7. Verification...")
print(f"   Test data rows: {len(test_df)}")
print(f"   Prediction rows: {len(predictions)}")
print(f"   Match: {len(test_df) == len(predictions)}")
print(f"   Missing values: {predictions.isna().sum().sum()}")
print(f"\n   Statistics:")
print(f"     Mean amount: {predictions['predicted_withdrawn_kwd'].mean():.2f} KWD")
print(f"     Mean count: {predictions['predicted_withdraw_count'].mean():.2f}")

print("\n" + "="*60)
print("PREDICTIONS COMPLETE")
print("="*60)
