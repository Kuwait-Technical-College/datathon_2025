"""
Challenge 2: Advanced ATM Cash Demand Forecasting - Training Script

This script trains advanced models with rich contextual features from multiple data sources.
"""

import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CHALLENGE 2: ADVANCED MODEL TRAINING")
print("="*60)

# Load all datasets
print("\n1. Loading datasets...")
train_df = pd.read_csv('atm_transactions_train.csv')
train_df['dt'] = pd.to_datetime(train_df['dt'])

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
print("\n" + "="*60)
print("CASH REPLENISHMENT DATA")
print("="*60)
print(f"Shape: {replenishment_df.shape}")
print(f"Columns: {replenishment_df.columns.tolist()}")

# Convert date column
replenishment_df['dt'] = pd.to_datetime(replenishment_df['dt'])
print(f"Date range: {replenishment_df['dt'].min()} to {replenishment_df['dt'].max()}")

print(f"\nTraining data: {train_df.shape}")
print(f"Calendar: {calendar_df.shape}")
print(f"Metadata: {metadata_df.shape}")
print(f"Replenishment: {replenishment_df.shape}")

# Feature engineering function
def create_features(df, calendar_df, metadata_df, replenishment_df, is_training=True):
    """
    Create comprehensive feature set from all data sources
    
    Features are designed to capture:
    - Temporal patterns (weekday, month, holidays)
    - ATM characteristics (age, location, status)
    - Cash flow patterns (recent replenishments, transaction history)
    """
    print("Creating features...")
    
    # Start with base data
    features_df = df.copy()
    
    # 1. CALENDAR FEATURES
    # Why: Public holidays, weekends, salary days, and Ramadan significantly affect ATM usage
    # Kuwait-specific: Friday-Saturday weekend, salary disbursement patterns
    print("  Adding calendar features...")
    # Only merge columns from calendar that don't conflict
    calendar_cols = [col for col in calendar_df.columns if col != 'dt']
    features_df = features_df.merge(calendar_df[['dt'] + calendar_cols], on='dt', how='left', suffixes=('', '_cal'))
    
    # 2. TIME-BASED FEATURES
    # Why: Transaction patterns vary by day of week, week of month, and time of year
    # These capture cyclical behavior (pay cycles, month-end patterns, seasonal trends)
    print("  Adding time-based features...")
    features_df['day_of_week'] = features_df['dt'].dt.dayofweek  # 0=Monday, 6=Sunday
    features_df['day_of_month'] = features_df['dt'].dt.day  # Day number in month (1-31)
    # Use week instead of isocalendar().week for compatibility
    features_df['week_of_year'] = features_df['dt'].dt.isocalendar().week.astype(int)  # ISO week number
    features_df['month'] = features_df['dt'].dt.month  # Month (1-12)
    features_df['quarter'] = features_df['dt'].dt.quarter  # Quarter (1-4)
    
    # Month start/end indicators
    # Why: Cash demand spikes at month start (bills/rent) and end (pre-salary withdrawals)
    features_df['is_month_start'] = (features_df['dt'].dt.day <= 5).astype(int)
    features_df['is_month_end'] = (features_df['dt'].dt.day >= 25).astype(int)
    
    # 3. ATM METADATA FEATURES
    # Why: Location and ATM characteristics affect usage patterns
    # Different regions and location types have distinct demand patterns
    print("  Adding ATM metadata features...")
    # Only merge the columns we need for features
    # Exclude name, latitude, longitude as they're not useful for ML
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
    
    # ATM age (days since installation)
    # Why: New ATMs have unstable patterns as they build customer base
    # Older ATMs have more established, predictable patterns
    if 'installed_date' in features_df.columns:
        # ATM age in days
        features_df['atm_age_days'] = (features_df['dt'] - features_df['installed_date']).dt.days
        features_df['atm_age_days'] = features_df['atm_age_days'].fillna(365)  # Default for missing
        # New ATM indicator (installed within last 3 months)
        features_df['is_new_atm'] = (features_df['atm_age_days'] < 90).astype(int)
    else:
        features_df['atm_age_days'] = 365  # Default value
        features_df['is_new_atm'] = 0
    
    # Decommissioning indicators
    # Why: ATMs near decommissioning or already decommissioned have abnormal patterns
    # Usage drops before removal, and decommissioned ATMs have zero transactions
    if 'decommissioned_date' in features_df.columns:
        # ATMs that will be decommissioned soon may have different patterns
        features_df['days_to_decommission'] = (features_df['decommissioned_date'] - features_df['dt']).dt.days
        features_df['is_near_decommission'] = ((features_df['days_to_decommission'] >= 0) & 
                                                (features_df['days_to_decommission'] <= 90)).astype(int)
        features_df['is_decommissioned'] = (features_df['days_to_decommission'] < 0).astype(int)
        # Drop the days_to_decommission as it's not useful directly
        features_df = features_df.drop('days_to_decommission', axis=1)
    else:
        features_df['is_near_decommission'] = 0
        features_df['is_decommissioned'] = 0
    
    # 4. REPLENISHMENT FEATURES
    # Why: Days since last cash replenishment affects availability and withdrawal behavior
    # Recent replenishments indicate full cash availability; longer gaps may signal shortages
    if len(replenishment_df) > 0:
        print("  Adding replenishment features...")
        # For each transaction date, find days since last replenishment
        # Use only top 5000 rows for performance
        replen_sorted = replenishment_df.sort_values('dt')
        
        days_since_replen = []
        for idx, row in enumerate(features_df.head(5000).iterrows()):
            _, row = row
            atm_replen = replen_sorted[
                (replen_sorted['atm_id'] == row['atm_id']) & 
                (replen_sorted['dt'] <= row['dt'])
            ]
            if len(atm_replen) > 0:
                days_since = (row['dt'] - atm_replen.iloc[-1]['dt']).days
            else:
                days_since = 999  # No replenishment found
            days_since_replen.append(days_since)
        
        # For remaining rows, use default value
        remaining_count = len(features_df) - 5000
        if remaining_count > 0:
            days_since_replen.extend([999] * remaining_count)
        
        features_df['days_since_replenishment'] = days_since_replen
        # Recent replenishment (within 1 day) - fresh cash supply
        features_df['recent_replenishment'] = (features_df['days_since_replenishment'] <= 1).astype(int)
    else:
        print("  Skipping replenishment features (no data)")
        features_df['days_since_replenishment'] = 999
        features_df['recent_replenishment'] = 0
    
    # 5. LAG FEATURES (only for training data with history)
    # Why: Recent transaction history is the strongest predictor of future behavior
    # Lag features capture trends, momentum, and ATM-specific patterns
    # Only created for training data where we have historical context
    if is_training:
        print("  Adding lag features...")
        # Sort by ATM and date
        features_df = features_df.sort_values(['atm_id', 'dt'])
        
        # Create lag features for each ATM
        # lag_1: Yesterday's value (immediate trend)
        # lag_7: Same day last week (weekly seasonality)
        # lag_14: Two weeks ago (longer-term patterns)
        for lag in [1, 7, 14]:
            features_df[f'amount_lag_{lag}'] = features_df.groupby('atm_id')['total_withdrawn_amount_kwd'].shift(lag)
            features_df[f'count_lag_{lag}'] = features_df.groupby('atm_id')['total_withdraw_txn_count'].shift(lag)
        
        # Rolling statistics (7-day window)
        # Why: Captures recent average behavior and volatility for each ATM
        # Mean: Average recent demand level
        # Std: Variability/unpredictability in recent demand
        features_df['amount_rolling_mean_7'] = features_df.groupby('atm_id')['total_withdrawn_amount_kwd'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        features_df['amount_rolling_std_7'] = features_df.groupby('atm_id')['total_withdrawn_amount_kwd'].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        )
    
    # 6. ENCODE CATEGORICAL FEATURES
    # Why: Convert text categories (region, location_type) into numeric features ML models can use
    # One-hot encoding: Each category becomes a binary (0/1) feature
    # drop_first=False: Keep all categories to avoid information loss
    print("  Encoding categorical features...")
    # Region - only if column exists
    if 'region' in features_df.columns:
        region_dummies = pd.get_dummies(features_df['region'], prefix='region', drop_first=False)
        features_df = pd.concat([features_df, region_dummies], axis=1)
    
    # Location type - only if column exists
    if 'location_type' in features_df.columns:
        location_dummies = pd.get_dummies(features_df['location_type'], prefix='location', drop_first=False)
        features_df = pd.concat([features_df, location_dummies], axis=1)
    
    print(f"✓ Feature engineering complete. Total features: {features_df.shape[1]}")
    
    return features_df

print("\n2. Creating features...")
train_features = create_features(train_df, calendar_df, metadata_df, replenishment_df, is_training=True)
print(f"   Features created: {train_features.shape}")

# Prepare training data
print("\n3. Preparing training data...")

# Select feature columns (exclude target variables and non-feature metadata columns)
feature_cols = [col for col in train_features.columns if col not in [
    # Target variables
    'total_withdrawn_amount_kwd', 'total_withdraw_txn_count',
    # Identifiers and dates
    'dt', 'atm_id',
    # Metadata columns that shouldn't be used as features
    'installed_date', 'decommissioned_date', 'name', 'latitude', 'longitude',
    # Categorical columns (we use their dummies instead)
    'region', 'location_type',
    # Calendar columns that might have issues (we use our own time features)
    'year'  # This column from calendar can cause issues
]]

# Create train/validation split
validation_cutoff = train_features['dt'].max() - pd.Timedelta(days=13)
train_mask = train_features['dt'] < validation_cutoff
val_mask = train_features['dt'] >= validation_cutoff

X_train = train_features[train_mask][feature_cols].fillna(0)
y_train_amount = train_features[train_mask]['total_withdrawn_amount_kwd']
y_train_count = train_features[train_mask]['total_withdraw_txn_count']

X_val = train_features[val_mask][feature_cols].fillna(0)
y_val_amount = train_features[val_mask]['total_withdrawn_amount_kwd']
y_val_count = train_features[val_mask]['total_withdraw_txn_count']

# Check for non-numeric columns
non_numeric = X_train.select_dtypes(include=['object', 'datetime64']).columns.tolist()
if non_numeric:
    print(f"   ⚠ WARNING: Non-numeric columns found: {non_numeric}")
    print("   These columns will be excluded...")
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])
    feature_cols = X_train.columns.tolist()

print(f"   Training samples: {len(X_train)}")
print(f"   Validation samples: {len(X_val)}")
print(f"   Final feature count: {len(feature_cols)}")

# Train models
print("\n4. Training models...")

# Random Forest for amount
print("   Training Random Forest (amount)...")
rf_amount = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_amount.fit(X_train, y_train_amount)

# Random Forest for count
print("   Training Random Forest (count)...")
rf_count = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_count.fit(X_train, y_train_count)

# Gradient Boosting for amount
print("   Training Gradient Boosting (amount)...")
gb_amount = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    learning_rate=0.1,
    random_state=42
)
gb_amount.fit(X_train, y_train_amount)

# Gradient Boosting for count
print("   Training Gradient Boosting (count)...")
gb_count = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    learning_rate=0.1,
    random_state=42
)
gb_count.fit(X_train, y_train_count)

# Validation
print("\n5. Validation (on last 14 days)...")

# Predictions on validation set
rf_pred_amount = rf_amount.predict(X_val)
rf_pred_count = rf_count.predict(X_val)
gb_pred_amount = gb_amount.predict(X_val)
gb_pred_count = gb_count.predict(X_val)

# Ensemble (simple average)
ensemble_amount = (rf_pred_amount + gb_pred_amount) / 2
ensemble_count = (rf_pred_count + gb_pred_count) / 2

# Calculate RMSE
rmse_rf_amount = np.sqrt(mean_squared_error(y_val_amount, rf_pred_amount))
rmse_gb_amount = np.sqrt(mean_squared_error(y_val_amount, gb_pred_amount))
rmse_ensemble_amount = np.sqrt(mean_squared_error(y_val_amount, ensemble_amount))

rmse_rf_count = np.sqrt(mean_squared_error(y_val_count, rf_pred_count))
rmse_gb_count = np.sqrt(mean_squared_error(y_val_count, gb_pred_count))
rmse_ensemble_count = np.sqrt(mean_squared_error(y_val_count, ensemble_count))

print(f"\n   Validation RMSE (Amount):")
print(f"     Random Forest: {rmse_rf_amount:.2f}")
print(f"     Gradient Boosting: {rmse_gb_amount:.2f}")
print(f"     Ensemble: {rmse_ensemble_amount:.2f}")

print(f"\n   Validation RMSE (Count):")
print(f"     Random Forest: {rmse_rf_count:.2f}")
print(f"     Gradient Boosting: {rmse_gb_count:.2f}")
print(f"     Ensemble: {rmse_ensemble_count:.2f}")

# Save models
print("\n6. Saving models...")
models = {
    'rf_amount': rf_amount,
    'rf_count': rf_count,
    'gb_amount': gb_amount,
    'gb_count': gb_count,
    'feature_cols': feature_cols
}

with open('challenge2_models.pkl', 'wb') as f:
    pickle.dump(models, f)

print("   ✓ Models saved to challenge2_models.pkl")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
