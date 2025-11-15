# Challenge 2: Advanced ATM Cash Demand Forecasting

## Feature Engineering

### 1. Calendar Features
- Public holidays
- weekends (Friday-Saturday in Kuwait)
- Ramadan periods
- Salary cycles (1st, 5th and 25th)

### 2. Time-Based Features
- Day of week, day of month, week of year
- Month, quarter
- Month start/end indicators

### 3. ATM Metadata Features
- Encoding regional column
- Location type (mall, street, bank branch, etc.)
- ATM age (days since installation)
- New ATM indicators (< 90 days old)
- Decommissioned ATMs (already decommissioned)
- Near decommission indicator (within 90 days of decommissioning)

### 4. Cash Replenishment Features
- Days since last replenishment
- Recent replenishment indicator (within 1 day)
- Captures cash-out events affecting availability

### 5. Lag Features
- 1-day, 7-day, and 14-day lags for amount and count
- Captures recent trends and weekly patterns

### 6. Rolling Statistics
- 7-day rolling mean and standard deviation
- Captures short-term volatility and trends

### Models Used

#### 1. Random Forest Regressor
- Handles non-linear relationships
- Robust to missing data and outliers
- Captures feature interactions
- Parameters: 100 trees, max_depth=15, min_samples_split=10, min_samples_leaf=5

#### 2. Gradient Boosting Regressor
- Sequential ensemble learning
- Captures complex patterns
- Reduces bias through boosting
- Parameters: 100 estimators, max_depth=5, learning_rate=0.1, min_samples_split=10, min_samples_leaf=5


### Handling Real-World Data Imperfections

**Missing Records**
- Fill missing lag features with 0
- Use default values for missing metadata (365 days for ATM age)
- Fill missing holiday names with 'Normal Day'
- Handle NaN values in decommissioned_date appropriately

**Column Name Standardization**
- Corrected metadata column names (installed_date, decommissioned_date)
- Properly handled date column conversions
- Ensured consistent column naming across all merges

**New ATMs with Limited History**
- Use rolling averages with min_periods=1
- Leverage metadata features (location type, region)
- Apply ensemble predictions to reduce variance
- Track ATM lifecycle (new vs established vs near-decommission)

#### Reporting Delays
- Use multi-day lag features (1, 7, 14 days)
- Rolling statistics smooth out delay effects

#### Seasonal & Regional Shifts
- Ramadan indicator captures seasonal behavior
- Region encoding captures geographic patterns
- Month/quarter features capture annual cycles
- Calendar year column excluded to prevent data type issues

#### Data Quality Improvement
- Merge suffixes used to prevent duplicate columns (for regiouns)
- We excluded non-feature columns (name, latitude, longitude)

#### Validation Approach
- Last 13 days of training data held out
- RMSE and MAE for both amount and count
- Lag features computed respecting time order
- We have visualized top features and category contributions

### Model Selection Rationale
We chose Random Forest and Gradient Boosting because:
1. Handle mixed data types (continuous, categorical, binary)
2. Better at handling missing values and outliers
4. Capture non-linear patterns in ATM usage
5. Provide feature importance metrics for interpretability


### Technical Challenges
- Fixed week_of_year extraction using `.astype(int)` for compatibility
- Added merge suffixes to prevent duplicate column names (region_x/region_y issues)
- Implemented automatic filtering of object and datetime columns from training data


# Results

## GRADIENT BOOSTING
Amount - RMSE: 184.60 KWD, MAE: 145.42 KWD
Count  - RMSE: 5.73, MAE: 4.53
Average RMSE: 95.16

### VALIDATION RESULTS
Amount - RMSE: 184.60 KWD, MAE: 145.42 KWD
Count  - RMSE: 5.73, MAE: 4.53
Average RMSE: 95.16

## RANDOM FOREST
Amount - RMSE: 186.11 KWD, MAE: 147.09 KWD
Count  - RMSE: 5.87, MAE: 4.65
Average RMSE: 95.99

### compared with the baseline model MA-28  
Amount - RMSE: 262.46 KWD
Count -  8.08

### Top 5 features
1. amount_rolling_mean_7               0.6815
2. total_deposit_txn_count             0.0613
3. days_to_salary                      0.0591
4. amount_rolling_std_7                0.0545
5. amount_lag_1                        0.0263
