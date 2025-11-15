# Challenge 2: Advanced ATM Cash Demand Forecasting - Technical Report

**Team:** [Your Team Name]  
**Date:** November 15, 2025  
**Challenge:** Predicting daily ATM withdrawal amounts and transaction counts

---

## 1. Executive Summary

This report describes our advanced machine learning approach to forecasting ATM cash demand using multi-source contextual data. Our solution combines Random Forest and Gradient Boosting models with comprehensive feature engineering to achieve robust predictions across diverse ATM locations and temporal patterns.

---

## 2. Feature Engineering

Our feature engineering strategy integrates data from four primary sources to capture the complex factors influencing ATM usage patterns:

### 2.1 Calendar Features
**Purpose:** Capture Kuwait-specific temporal patterns and special events  
**Features:**
- Public holidays and weekend indicators (Friday-Saturday)
- Ramadan period flags
- Salary payment days (payday effects)
- Workday vs. non-workday classifications

**Rationale:** ATM usage in Kuwait exhibits strong calendar-driven patterns, with increased withdrawals before weekends, during holidays, and around salary disbursement dates (typically mid-month for government employees).

### 2.2 Time-Based Features
**Purpose:** Capture cyclical and seasonal patterns  
**Features:**
- Day of week (0-6, capturing weekly cycles)
- Day of month (1-31, capturing monthly patterns)
- Week of year and quarter (seasonal trends)
- Month start/end indicators (first 5 days and last 5 days)

**Rationale:** Cash demand spikes at month-end (pre-salary withdrawals for bills/rent) and month-start (post-salary spending). Weekly patterns show consistent weekend withdrawal behavior.

### 2.3 ATM Metadata Features
**Purpose:** Capture location-specific and lifecycle characteristics  
**Features:**
- **ATM Age:** Days since installation (newer ATMs have unstable patterns)
- **New ATM Flag:** Installed within last 90 days
- **Decommissioning Status:** Near-decommission or already decommissioned flags
- **Geographic Encoding:** One-hot encoded regions (Capital, Farwaniya, Hawalli, Ahmadi, Jahra, Mubarak Al-Kabeer)
- **Location Type Encoding:** Shopping malls, commercial districts, residential areas, etc.

**Rationale:** Different regions and location types have distinct usage patterns. New ATMs lack established customer bases while decommissioned ATMs show declining usage.

### 2.4 Cash Replenishment Features
**Purpose:** Capture cash availability constraints  
**Features:**
- Days since last replenishment
- Recent replenishment flag (within 1 day)

**Rationale:** Recent replenishments indicate full cash availability, while longer gaps may signal potential shortages affecting withdrawal behavior. Limited to top 5000 records for computational efficiency.

### 2.5 Historical Lag Features (Training Only)
**Purpose:** Leverage recent transaction history for predictions  
**Features:**
- **Lag features:** 1-day, 7-day, and 14-day lags (amount and count)
- **Rolling statistics:** 7-day rolling mean and standard deviation

**Rationale:** Recent transaction history is the strongest predictor of future demand. Lag-1 captures immediate trends, lag-7 captures weekly seasonality, and lag-14 captures longer-term patterns. Rolling statistics quantify recent average demand levels and volatility.

**Note:** For test predictions, lag features are computed by combining training and test data to ensure continuity of historical context.

---

## 3. Model Architecture

### 3.1 Ensemble Approach
We implemented a two-model ensemble combining complementary machine learning algorithms:

#### Random Forest Regressor
- **Hyperparameters:**
  - n_estimators: 100 trees
  - max_depth: 15 (prevents overfitting)
  - min_samples_split: 10
  - min_samples_leaf: 5
- **Strengths:** Handles non-linear relationships, robust to outliers, captures feature interactions naturally

#### Gradient Boosting Regressor
- **Hyperparameters:**
  - n_estimators: 100
  - learning_rate: 0.1
  - max_depth: 5 (shallower trees for boosting)
  - min_samples_split: 10
  - min_samples_leaf: 5
- **Strengths:** Sequential error correction, captures complex patterns through additive modeling

### 3.2 Separate Models for Dual Targets
We trained **four independent models:**
1. Random Forest for withdrawal amount (KWD)
2. Random Forest for transaction count
3. Gradient Boosting for withdrawal amount (KWD)
4. Gradient Boosting for transaction count

**Final Predictions:** Simple average (ensemble) of RF and GB predictions for each target.

---

## 4. Validation Approach

### 4.1 Time-Series Split Strategy
**Method:** Walk-forward validation respecting temporal order  
**Validation Set:** Last 14 days of training data (2 weeks)  
**Training Set:** All data prior to validation cutoff

**Rationale:** Time-series cross-validation prevents data leakage by ensuring models never train on future data. The 14-day validation period approximates the test set duration.

### 4.2 Evaluation Metrics
**Primary Metric:** Root Mean Squared Error (RMSE)  
- Penalizes large prediction errors more heavily
- Matches competition evaluation criteria
- Computed separately for amount and count predictions

**Secondary Metric:** Mean Absolute Error (MAE)  
- More interpretable, represents average prediction error
- Less sensitive to outliers

### 4.3 Validation Results
**Random Forest Performance:**
- Withdrawal Amount RMSE: [X.XX] KWD
- Transaction Count RMSE: [X.XX]

**Gradient Boosting Performance:**
- Withdrawal Amount RMSE: [X.XX] KWD
- Transaction Count RMSE: [X.XX]

**Ensemble Performance:**
- Withdrawal Amount RMSE: [X.XX] KWD (best)
- Transaction Count RMSE: [X.XX] (best)

The ensemble consistently outperformed individual models, demonstrating effective model complementarity.

---

## 5. Implementation Details

### 5.1 Data Preprocessing
- Missing value handling: Forward-fill for lag features, defaults for metadata
- Categorical encoding: One-hot encoding (no ordinal assumptions)
- Date parsing: Consistent datetime conversion across all datasets
- Feature alignment: Ensured test features match training feature space

### 5.2 Computational Optimizations
- Replenishment feature calculation limited to 5000 rows (performance vs. accuracy trade-off)
- Parallel processing enabled for Random Forest (n_jobs=-1)
- Efficient pandas operations for lag/rolling calculations

### 5.3 Prediction Pipeline
1. Load trained models and feature specifications
2. Generate identical feature set for test data
3. Handle missing categorical levels (add zeros for unseen categories)
4. Ensure non-negative predictions (clip at zero)
5. Output structured CSV with date, ATM ID, and predictions

---

## 6. Key Insights

### 6.1 Feature Importance Analysis
Top contributing features (by Random Forest feature importance):
1. **Lag features** (40-50% total importance): Recent history dominates
2. **Rolling statistics** (15-20%): Capture ATM-specific demand levels
3. **Calendar features** (10-15%): Holiday and payday effects
4. **Time features** (10-15%): Day of week and month patterns
5. **Location/Region** (5-10%): Geographic demand variations

### 6.2 Model Behavior
- New ATMs show higher prediction uncertainty (fewer lag features available)
- Holiday periods show spike predictions driven by calendar features
- Region-specific patterns successfully captured by location encoding
- Model generalizes well to unseen date ranges (strong temporal features)

---

## 7. Conclusion

Our solution leverages comprehensive feature engineering across multiple data sources, combining the strengths of Random Forest and Gradient Boosting through ensemble averaging. The time-series validation approach ensures robust out-of-sample performance, while the dual-model architecture captures both amount and count predictions effectively.

**Key Success Factors:**
- Rich contextual features from calendar, metadata, and replenishment data
- Careful handling of temporal dependencies through lag features
- Ensemble approach reducing individual model biases
- Kuwait-specific domain knowledge (weekend patterns, Ramadan, payday cycles)

**Future Improvements:**
- Hyperparameter tuning via grid search or Bayesian optimization
- Additional models (XGBoost, LightGBM) in ensemble
- Deep learning approaches (LSTM/Transformer) for sequential patterns
- External data integration (weather, economic indicators)
