# Challenge 3: Model Explainability and Operational Insights

## Overview
This analysis uses explainability tools (SHAP, permutation importance, feature importance) to identify the main drivers of ATM cash withdrawals and translate them into actionable operational insights for cash planning and replenishment decisions.

## Baseline Models
We use the trained Random Forest and Gradient Boosting models from Challenge 2:
- **Random Forest Regressor**: 100 trees, max_depth=15, trained on 6+ feature categories
- **Gradient Boosting Regressor**: 100 estimators, learning_rate=0.1, max_depth=5
- **Features**: 60+ features including lag features, rolling statistics, calendar effects, ATM metadata, and replenishment patterns

## Validation Approach
- **Time-series split**: Last 14 days of training data as validation set
- **Sample-based analysis**: Used 500-1000 sample subsets for computationally expensive SHAP calculations
- **Cross-validation**: Permutation importance with 10 repeats for robustness

## Parameters Used

### Feature Importance
- **Method**: Built-in Random Forest feature_importances_ (Gini importance)
- **Top features analyzed**: 20-25 most important features

### Permutation Importance
- **n_repeats**: 10
- **Sample size**: 1000 validation samples
- **Metric**: RMSE on withdrawal amount predictions

### SHAP Analysis
- **Explainer**: TreeExplainer (optimized for tree-based models)
- **Sample size**: 500 validation samples
- **Visualization types**: 
  - Summary plot (global feature impact with directionality)
  - Bar plot (mean absolute SHAP values)
  - Dependence plots (feature interaction effects for top 6 features)

## Running the Analysis

## Key Findings Preview

### Global Drivers
- Lag features: (yesterday's transactions): 40-50% of total importance
- Rolling statistics: (7-day averages): 15-20% of importance
- Calendar effects: (holidays, weekends, payday): 10-15% of importance

### Local Drivers
- Regional variations: Capital and Ahmadi show distinct patterns
- Location types: Shopping malls vs. residential areas have different demand profiles
- ATM characteristics: New ATMs (< 90 days) show higher variability

### Operational Insights
- Implement daily forecasting based on previous day's activity
- Schedule replenishments considering weekly patterns (especially Friday preparation)
- Increase cash buffers during month-end/start periods
- Apply region-specific strategies (e.g., Capital region needs 20% more capacity)

## Dependencies
See `challenge3_requirements.txt` for complete list.

## Notes
- Analysis requires trained models from Challenge 2 (`challenge2_models.pkl`)
- Uses same feature engineering pipeline as Challenge 2
- SHAP calculations are computationally intensive; sample sizes can be adjusted for faster runtime
