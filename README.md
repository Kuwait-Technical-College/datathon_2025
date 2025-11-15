# 1.A Sub-Task 1 - Baseline Forecasting

## Libraries used
- pandas- Data manipulation and time series handling
- numpy - Numerical operations and array computations
- statsmodels: We gout our models from this library (Exponential Smoothing, SARIMA)
- scikit-learn: Model evaluation metrics (RMSE, MAE)
- matplotlib / seaborn: Data visualization and plotting

## Baseline Models
### 1. Naive Forecast
- We have used the last record for each ATM. 
- We have sorted by the most reacent date for each ATM and extracted withdrawn amount and withdrawn count.

### 2. Moving Average (MA-7, MA-14, MA-28)
- We have used the mean of the last (recent) 7, 14 and 28 observations for each ATM. 
- We have computed rolling mean over different window sizes
- This method provides moderate variability but no strong seasonal patterns

### 3. Exponential Smoothing
- This model captures both trend and weekly seasonality in ATM usage
- We have implemented this model with the following parameters:
  - Seasonal periods = 7 (weekly pattern for Kuwait's work week)
  - Trend = additive
  - Seasonal = additive

### 4. SARIMA (Seasonal ARIMA)
- This model uses seasonal AutoRegressive Integrated Moving Average
- Non-seasonal Order (p,d,q) = (1,1,1):
    - `p=1`: AutoRegressive term - uses 1 previous time step to predict current value
    - `d=1`: Differencing order - takes first difference to make series stationary
    - `q=1`: Moving Average term - uses 1 previous forecast error in prediction
- Seasonal Order (P,D,Q,s) = (1,1,1,7):
    - `P=1`: Seasonal AutoRegressive - uses value from 1 week ago (7 days back)
    - `D=1`: Seasonal differencing - removes weekly trend by subtracting last week's value
    - `Q=1`: Seasonal Moving Average - accounts for forecast error from 1 week ago
    - `s=7`: Seasonal period - 7 days for weekly cycle
- We added kuwait-specific details: 7-day seasonal cycle for work week (Sun-Thu) vs weekend (Fri-Sat)

### 5. Ensemble Approach
We decided to combine all models into one final predictions moidel in hope for better accuracy. Use a simple average of all 6 models (Naive, MA-7, MA-14, MA-28, Exponential Smoothing, SARIMA).

## Validation
1. MA-28                     - Avg RMSE: 135.27 (Amount: 262.46 KWD, Count:  8.08)
2. Ensemble (Avg of 6)       - Avg RMSE: 135.50 (Amount: 262.84 KWD, Count:  8.16)
3. SARIMA                    - Avg RMSE: 136.04 (Amount: 263.87 KWD, Count:  8.21)
4. Exponential Smoothing     - Avg RMSE: 139.92 (Amount: 271.08 KWD, Count:  8.76)
5. MA-14                     - Avg RMSE: 146.92 (Amount: 284.96 KWD, Count:  8.88)
6. MA-7                      - Avg RMSE: 150.34 (Amount: 291.63 KWD, Count:  9.06)
7. Naive                     - Avg RMSE: 179.90 (Amount: 349.05 KWD, Count: 10.75)
