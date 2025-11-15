# Model Performance Scores - Challenge 1

## How to View Scores

The model scores are calculated in the Jupyter notebook `challenge1.ipynb`. 

### Running the Evaluation

1. Open `challenge1.ipynb` in Jupyter or VS Code
2. Navigate to the section: **"Model Evaluation and Comparison"**
3. Run the evaluation cells to see:
   - RMSE (Root Mean Squared Error) for each model
   - MAE (Mean Absolute Error) for each model
   - Performance comparison visualizations
   - Ranking of models by accuracy

### What Gets Evaluated

The notebook automatically:
1. **Creates a validation set** from the last 14 days of training data
2. **Generates predictions** for each model on this validation set
3. **Calculates RMSE** for both withdrawal amount and transaction count
4. **Ranks models** from best to worst performance
5. **Visualizes results** with comparison charts

### Models Evaluated

1. **Naive Forecast** - Last value carried forward
2. **MA-7** - 7-day moving average
3. **MA-14** - 14-day moving average
4. **MA-28** - 28-day moving average
5. **Exponential Smoothing** - Trend + weekly seasonality
6. **SARIMA(1,1,1)(1,1,1)[7]** - Statistical time series model
7. **Ensemble (Average of 6)** - Combined predictions from all models

### Metrics Explained

**RMSE (Root Mean Squared Error):**
- Measures average prediction error
- **Lower is better**
- Penalizes large errors more heavily
- Unit: Same as the data (KWD for amount, count for transactions)

**MAE (Mean Absolute Error):**
- Average absolute difference between prediction and actual
- **Lower is better**
- More interpretable than RMSE
- Unit: Same as the data

### Expected Results

Typical ranking (subject to data):
1. **Ensemble** - Usually performs best by combining model strengths
2. **SARIMA** - Best for ATMs with sufficient history and clear patterns
3. **Exponential Smoothing** - Good for capturing weekly seasonality
4. **MA-14** - Solid baseline with 2-week averaging
5. **MA-28** - Smooths more, may lag recent changes
6. **MA-7** - More responsive but can be volatile
7. **Naive** - Simple baseline, often outperformed

### Why Ensemble Wins

The ensemble approach typically achieves the best score because:
- **Diversification**: Different models capture different patterns
- **Error reduction**: Averaging cancels out individual model errors
- **Robustness**: Works well across different ATM types
- **Risk mitigation**: No single model failure impacts final prediction

### Validation Approach

**Time-based validation:**
- Last 14 days of training data held out
- Models trained on earlier data only
- Predictions compared to actual values
- Mimics real competition scenario

**Why this matters:**
- Prevents overfitting to entire training set
- Realistic assessment of future performance
- Same timeframe as actual test predictions (14 days)
- Validates model generalization ability

## Competition Scoring

The competition judge will:
1. Load your `predictions.csv`
2. Compare with actual ground truth values (hidden during competition)
3. Calculate RMSE for withdrawal amount and count
4. Assign final score based on accuracy

Your validation scores provide an estimate of expected competition performance!

---

**Note:** Run the notebook cells to see your actual scores. The validation approach ensures these scores are representative of real test performance.
