"""
predict.py - Generate predictions using trained baseline models

This script loads trained models and generates predictions for the test set.
It creates an ensemble of multiple baseline models for robust forecasting.

Usage:
    python predict.py
"""

import pandas as pd
import numpy as np
import warnings
import pickle
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

def load_models():
    """Load trained models from disk"""
    print("Loading trained models...")
    with open('trained_models.pkl', 'rb') as f:
        models = pickle.load(f)
    print("✓ Models loaded successfully")
    return models

def load_test_data():
    """Load test data"""
    print("\nLoading test data...")
    test_df = pd.read_csv('atm_transactions_test.csv')
    test_df['dt'] = pd.to_datetime(test_df['dt'])
    print(f"✓ Loaded {len(test_df)} test records")
    print(f"  Date range: {test_df['dt'].min()} to {test_df['dt'].max()}")
    print(f"  Unique ATMs: {test_df['atm_id'].nunique()}")
    return test_df

def naive_predict(models, test_df):
    """Generate naive forecasts"""
    print("\n1. Generating Naive forecasts...")
    predictions = []
    
    for atm_id in test_df['atm_id'].unique():
        if atm_id in models['naive']['last_values']:
            last_amount = models['naive']['last_values'][atm_id]['total_withdrawn_amount_kwd']
            last_count = models['naive']['last_values'][atm_id]['total_withdraw_txn_count']
        else:
            last_amount = models['naive']['global_mean_amount']
            last_count = models['naive']['global_mean_count']
        
        atm_test = test_df[test_df['atm_id'] == atm_id]
        for _, row in atm_test.iterrows():
            predictions.append({
                'dt': row['dt'],
                'atm_id': atm_id,
                'predicted_withdrawn_kwd': max(0, last_amount),
                'predicted_withdraw_count': max(0, last_count)
            })
    
    print(f"✓ Generated {len(predictions)} predictions")
    return pd.DataFrame(predictions)

def moving_average_predict(models, test_df, window=14):
    """Generate moving average forecasts"""
    print(f"\n2. Generating MA-{window} forecasts...")
    predictions = []
    model_key = f'ma_{window}'
    
    for atm_id in test_df['atm_id'].unique():
        if atm_id in models[model_key]['values']:
            avg_amount = models[model_key]['values'][atm_id]['amount']
            avg_count = models[model_key]['values'][atm_id]['count']
        else:
            avg_amount = models[model_key]['global_mean_amount']
            avg_count = models[model_key]['global_mean_count']
        
        atm_test = test_df[test_df['atm_id'] == atm_id]
        for _, row in atm_test.iterrows():
            predictions.append({
                'dt': row['dt'],
                'atm_id': atm_id,
                'predicted_withdrawn_kwd': max(0, avg_amount),
                'predicted_withdraw_count': max(0, avg_count)
            })
    
    print(f"✓ Generated {len(predictions)} predictions")
    return pd.DataFrame(predictions)

def exponential_smoothing_predict(models, test_df, train_df):
    """Generate exponential smoothing forecasts"""
    print("\n3. Generating Exponential Smoothing forecasts...")
    predictions = []
    
    for atm_id in test_df['atm_id'].unique():
        # Get training data for this ATM
        atm_train = train_df[train_df['atm_id'] == atm_id].sort_values('dt')
        
        # Get test dates for this ATM
        atm_test = test_df[test_df['atm_id'] == atm_id].sort_values('dt')
        forecast_horizon = len(atm_test)
        
        if len(atm_train) >= 14:
            try:
                # Fit and forecast
                model_amount = ExponentialSmoothing(
                    atm_train['total_withdrawn_amount_kwd'],
                    seasonal_periods=7,
                    trend='add',
                    seasonal='add'
                )
                fit_amount = model_amount.fit()
                forecast_amount = fit_amount.forecast(steps=forecast_horizon).values
                
                model_count = ExponentialSmoothing(
                    atm_train['total_withdraw_txn_count'],
                    seasonal_periods=7,
                    trend='add',
                    seasonal='add'
                )
                fit_count = model_count.fit()
                forecast_count = fit_count.forecast(steps=forecast_horizon).values
            except:
                forecast_amount = [atm_train['total_withdrawn_amount_kwd'].mean()] * forecast_horizon
                forecast_count = [atm_train['total_withdraw_txn_count'].mean()] * forecast_horizon
        else:
            forecast_amount = [models['global_stats']['mean_amount']] * forecast_horizon
            forecast_count = [models['global_stats']['mean_count']] * forecast_horizon
        
        for i, row in enumerate(atm_test.itertuples()):
            predictions.append({
                'dt': row.dt,
                'atm_id': atm_id,
                'predicted_withdrawn_kwd': max(0, forecast_amount[i]),
                'predicted_withdraw_count': max(0, forecast_count[i])
            })
    
    print(f"✓ Generated {len(predictions)} predictions")
    return pd.DataFrame(predictions)

def sarima_predict(models, test_df, train_df):
    """Generate SARIMA forecasts"""
    print("\n4. Generating SARIMA forecasts...")
    predictions = []
    
    for atm_id in test_df['atm_id'].unique():
        atm_train = train_df[train_df['atm_id'] == atm_id].sort_values('dt')
        
        # Get test dates for this ATM
        atm_test = test_df[test_df['atm_id'] == atm_id].sort_values('dt')
        forecast_horizon = len(atm_test)
        
        if len(atm_train) >= 28:
            try:
                model_amount = SARIMAX(
                    atm_train['total_withdrawn_amount_kwd'],
                    order=(1,1,1),
                    seasonal_order=(1,1,1,7)
                )
                fit_amount = model_amount.fit(disp=False, maxiter=50)
                forecast_amount = fit_amount.forecast(steps=forecast_horizon).values
                
                model_count = SARIMAX(
                    atm_train['total_withdraw_txn_count'],
                    order=(1,1,1),
                    seasonal_order=(1,1,1,7)
                )
                fit_count = model_count.fit(disp=False, maxiter=50)
                forecast_count = fit_count.forecast(steps=forecast_horizon).values
            except:
                forecast_amount = [atm_train['total_withdrawn_amount_kwd'].mean()] * forecast_horizon
                forecast_count = [atm_train['total_withdraw_txn_count'].mean()] * forecast_horizon
        else:
            forecast_amount = [models['global_stats']['mean_amount']] * forecast_horizon
            forecast_count = [models['global_stats']['mean_count']] * forecast_horizon
        
        for i, row in enumerate(atm_test.itertuples()):
            predictions.append({
                'dt': row.dt,
                'atm_id': atm_id,
                'predicted_withdrawn_kwd': max(0, forecast_amount[i]),
                'predicted_withdraw_count': max(0, forecast_count[i])
            })
    
    print(f"✓ Generated {len(predictions)} predictions")
    return pd.DataFrame(predictions)

def create_ensemble(predictions_list):
    """Create ensemble from multiple models"""
    print("\n5. Creating ensemble predictions...")
    
    # Average all predictions
    ensemble = predictions_list[0].copy()
    ensemble['predicted_withdrawn_kwd'] = sum(
        pred['predicted_withdrawn_kwd'] for pred in predictions_list
    ) / len(predictions_list)
    ensemble['predicted_withdraw_count'] = sum(
        pred['predicted_withdraw_count'] for pred in predictions_list
    ) / len(predictions_list)
    
    print(f"✓ Ensemble created from {len(predictions_list)} models")
    return ensemble

def save_predictions(predictions, filename='predictions.csv'):
    """Save predictions to CSV"""
    predictions.to_csv(filename, index=False)
    print(f"\n✓ Predictions saved to {filename}")
    
    # Print summary statistics
    print("\nPrediction Summary:")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Mean withdrawal amount: {predictions['predicted_withdrawn_kwd'].mean():.2f} KWD")
    print(f"  Mean withdrawal count: {predictions['predicted_withdraw_count'].mean():.2f}")
    print(f"  Missing values: {predictions.isna().sum().sum()}")

def main():
    # Load models
    models = load_models()
    
    # Load test data
    test_df = load_test_data()
    
    # Load training data (needed for some models)
    train_df = pd.read_csv('atm_transactions_train.csv')
    train_df['dt'] = pd.to_datetime(train_df['dt'])
    
    # Generate predictions from all models
    print("\n" + "="*60)
    print("Generating Predictions")
    print("="*60)
    
    naive_pred = naive_predict(models, test_df)
    ma14_pred = moving_average_predict(models, test_df, window=14)
    es_pred = exponential_smoothing_predict(models, test_df, train_df)
    sarima_pred = sarima_predict(models, test_df, train_df)
    
    # Create ensemble
    ensemble_pred = create_ensemble([naive_pred, ma14_pred, es_pred, sarima_pred])
    
    # Save final predictions
    print("\n" + "="*60)
    print("Saving Predictions")
    print("="*60)
    save_predictions(ensemble_pred, 'predictions.csv')
    
    # Also save individual model predictions
    save_predictions(naive_pred, 'predictions_naive.csv')
    save_predictions(ma14_pred, 'predictions_ma14.csv')
    save_predictions(es_pred, 'predictions_es.csv')
    save_predictions(sarima_pred, 'predictions_sarima.csv')
    
    print("\n" + "="*60)
    print("Prediction Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
