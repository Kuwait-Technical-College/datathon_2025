"""
train.py - Train baseline forecasting models for ATM cash demand prediction

This script trains multiple baseline models on the training data.
The models include:
1. Naive (last day value)
2. Moving Average (7, 14, 28 days)
3. Exponential Smoothing
4. SARIMA

Usage:
    python train.py
"""

import pandas as pd
import numpy as np
import warnings
from datetime import timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle

warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare training data"""
    print("Loading training data...")
    train_df = pd.read_csv('atm_transactions_train.csv')
    train_df['dt'] = pd.to_datetime(train_df['dt'])
    print(f"✓ Loaded {len(train_df)} training records")
    return train_df

def train_models(train_df):
    """
    Train and save model parameters/statistics
    For baseline models, we mainly store statistics needed for prediction
    """
    models = {}
    
    print("\n" + "="*60)
    print("Training Baseline Models")
    print("="*60)
    
    # 1. Naive model - store last values per ATM
    print("\n1. Naive Model")
    last_values = train_df.groupby('atm_id').apply(
        lambda x: x.nlargest(1, 'dt')[['total_withdrawn_amount_kwd', 'total_withdraw_txn_count']].iloc[0]
    ).to_dict('index')
    models['naive'] = {
        'last_values': last_values,
        'global_mean_amount': train_df['total_withdrawn_amount_kwd'].mean(),
        'global_mean_count': train_df['total_withdraw_txn_count'].mean()
    }
    print(f"✓ Stored last values for {len(last_values)} ATMs")
    
    # 2. Moving Average models - store averages per ATM
    print("\n2. Moving Average Models")
    for window in [7, 14, 28]:
        ma_values = {}
        for atm_id in train_df['atm_id'].unique():
            atm_data = train_df[train_df['atm_id'] == atm_id].sort_values('dt')
            if len(atm_data) >= window:
                last_n = atm_data.tail(window)
                ma_values[atm_id] = {
                    'amount': last_n['total_withdrawn_amount_kwd'].mean(),
                    'count': last_n['total_withdraw_txn_count'].mean()
                }
            elif len(atm_data) > 0:
                ma_values[atm_id] = {
                    'amount': atm_data['total_withdrawn_amount_kwd'].mean(),
                    'count': atm_data['total_withdraw_txn_count'].mean()
                }
        
        models[f'ma_{window}'] = {
            'values': ma_values,
            'window': window,
            'global_mean_amount': train_df['total_withdrawn_amount_kwd'].mean(),
            'global_mean_count': train_df['total_withdraw_txn_count'].mean()
        }
        print(f"✓ MA-{window}: Computed averages for {len(ma_values)} ATMs")
    
    # 3. Exponential Smoothing - train models for sample ATMs
    print("\n3. Exponential Smoothing")
    es_params = {}
    atm_ids = train_df['atm_id'].unique()
    sample_size = min(100, len(atm_ids))  # Train on sample for speed
    
    for atm_id in atm_ids[:sample_size]:
        atm_data = train_df[train_df['atm_id'] == atm_id].sort_values('dt')
        if len(atm_data) >= 14:
            try:
                model_amount = ExponentialSmoothing(
                    atm_data['total_withdrawn_amount_kwd'],
                    seasonal_periods=7,
                    trend='add',
                    seasonal='add'
                )
                fit_amount = model_amount.fit()
                
                model_count = ExponentialSmoothing(
                    atm_data['total_withdraw_txn_count'],
                    seasonal_periods=7,
                    trend='add',
                    seasonal='add'
                )
                fit_count = model_count.fit()
                
                es_params[atm_id] = {
                    'amount_params': fit_amount.params,
                    'count_params': fit_count.params,
                    'history_amount': atm_data['total_withdrawn_amount_kwd'].values,
                    'history_count': atm_data['total_withdraw_txn_count'].values
                }
            except:
                pass
    
    models['exponential_smoothing'] = {
        'params': es_params,
        'seasonal_periods': 7,
        'global_mean_amount': train_df['total_withdrawn_amount_kwd'].mean(),
        'global_mean_count': train_df['total_withdraw_txn_count'].mean()
    }
    print(f"✓ Trained ES models for {len(es_params)} ATMs")
    
    # 4. Global statistics
    models['global_stats'] = {
        'mean_amount': train_df['total_withdrawn_amount_kwd'].mean(),
        'mean_count': train_df['total_withdraw_txn_count'].mean(),
        'std_amount': train_df['total_withdrawn_amount_kwd'].std(),
        'std_count': train_df['total_withdraw_txn_count'].std(),
        'median_amount': train_df['total_withdrawn_amount_kwd'].median(),
        'median_count': train_df['total_withdraw_txn_count'].median()
    }
    
    return models

def save_models(models):
    """Save trained models to disk"""
    print("\nSaving models...")
    with open('trained_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    print("✓ Models saved to trained_models.pkl")

def main():
    # Load data
    train_df = load_data()
    
    # Train models
    models = train_models(train_df)
    
    # Save models
    save_models(models)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nNext step: Run predict.py to generate forecasts")

if __name__ == "__main__":
    main()
