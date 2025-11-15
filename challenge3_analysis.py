"""
Challenge 3: Model Explainability and Operational Insights

This script uses SHAP (SHapley Additive exPlanations) and other explainability tools
to identify the main drivers of ATM cash withdrawals and provide actionable insights.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import shap
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CHALLENGE 3: MODEL EXPLAINABILITY ANALYSIS")
print("="*80)

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Load trained models
print("\n1. Loading trained models...")
with open('challenge2_models.pkl', 'rb') as f:
    models = pickle.load(f)

rf_amount = models['rf_amount']
rf_count = models['rf_count']
gb_amount = models['gb_amount']
gb_count = models['gb_count']
feature_cols = models['feature_cols']

print("   ✓ Models loaded successfully")

# Load and prepare data
print("\n2. Loading and preparing data...")
train_df = pd.read_csv('atm_transactions_train.csv')
train_df['dt'] = pd.to_datetime(train_df['dt'])

calendar_df = pd.read_csv('calendar.csv')
calendar_df['dt'] = pd.to_datetime(calendar_df['dt'])
if 'holiday_name' in calendar_df.columns:
    calendar_df['holiday_name'] = calendar_df['holiday_name'].fillna('Normal Day')

metadata_df = pd.read_csv('atm_metadata.csv')
if 'installed_date' in metadata_df.columns:
    metadata_df['installed_date'] = pd.to_datetime(metadata_df['installed_date'])
if 'decommissioned_date' in metadata_df.columns:
    metadata_df['decommissioned_date'] = pd.to_datetime(metadata_df['decommissioned_date'])

replenishment_df = pd.read_csv('cash_replenishment.csv')
replenishment_df['dt'] = pd.to_datetime(replenishment_df['dt'])

# Load training features (reuse from challenge 2)
from challenge2_train import create_features

train_features = create_features(train_df, calendar_df, metadata_df, replenishment_df, is_training=True)

# Prepare validation set for analysis
validation_cutoff = train_df['dt'].max() - pd.Timedelta(days=13)
val_mask = train_features['dt'] >= validation_cutoff

X_val = train_features[val_mask][feature_cols].fillna(0)
y_val_amount = train_features[val_mask]['total_withdrawn_amount_kwd']
y_val_count = train_features[val_mask]['total_withdraw_txn_count']

# Filter numeric columns only
X_val = X_val.select_dtypes(include=[np.number])

print(f"   Validation samples: {len(X_val)}")
print(f"   Features: {len(X_val.columns)}")

# ============================================================================
# PART 1: GLOBAL FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 1: GLOBAL FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# 1.1 Random Forest Feature Importance
print("\n1.1 Random Forest Feature Importance (Withdrawal Amount)")
rf_importance = pd.DataFrame({
    'feature': X_val.columns,
    'importance': rf_amount.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(rf_importance.head(20).to_string(index=False))

# Visualize top features
plt.figure(figsize=(14, 8))
top_n = 25
top_features = rf_importance.head(top_n)
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.8)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance', fontsize=14, fontweight='bold')
plt.ylabel('Features', fontsize=12)
plt.title(f'Top {top_n} Most Important Features (Random Forest)', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_rf.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: feature_importance_rf.png")

# 1.2 Feature Category Analysis
print("\n1.2 Feature Category Analysis")
categories = {
    'Lag Features': [c for c in X_val.columns if 'lag' in c and 'rolling' not in c],
    'Rolling Statistics': [c for c in X_val.columns if 'rolling' in c],
    'Time Features': [c for c in X_val.columns if any(x in c for x in ['day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter', 'is_month'])],
    'Calendar/Holiday': [c for c in X_val.columns if any(x in c for x in ['is_weekend', 'is_holiday', 'is_workday', 'is_payday', 'is_ramadan', 'holiday'])],
    'ATM Characteristics': [c for c in X_val.columns if any(x in c for x in ['atm_age', 'is_new_atm', 'decommission'])],
    'Region': [c for c in X_val.columns if c.startswith('region_')],
    'Location Type': [c for c in X_val.columns if c.startswith('location_')],
    'Replenishment': [c for c in X_val.columns if 'replenishment' in c]
}

category_importance = {}
for category, features in categories.items():
    importance_sum = rf_importance[rf_importance['feature'].isin(features)]['importance'].sum()
    category_importance[category] = importance_sum
    print(f"  {category:25s}: {importance_sum:.4f} ({importance_sum/rf_importance['importance'].sum()*100:.1f}%)")

# Visualize category importance
plt.figure(figsize=(12, 8))
sorted_categories = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
categories_list = [c[0] for c in sorted_categories]
importance_list = [c[1] for c in sorted_categories]

plt.barh(range(len(categories_list)), importance_list, color='coral', alpha=0.8)
plt.yticks(range(len(categories_list)), categories_list)
plt.xlabel('Total Importance', fontsize=14, fontweight='bold')
plt.ylabel('Feature Category', fontsize=12)
plt.title('Feature Category Importance for ATM Withdrawal Prediction', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('category_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: category_importance.png")

# ============================================================================
# PART 2: PERMUTATION IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("PART 2: PERMUTATION IMPORTANCE ANALYSIS")
print("="*80)

print("\nCalculating permutation importance (this may take a few minutes)...")
# Use a sample for faster computation
sample_size = min(1000, len(X_val))
X_sample = X_val.sample(n=sample_size, random_state=42)
y_sample = y_val_amount.loc[X_sample.index]

perm_importance = permutation_importance(
    rf_amount, X_sample, y_sample, 
    n_repeats=10, 
    random_state=42,
    n_jobs=-1
)

perm_importance_df = pd.DataFrame({
    'feature': X_val.columns,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}).sort_values('importance', ascending=False)

print("\nTop 20 Features by Permutation Importance:")
print(perm_importance_df.head(20).to_string(index=False))

# Visualize
plt.figure(figsize=(14, 8))
top_perm = perm_importance_df.head(20)
plt.barh(range(len(top_perm)), top_perm['importance'], 
         xerr=top_perm['std'], color='green', alpha=0.7)
plt.yticks(range(len(top_perm)), top_perm['feature'])
plt.xlabel('Permutation Importance', fontsize=14, fontweight='bold')
plt.ylabel('Features', fontsize=12)
plt.title('Top 20 Features by Permutation Importance', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('permutation_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: permutation_importance.png")

# ============================================================================
# PART 3: SHAP ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 3: SHAP (SHapley Additive exPlanations) ANALYSIS")
print("="*80)

print("\nCalculating SHAP values (this may take several minutes)...")
# Use a smaller sample for SHAP (computationally expensive)
shap_sample_size = min(500, len(X_val))
X_shap = X_val.sample(n=shap_sample_size, random_state=42)

# Create SHAP explainer
explainer = shap.TreeExplainer(rf_amount)
shap_values = explainer.shap_values(X_shap)

print("✓ SHAP values calculated")

# 3.1 SHAP Summary Plot (Global Feature Importance)
print("\n3.1 Generating SHAP summary plot...")
plt.figure(figsize=(14, 10))
shap.summary_plot(shap_values, X_shap, max_display=20, show=False)
plt.title('SHAP Feature Importance (Global Impact)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: shap_summary_plot.png")

# 3.2 SHAP Bar Plot (Mean Absolute SHAP Values)
print("\n3.2 Generating SHAP bar plot...")
plt.figure(figsize=(14, 8))
shap.summary_plot(shap_values, X_shap, plot_type="bar", max_display=20, show=False)
plt.title('Mean Absolute SHAP Values (Feature Impact Magnitude)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: shap_bar_plot.png")

# 3.3 SHAP Dependence Plots for Top Features
print("\n3.3 Generating SHAP dependence plots for top 6 features...")
top_features_list = rf_importance.head(6)['feature'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, feature in enumerate(top_features_list):
    if feature in X_shap.columns:
        shap.dependence_plot(
            feature, shap_values, X_shap, 
            ax=axes[idx], show=False
        )
        axes[idx].set_title(f'SHAP Dependence: {feature}', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('shap_dependence_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: shap_dependence_plots.png")

# ============================================================================
# PART 4: REGIONAL AND LOCAL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 4: REGIONAL AND LOCAL ANALYSIS")
print("="*80)

# 4.1 Region-specific analysis
print("\n4.1 Regional Analysis")
region_cols = [c for c in X_val.columns if c.startswith('region_')]
if region_cols:
    region_importance = rf_importance[rf_importance['feature'].isin(region_cols)].sort_values('importance', ascending=False)
    print("\nRegion Importance Ranking:")
    print(region_importance.to_string(index=False))
    
    # Visualize
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(region_importance)), region_importance['importance'], color='purple', alpha=0.7)
    plt.xticks(range(len(region_importance)), 
               [r.replace('region_', '') for r in region_importance['feature']], 
               rotation=45, ha='right')
    plt.ylabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.xlabel('Region', fontsize=12, fontweight='bold')
    plt.title('Regional Importance in ATM Withdrawal Predictions', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('regional_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: regional_importance.png")

# 4.2 Location type analysis
print("\n4.2 Location Type Analysis")
location_cols = [c for c in X_val.columns if c.startswith('location_')]
if location_cols:
    location_importance = rf_importance[rf_importance['feature'].isin(location_cols)].sort_values('importance', ascending=False)
    print("\nLocation Type Importance Ranking:")
    print(location_importance.to_string(index=False))
    
    # Visualize
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(location_importance)), location_importance['importance'], color='teal', alpha=0.7)
    plt.xticks(range(len(location_importance)), 
               [l.replace('location_', '') for l in location_importance['feature']], 
               rotation=45, ha='right')
    plt.ylabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.xlabel('Location Type', fontsize=12, fontweight='bold')
    plt.title('Location Type Importance in ATM Withdrawal Predictions', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('location_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: location_importance.png")

# ============================================================================
# PART 5: OPERATIONAL INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("PART 5: OPERATIONAL INSIGHTS FOR CASH PLANNING")
print("="*80)

# 5.1 Temporal patterns
print("\n5.1 Temporal Pattern Insights")
time_features = [c for c in X_val.columns if any(x in c for x in ['day_of_week', 'day_of_month', 'is_month_start', 'is_month_end'])]
time_importance = rf_importance[rf_importance['feature'].isin(time_features)].sort_values('importance', ascending=False)
print("\nKey Temporal Drivers:")
print(time_importance.head(10).to_string(index=False))

# 5.2 Calendar effects
print("\n5.2 Calendar and Holiday Effects")
calendar_features = [c for c in X_val.columns if any(x in c for x in ['is_weekend', 'is_holiday', 'is_payday', 'is_ramadan'])]
calendar_importance = rf_importance[rf_importance['feature'].isin(calendar_features)].sort_values('importance', ascending=False)
print("\nCalendar Event Impact:")
print(calendar_importance.to_string(index=False))

# 5.3 Create actionable insights summary
print("\n5.3 Generating Actionable Insights Summary...")

insights = {
    'High Priority Actions': [],
    'Medium Priority Actions': [],
    'Strategic Recommendations': []
}

# Analyze top features for insights
top_10_features = rf_importance.head(10)

for _, row in top_10_features.iterrows():
    feature = row['feature']
    importance = row['importance']
    
    if 'lag_1' in feature:
        insights['High Priority Actions'].append(
            f"✓ Yesterday's activity (importance: {importance:.3f}) is the strongest predictor. "
            "Implement daily cash level monitoring and next-day demand forecasting."
        )
    elif 'lag_7' in feature:
        insights['High Priority Actions'].append(
            f"✓ Weekly patterns (importance: {importance:.3f}) strongly influence demand. "
            "Schedule replenishments based on day-of-week trends."
        )
    elif 'rolling' in feature:
        insights['Medium Priority Actions'].append(
            f"✓ Recent average demand (importance: {importance:.3f}) matters. "
            "Use 7-day moving averages for replenishment planning."
        )
    elif 'is_month_end' in feature or 'is_month_start' in feature:
        insights['Strategic Recommendations'].append(
            f"✓ Month-end/start periods (importance: {importance:.3f}) show demand spikes. "
            "Increase cash buffers during these periods."
        )
    elif 'is_weekend' in feature:
        insights['Strategic Recommendations'].append(
            f"✓ Weekend patterns (importance: {importance:.3f}) affect usage. "
            "Ensure Friday cash availability for weekend demand."
        )
    elif 'region_' in feature:
        region_name = feature.replace('region_', '')
        insights['Strategic Recommendations'].append(
            f"✓ {region_name} shows unique patterns (importance: {importance:.3f}). "
            "Apply region-specific replenishment strategies."
        )

# Print insights
print("\n" + "="*80)
print("ACTIONABLE INSIGHTS FOR CASH PLANNING")
print("="*80)

print("\n🔴 HIGH PRIORITY ACTIONS:")
for insight in insights['High Priority Actions']:
    print(f"  {insight}")

print("\n🟡 MEDIUM PRIORITY ACTIONS:")
for insight in insights['Medium Priority Actions']:
    print(f"  {insight}")

print("\n🔵 STRATEGIC RECOMMENDATIONS:")
for insight in insights['Strategic Recommendations']:
    print(f"  {insight}")

# Save insights to file
with open('operational_insights.txt', 'w') as f:
    f.write("OPERATIONAL INSIGHTS FOR ATM CASH PLANNING\n")
    f.write("="*80 + "\n\n")
    
    f.write("HIGH PRIORITY ACTIONS:\n")
    for insight in insights['High Priority Actions']:
        f.write(f"  {insight}\n\n")
    
    f.write("\nMEDIUM PRIORITY ACTIONS:\n")
    for insight in insights['Medium Priority Actions']:
        f.write(f"  {insight}\n\n")
    
    f.write("\nSTRATEGIC RECOMMENDATIONS:\n")
    for insight in insights['Strategic Recommendations']:
        f.write(f"  {insight}\n\n")

print("\n✓ Saved: operational_insights.txt")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE - FILES GENERATED")
print("="*80)
print("\nGenerated Visualizations:")
print("  1. feature_importance_rf.png - Random Forest feature importance")
print("  2. category_importance.png - Feature category breakdown")
print("  3. permutation_importance.png - Permutation-based importance")
print("  4. shap_summary_plot.png - SHAP global feature importance")
print("  5. shap_bar_plot.png - SHAP mean absolute values")
print("  6. shap_dependence_plots.png - SHAP dependence for top features")
print("  7. regional_importance.png - Regional analysis")
print("  8. location_importance.png - Location type analysis")
print("\nGenerated Reports:")
print("  9. operational_insights.txt - Actionable insights summary")

print("\n" + "="*80)
print("All analyses completed successfully!")
print("="*80)
