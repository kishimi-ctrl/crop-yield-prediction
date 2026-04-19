"""
Crop Yield Prediction Model
============================
Predicts maize yield (bags per hectare) using Random Forest Regressor.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load data
print("="*60)
print("CROP YIELD PREDICTION MODEL")
print("="*60)

df = pd.read_csv('farm_data_v2.csv')
print(f"\nDataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# =============================================================================
# FEATURE SELECTION - Top features based on correlation analysis
# =============================================================================
print("\n" + "="*60)
print("FEATURE SELECTION")
print("="*60)

# Select features (top predictors from EDA)
# Numeric features with high correlation to yield
numeric_features = [
    'is_irrigated',
    'flooding_occurrence',
    'total_rainfall_mm',
    'water_retention_score',
    'inorganic_fert_qty_kg',
    'organic_fert_qty_kg',
    'pest_attack_occurred',
    'weeks_before_attack',
    'farm_size_ha',
    'experience_years',
    'weeding_frequency',
    'pesticide_used',
    'irrigation_vol_ltrs',
]

# Categorical features to encode
categorical_features = [
    'seed_type',
    'maize_variety',
    'soil_type',
]

# Encode categorical variables
df_encoded = df.copy()
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")  # type: ignore[arg-type]

# Final feature list
feature_cols = numeric_features + [col + '_encoded' for col in categorical_features]

print(f"\nSelected {len(feature_cols)} features:")
for f in feature_cols:
    print(f"  - {f}")

# Prepare X and y
X = df_encoded[feature_cols]
y = df_encoded['yield_bags_per_ha']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# =============================================================================
# TRAIN-TEST SPLIT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# =============================================================================
# MODEL TRAINING
# =============================================================================
print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

# Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest Regressor...")
model.fit(X_train, y_train)
print("Training complete!")

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"\nCross-validation R² scores: {cv_scores.round(3)}")
print(f"Mean CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# =============================================================================
# EVALUATION
# =============================================================================
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Training metrics
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Test metrics
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("\n--- Training Set Performance ---")
print(f"  R² Score:  {train_r2:.4f}")
print(f"  MAE:       {train_mae:.2f} bags/ha")
print(f"  RMSE:      {train_rmse:.2f} bags/ha")

print("\n--- Test Set Performance ---")
print(f"  R² Score:  {test_r2:.4f}")
print(f"  MAE:       {test_mae:.2f} bags/ha")
print(f"  RMSE:      {test_rmse:.2f} bags/ha")

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")

# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Figure 1: Feature Importance
fig1, ax1 = plt.subplots(figsize=(10, 8))
top_features = feature_importance.head(15)
cmap = plt.colormaps["RdYlGn"]
colors = cmap(top_features['importance'] / top_features['importance'].max())
ax1.barh(range(len(top_features)), top_features['importance'], color=colors)
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features['feature'])
ax1.invert_yaxis()
ax1.set_xlabel('Importance')
ax1.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: feature_importance.png")

# Figure 2: Predicted vs Actual
fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
# Perfect prediction line
min_val = min(np.min(y_test), np.min(y_test_pred))
max_val = max(np.max(y_test), np.max(y_test_pred))
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Yield (bags/ha)', fontsize=12)
ax2.set_ylabel('Predicted Yield (bags/ha)', fontsize=12)
ax2.set_title(f'Predicted vs Actual Yield (R² = {test_r2:.3f})', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: predicted_vs_actual.png")

# Figure 3: Residuals Distribution
fig3, ax3 = plt.subplots(figsize=(8, 6))
residuals = y_test - y_test_pred
ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax3.axvline(residuals.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: {residuals.mean():.2f}')
ax3.set_xlabel('Residual (Actual - Predicted)', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
ax3.legend()
plt.tight_layout()
plt.savefig('residuals_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: residuals_distribution.png")

print("\n" + "="*60)
print("MODEL COMPLETE!")
print("="*60)
print("\nSummary:")
print(f"  - Test R² Score: {test_r2:.4f} ({test_r2*100:.1f}% variance explained)")
print(f"  - Test MAE: {test_mae:.2f} bags/ha average error")
print(f"  - Model can predict yield within ~{test_rmse:.1f} bags/ha RMSE")
print("\nVisualizations saved:")
print("  - feature_importance.png")
print("  - predicted_vs_actual.png")
print("  - residuals_distribution.png")
