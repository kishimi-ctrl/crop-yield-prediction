"""
Crop Yield Prediction Model - Tuned Version
=============================================
Predicts maize yield (bags per hectare) using Random Forest Regressor.
With hyperparameter tuning to reduce overfitting.
"""

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Load data
print("=" * 60)
print("CROP YIELD PREDICTION MODEL (TUNED)")
print("=" * 60)

df = pd.read_csv("farm_data_v2.csv")
print(f"\nDataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# =============================================================================
# FEATURE SELECTION - Top features based on correlation analysis
# =============================================================================
print("\n" + "=" * 60)
print("FEATURE SELECTION")
print("=" * 60)

# Select features (top predictors from EDA)
numeric_features = [
    "is_irrigated",
    "flooding_occurrence",
    "total_rainfall_mm",
    "water_retention_score",
    "inorganic_fert_qty_kg",
    "organic_fert_qty_kg",
    "pest_attack_occurred",
    "weeks_before_attack",
    "farm_size_ha",
    "experience_years",
    "weeding_frequency",
    "pesticide_used",
    "irrigation_vol_ltrs",
]

categorical_features = [
    "seed_type",
    "maize_variety",
    "soil_type",
]

# Encode categorical variables
df_encoded = df.copy()
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col + "_encoded"] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")  # type: ignore[arg-type]

feature_cols = numeric_features + [col + "_encoded" for col in categorical_features]

print(f"\nSelected {len(feature_cols)} features")

# Prepare X and y
X = df_encoded[feature_cols]
y = df_encoded["yield_bags_per_ha"]

# =============================================================================
# TRAIN-TEST SPLIT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING (GridSearchCV)")
print("=" * 60)

# Define parameter grid - focusing on reducing overfitting
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 7, 10, None],
    "min_samples_split": [5, 10, 15],
    "min_samples_leaf": [2, 4, 6],
    "max_features": ["sqrt", "log2", 0.5],
}

# Use smaller grid for faster tuning
param_grid_small = {
    "n_estimators": [50, 100],
    "max_depth": [5, 7, 10],
    "min_samples_split": [10, 15],
    "min_samples_leaf": [4, 6],
    "max_features": ["sqrt", 0.5],
}

print("\nSearching for best hyperparameters...")
base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    base_model, param_grid_small, cv=5, scoring="r2", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV R² score: {grid_search.best_score_:.4f}")

# Get best model
best_model = grid_search.best_estimator_

# =============================================================================
# MODEL EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Predictions
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

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

# Overfitting gap
print("\n--- Overfitting Analysis ---")
print(f"  Train R²: {train_r2:.4f}")
print(f"  Test R²:  {test_r2:.4f}")
print(f"  Gap:      {train_r2 - test_r2:.4f} (lower is better)")

# Cross-validation on best model
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="r2")
print(
    f"\n  Cross-validation R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})"
)

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

feature_importance = pd.DataFrame(
    {"feature": feature_cols, "importance": best_model.feature_importances_}
).sort_values("importance", ascending=False)

print("\nTop 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")

# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Figure 1: Feature Importance
fig1, ax1 = plt.subplots(figsize=(10, 8))
top_features = feature_importance.head(15)
cmap = plt.colormaps["RdYlGn"]
colors = cmap(top_features["importance"] / top_features["importance"].max())
ax1.barh(range(len(top_features)), top_features["importance"], color=colors)
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features["feature"])
ax1.invert_yaxis()
ax1.set_xlabel("Importance")
ax1.set_title(
    "Feature Importance (Tuned Random Forest)", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: feature_importance.png")

# Figure 2: Predicted vs Actual
fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.scatter(
    y_test, y_test_pred, alpha=0.6, edgecolors="k", linewidth=0.5, c="steelblue"
)
min_val = min(np.min(y_test), np.min(y_test_pred))
max_val = max(np.max(y_test), np.max(y_test_pred))
ax2.plot(
    [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction"
)
ax2.set_xlabel("Actual Yield (bags/ha)", fontsize=12)
ax2.set_ylabel("Predicted Yield (bags/ha)", fontsize=12)
ax2.set_title(
    f"Predicted vs Actual Yield (R² = {test_r2:.3f})", fontsize=14, fontweight="bold"
)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: predicted_vs_actual.png")

# Figure 3: Residuals Distribution
fig3, ax3 = plt.subplots(figsize=(8, 6))
residuals = y_test - y_test_pred
ax3.hist(residuals, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
ax3.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Error")
ax3.axvline(
    residuals.mean(),
    color="green",
    linestyle="-",
    linewidth=2,
    label=f"Mean: {residuals.mean():.2f}",
)
ax3.set_xlabel("Residual (Actual - Predicted)", fontsize=12)
ax3.set_ylabel("Frequency", fontsize=12)
ax3.set_title("Residuals Distribution", fontsize=14, fontweight="bold")
ax3.legend()
plt.tight_layout()
plt.savefig("residuals_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: residuals_distribution.png")

# Figure 4: Model Comparison (Before vs After Tuning)
fig4, ax4 = plt.subplots(figsize=(10, 5))
models = ["Original\nModel", "Tuned\nModel"]
train_r2s = [0.9357, train_r2]
test_r2s = [0.7607, test_r2]
x = np.arange(len(models))
width = 0.35
bars1 = ax4.bar(x - width / 2, train_r2s, width, label="Train R²", color="#3498db")
bars2 = ax4.bar(x + width / 2, test_r2s, width, label="Test R²", color="#e74c3c")
ax4.set_ylabel("R² Score")
ax4.set_title(
    "Model Performance: Before vs After Tuning", fontsize=14, fontweight="bold"
)
ax4.set_xticks(x)
ax4.set_xticklabels(models)
ax4.legend()
ax4.set_ylim(0, 1)
# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax4.annotate(
        f"{height:.3f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
    )
for bar in bars2:
    height = bar.get_height()
    ax4.annotate(
        f"{height:.3f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
    )
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: model_comparison.png")

print("\n" + "=" * 60)
print("MODEL TUNING COMPLETE!")
print("=" * 60)
print("\nSummary:")
print(f"  - Best Parameters: {grid_search.best_params_}")
print(f"  - Test R² Score: {test_r2:.4f} (was 0.7607)")
print(f"  - Overfitting Gap: {train_r2 - test_r2:.4f} (was 0.175)")
print(f"  - Test MAE: {test_mae:.2f} bags/ha")

# Save best model and label encoders for the app
os.makedirs("models", exist_ok=True)
with open("models/rf_model_tuned.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
print("\n  Saved: models/rf_model_tuned.pkl")
print("  Saved: models/label_encoders.pkl")

# Save results to CSV for documentation
results_df = pd.DataFrame(
    {
        "Metric": [
            "Train R²",
            "Test R²",
            "Test MAE",
            "Test RMSE",
            "CV R² Mean",
            "Overfitting Gap",
        ],
        "Original Model": [0.9357, 0.7607, 3.21, 4.01, 0.733, 0.175],
        "Tuned Model": [
            train_r2,
            test_r2,
            test_mae,
            test_rmse,
            cv_scores.mean(),
            train_r2 - test_r2,
        ],
    }
)
results_df.to_csv("model_results_comparison.csv", index=False)
print("\n  Saved: model_results_comparison.csv")
