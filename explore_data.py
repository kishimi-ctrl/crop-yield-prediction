import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
df = pd.read_csv('farm_data_v2.csv')

print("="*60)
print("FARM YIELD DATA EXPLORATION")
print("="*60)

# Basic info
print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")

# 1. TARGET VARIABLE ANALYSIS
print("\n" + "="*60)
print("1. TARGET VARIABLE ANALYSIS: yield_bags_per_ha")
print("="*60)

print("\nSummary Statistics:")
print(df['yield_bags_per_ha'].describe())

print(f"\nSkewness: {df['yield_bags_per_ha'].skew():.2f}")
print(f"Kurtosis: {df['yield_bags_per_ha'].kurtosis():.2f}")

# 2. GEOGRAPHIC ANALYSIS
print("\n" + "="*60)
print("2. GEOGRAPHIC ANALYSIS")
print("="*60)

print("\n--- Yield by State ---")
state_stats = df.groupby('state')['yield_bags_per_ha'].agg(['mean', 'median', 'std', 'count']).round(2)
state_stats = state_stats.sort_values('mean', ascending=False)
print(state_stats)

print("\n--- Farm Size by State ---")
farm_size_state = df.groupby('state')['farm_size_ha'].agg(['mean', 'median']).round(2)
print(farm_size_state)

# Unique LGAs per state
print("\n--- Number of LGAs per State ---")
lga_count = df.groupby('state')['lga'].nunique()
print(lga_count)

# 3. ENVIRONMENTAL FACTORS
print("\n" + "="*60)
print("3. ENVIRONMENTAL FACTORS")
print("="*60)

print("\n--- Rainfall Statistics ---")
print(f"Mean: {df['total_rainfall_mm'].mean():.2f} mm")
print(f"Range: {df['total_rainfall_mm'].min():.2f} - {df['total_rainfall_mm'].max():.2f} mm")

print("\n--- Irrigation vs Rain-fed ---")
irrigation_stats = df.groupby('is_irrigated')['yield_bags_per_ha'].agg(['mean', 'count']).round(2)
irrigation_stats.index = ['Rain-fed', 'Irrigated']
print(irrigation_stats)

print("\n--- Water Retention Score vs Yield ---")
wr_stats = df.groupby('water_retention_score')['yield_bags_per_ha'].agg(['mean', 'count']).round(2)
print(wr_stats)

print("\n--- Flooding Impact ---")
flood_stats = df.groupby('flooding_occurrence')['yield_bags_per_ha'].agg(['mean', 'count']).round(2)
flood_stats.index = ['No Flooding', 'Flooding']
print(flood_stats)

# 4. FARM MANAGEMENT
print("\n" + "="*60)
print("4. FARM MANAGEMENT")
print("="*60)

print("\n--- Experience vs Yield Correlation ---")
corr_exp = df['experience_years'].corr(df['yield_bags_per_ha'])
print(f"Correlation: {corr_exp:.3f}")

print("\n--- Farm Size vs Yield Correlation ---")
corr_size = df['farm_size_ha'].corr(df['yield_bags_per_ha'])
print(f"Correlation: {corr_size:.3f}")

print("\n--- Fertilizer Usage ---")
print(f"Farms using inorganic fertilizer: {df['inorganic_fert_used'].sum()} ({df['inorganic_fert_used'].mean()*100:.1f}%)")
print(f"Farms using organic fertilizer: {df['organic_fert_used'].sum()} ({df['organic_fert_used'].mean()*100:.1f}%)")

fert_stats = df.groupby(['inorganic_fert_used', 'organic_fert_used'])['yield_bags_per_ha'].agg(['mean', 'count']).round(2)
print("\nYield by Fertilizer Combination:")
print(fert_stats)

print("\n--- Fertilizer Type ---")
fert_type_stats = df[df['inorganic_fert_used']==1].groupby('inorganic_fert_type')['yield_bags_per_ha'].agg(['mean', 'count']).round(2)
print(fert_type_stats)

# 5. CROP CHARACTERISTICS
print("\n" + "="*60)
print("5. CROP CHARACTERISTICS")
print("="*60)

print("\n--- Maize Variety ---")
variety_stats = df.groupby('maize_variety')['yield_bags_per_ha'].agg(['mean', 'median', 'std', 'count']).round(2)
print(variety_stats)

print("\n--- Seed Type ---")
seed_stats = df.groupby('seed_type')['yield_bags_per_ha'].agg(['mean', 'median', 'std', 'count']).round(2)
print(seed_stats)

print("\n--- Planting Month ---")
month_stats = df.groupby('planting_month')['yield_bags_per_ha'].agg(['mean', 'count']).round(2).sort_values('mean', ascending=False)
print(month_stats)

# 6. PEST & DISEASE
print("\n" + "="*60)
print("6. PEST & DISEASE")
print("="*60)

print("\n--- Pest Attack Frequency ---")
pest_count = df['pest_attack_occurred'].value_counts()
print(f"No pest attack: {pest_count[0]} ({pest_count[0]/len(df)*100:.1f}%)")
print(f"Pest attack: {pest_count[1]} ({pest_count[1]/len(df)*100:.1f}%)")

print("\n--- Pest Impact on Yield ---")
pest_impact = df.groupby('pest_attack_occurred')['yield_bags_per_ha'].agg(['mean', 'count']).round(2)
pest_impact.index = ['No Pest Attack', 'Pest Attack']
print(pest_impact)

print("\n--- Pest Types ---")
pest_types = df[df['pest_attack_occurred']==1]['pest_type'].value_counts()
print(pest_types)

print("\n--- Pesticide Usage ---")
pest_chem = df.groupby('pesticide_used')['yield_bags_per_ha'].agg(['mean', 'count']).round(2)
pest_chem.index = ['No Pesticide', 'Pesticide Used']
print(pest_chem)

# 7. FEATURE CORRELATIONS
print("\n" + "="*60)
print("7. FEATURE CORRELATIONS WITH YIELD")
print("="*60)

numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corr()['yield_bags_per_ha'].drop('yield_bags_per_ha').sort_values(ascending=False)
print(correlations.round(3))

print("\n" + "="*60)
print("DATA EXPLORATION COMPLETE")
print("="*60)
