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

# Create figure with subplots
fig = plt.figure(figsize=(20, 24))

# 1. TARGET VARIABLE DISTRIBUTION
ax1 = fig.add_subplot(4, 3, 1)
sns.histplot(df['yield_bags_per_ha'], kde=True, ax=ax1, bins=30, color='steelblue')
ax1.axvline(df['yield_bags_per_ha'].mean(), color='red', linestyle='--', label=f'Mean: {df["yield_bags_per_ha"].mean():.1f}')
ax1.axvline(df['yield_bags_per_ha'].median(), color='green', linestyle='--', label=f'Median: {df["yield_bags_per_ha"].median():.1f}')
ax1.set_title('Yield Distribution (Target Variable)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Yield (bags/ha)')
ax1.legend()

# 2. YIELD BY STATE
ax2 = fig.add_subplot(4, 3, 2)
state_order = df.groupby('state')['yield_bags_per_ha'].mean().sort_values(ascending=False).index
sns.boxplot(data=df, x='state', y='yield_bags_per_ha', order=state_order, ax=ax2, palette='coolwarm')
ax2.set_title('Yield Distribution by State', fontsize=12, fontweight='bold')
ax2.set_xlabel('State')
ax2.set_ylabel('Yield (bags/ha)')
ax2.tick_params(axis='x', rotation=45)

# 3. IRRIGATION IMPACT
ax3 = fig.add_subplot(4, 3, 3)
irrigation_labels = {0: 'Rain-fed', 1: 'Irrigated'}
df['irrigation_type'] = df['is_irrigated'].map(irrigation_labels)
sns.boxplot(data=df, x='irrigation_type', y='yield_bags_per_ha', ax=ax3, palette=['#ff7f7f', '#7fbf7f'])
ax3.set_title('Irrigation vs Rain-fed Yield', fontsize=12, fontweight='bold')
ax3.set_xlabel('Irrigation Type')
ax3.set_ylabel('Yield (bags/ha)')

# 4. FLOODING IMPACT
ax4 = fig.add_subplot(4, 3, 4)
flood_labels = {0: 'No Flooding', 1: 'Flooding'}
df['flood_status'] = df['flooding_occurrence'].map(flood_labels)
sns.boxplot(data=df, x='flood_status', y='yield_bags_per_ha', ax=ax4, palette=['#7fbf7f', '#ff7f7f'])
ax4.set_title('Flooding Impact on Yield', fontsize=12, fontweight='bold')
ax4.set_xlabel('Flooding Status')
ax4.set_ylabel('Yield (bags/ha)')

# 5. RAINFALL VS YIELD (Scatter)
ax5 = fig.add_subplot(4, 3, 5)
# Remove zero rainfall (irrigated only)
rain_df = df[df['total_rainfall_mm'] > 0]
sns.scatterplot(data=rain_df, x='total_rainfall_mm', y='yield_bags_per_ha',
                hue='flooding_occurrence', ax=ax5, palette=['green', 'red'], alpha=0.6)
ax5.set_title('Rainfall vs Yield (colored by flooding)', fontsize=12, fontweight='bold')
ax5.set_xlabel('Total Rainfall (mm)')
ax5.set_ylabel('Yield (bags/ha)')
ax5.legend(title='Flooding', labels=['No', 'Yes'])

# 6. WATER RETENTION SCORE
ax6 = fig.add_subplot(4, 3, 6)
sns.boxplot(data=df, x='water_retention_score', y='yield_bags_per_ha', ax=ax6, palette='YlOrRd')
ax6.set_title('Water Retention Score vs Yield', fontsize=12, fontweight='bold')
ax6.set_xlabel('Water Retention Score (1=low, 5=high)')
ax6.set_ylabel('Yield (bags/ha)')

# 7. SEED TYPE IMPACT
ax7 = fig.add_subplot(4, 3, 7)
sns.boxplot(data=df, x='seed_type', y='yield_bags_per_ha', ax=ax7, palette=['#ffafaf', '#7fbf7f'])
ax7.set_title('Seed Type: Local vs Improved', fontsize=12, fontweight='bold')
ax7.set_xlabel('Seed Type')
ax7.set_ylabel('Yield (bags/ha)')

# 8. FERTILIZER COMBINATIONS
ax8 = fig.add_subplot(4, 3, 8)
fert_labels = {(0,0): 'None', (1,0): 'Inorganic Only', (0,1): 'Organic Only', (1,1): 'Both'}
df['fert_combo'] = df.apply(lambda x: fert_labels[(x['inorganic_fert_used'], x['organic_fert_used'])], axis=1)
fert_order = ['None', 'Inorganic Only', 'Organic Only', 'Both']
sns.boxplot(data=df, x='fert_combo', y='yield_bags_per_ha', order=fert_order, ax=ax8, palette='Blues')
ax8.set_title('Fertilizer Combinations vs Yield', fontsize=12, fontweight='bold')
ax8.set_xlabel('Fertilizer Type')
ax8.set_ylabel('Yield (bags/ha)')
ax8.tick_params(axis='x', rotation=20)

# 9. PEST ATTACK IMPACT
ax9 = fig.add_subplot(4, 3, 9)
pest_labels = {0: 'No Pest Attack', 1: 'Pest Attack'}
df['pest_status'] = df['pest_attack_occurred'].map(pest_labels)
sns.boxplot(data=df, x='pest_status', y='yield_bags_per_ha', ax=ax9, palette=['#7fbf7f', '#ff7f7f'])
ax9.set_title('Pest Attack Impact on Yield', fontsize=12, fontweight='bold')
ax9.set_xlabel('Pest Status')
ax9.set_ylabel('Yield (bags/ha)')

# 10. PLANTING MONTH
ax10 = fig.add_subplot(4, 3, 10)
month_order = ['April', 'May', 'June', 'July', 'August']
sns.boxplot(data=df, x='planting_month', y='yield_bags_per_ha', order=month_order, ax=ax10, palette='Spectral')
ax10.set_title('Planting Month vs Yield', fontsize=12, fontweight='bold')
ax10.set_xlabel('Planting Month')
ax10.set_ylabel('Yield (bags/ha)')

# 11. MAIZE VARIETY
ax11 = fig.add_subplot(4, 3, 11)
sns.boxplot(data=df, x='maize_variety', y='yield_bags_per_ha', ax=ax11, palette=['#ffe66d', '#ffaa00'])
ax11.set_title('Maize Variety vs Yield', fontsize=12, fontweight='bold')
ax11.set_xlabel('Maize Variety')
ax11.set_ylabel('Yield (bags/ha)')

# 12. CORRELATION HEATMAP
ax12 = fig.add_subplot(4, 3, 12)
numeric_cols = ['farm_size_ha', 'experience_years', 'water_retention_score',
                'is_irrigated', 'irrigation_vol_ltrs', 'total_rainfall_mm',
                'flooding_occurrence', 'inorganic_fert_used', 'inorganic_fert_qty_kg',
                'organic_fert_used', 'organic_fert_qty_kg', 'weeding_frequency',
                'pest_attack_occurred', 'weeks_before_attack', 'pesticide_used', 'yield_bags_per_ha']
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
            ax=ax12, square=True, cbar_kws={'shrink': 0.8})
ax12.set_title('Correlation Heatmap (Numeric Features)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('data_exploration_visualizations.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nVisualizations saved to 'data_exploration_visualizations.png'")

# Additional: Top Correlations Bar Chart
fig2, ax = plt.subplots(figsize=(10, 6))
correlations = df[numeric_cols].corr()['yield_bags_per_ha'].drop('yield_bags_per_ha').sort_values()
colors = ['#ff4444' if x < 0 else '#44bb44' for x in correlations]
correlations.plot(kind='barh', ax=ax, color=colors)
ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_title('Feature Correlations with Yield', fontsize=14, fontweight='bold')
ax.set_xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('correlation_with_yield.png', dpi=150, bbox_inches='tight')
plt.show()

print("Correlation chart saved to 'correlation_with_yield.png'")
