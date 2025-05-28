import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from A4_datawrangling import *

X_columns = df[numerical_cols + categorical_cols + year]
target_column = df["Mareks_Diagnosis_Level"]

X, y = get_dataset(outlier_removal='none')
df_plot = pd.concat([X, y], axis=1)

print (df_plot)
# —————— 1. Trend of “High” cases by Year ——————
# Compute yearly totals and “High” counts
yearly_total = df_plot.groupby('Year').size()
yearly_high  = df[df_plot['Likelihood to get Marek'] == 'High'] \
                  .groupby('Year').size()

# Calculate percent of “High” each year
pct_high = (yearly_high / yearly_total * 100).sort_index()

# Plot as line + optional area under the curve
plt.figure(figsize=(8,4))
sns.lineplot(x=pct_high.index, y=pct_high.values, marker='o')
plt.fill_between(pct_high.index, pct_high.values, alpha=0.3)
plt.title('Trend of “High” Risk (%) by Year')
plt.xlabel('Year')
plt.ylabel('Percent “High”')
plt.xticks(pct_high.index, rotation=45)
plt.tight_layout()
plt.show()


# —————— 2. FacetGrid: Distribution by Season ——————
# Histogram of raw %‐diagnosis (or raw count if you prefer)
g1 = sns.FacetGrid(df, col='Season', col_wrap=2, height=3, sharex=False, sharey=False)
g1.map(sns.histplot, "Percentage of Marek's Disease Diagnoses", bins=15)
g1.set_axis_labels("Percentage of Diagnoses", "Count")
g1.fig.subplots_adjust(top=0.9)
g1.fig.suptitle("Histogram of % Diagnoses by Season")
plt.show()

# Boxplot of raw %‐diagnosis by Season
g2 = sns.FacetGrid(df, col='Season', col_wrap=2, height=3, sharey=True)
g2.map(sns.boxplot, "Season", "Percentage of Marek's Disease Diagnoses", order=['Winter','Spring','Summer','Autumn'])
g2.set_axis_labels("", "Percentage of Diagnoses")
g2.fig.subplots_adjust(top=0.9)
g2.fig.suptitle("Boxplot of % Diagnoses by Season")
plt.show()
