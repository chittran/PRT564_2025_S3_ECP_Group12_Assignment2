import seaborn as sns
import pathlib
from A4_datawrangling import *

# Plot the matrix as a heatmap to check remove multicollinearity
pathlib.Path('output/correlation/').mkdir(parents=True, exist_ok=True) 

dummies = pd.get_dummies(df[["Region", "Season"]], drop_first=True)
cols = [
    "Found dead",
    "Respiratory",
    "Wasting",
    "Nervous",
    "Recumbent",
    "Other/Unknown",
    "Mean temperature",
    "rainfall"
]
numeric = df[cols]
df_corr = pd.concat([numeric, dummies], axis=1)
corr = df_corr.corr()


# corr = df.iloc[:,:-1].corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=False,
    annot=True
)
ax.figure.set_size_inches(24, 16)

# customise the labels
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

ax.figure.savefig(f'output/correlation/heatmapCorrel_A4.png')
ax.figure.clf()


