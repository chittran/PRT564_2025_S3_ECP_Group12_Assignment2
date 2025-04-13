import pandas as pd
import seaborn as sns
import pathlib
from Assignment2_Explore_relationship import df, Clinical_Sign

# Plot the matrix as a heatmap to check remove multicollinearity
pathlib.Path('output/correlation/').mkdir(parents=True, exist_ok=True) 
corr = df[Clinical_Sign].corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=False,
    annot=True
)
ax.figure.set_size_inches(12, 8)

# customise the labels
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

ax.figure.savefig(f'output/correlation/heatmapCorrel.png')
ax.figure.clf()


