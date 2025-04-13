import pandas as pd
import seaborn as sns
import pathlib

# Read dataset into a DataFrame
df = pd.read_csv("Avian_Disease_Dataset.csv", encoding='ISO-8859-1')

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,3:-1].values
y = df.iloc[:,-1].values

Clinical_Sign = df.columns[3:-1].values
print(Clinical_Sign)

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


