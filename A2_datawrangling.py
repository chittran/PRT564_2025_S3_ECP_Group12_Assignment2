import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Read dataset into a DataFrame
index_columns = ['Year', 'Region', 'Season']
df = pd.read_csv("data/Avian_Disease_Dataset.csv").set_index(index_columns)

df = df.replace([np.inf, -np.inf, np.nan], 0)

feature_columns = df.columns[:-1].values
target_column = df.columns[-1]

# print(feature_columns)
# print(target_column)