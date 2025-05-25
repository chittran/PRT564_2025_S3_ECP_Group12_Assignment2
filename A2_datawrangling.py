import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Read dataset into a DataFrame
index_columns = ['Year', 'Region', 'Season']
df_o = pd.read_csv("data/Avian_disease_Dataset.csv", encoding='ISO-8859-1').set_index(index_columns)

df = df_o.replace([np.inf, -np.inf, np.nan], 0)

# Remove multicollinearity
df = df.drop(columns=['Rain days more than 1mm', 'Days of air frost'])

# # Remove Skin (backward elimination)
# df = df.drop(columns=['Non-specific'])
# df = df.drop(columns=['Skin'])
# df = df.drop(columns=['Egg drop/total'])
# df = df.drop(columns=['Abnormal faeces or other GIT'])
# df = df.drop(columns=['Sunshine'])
# df = df.drop(columns=['Recumbent'])

feature_columns = df.columns[:-1].values
target_column = df.columns[-1]
