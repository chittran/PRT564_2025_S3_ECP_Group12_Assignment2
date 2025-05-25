import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Read dataset into a DataFrame
index_columns = ['Year', 'Region', 'Season']
df_o = pd.read_csv("data/Avian_disease_Dataset.csv", encoding='ISO-8859-1').set_index(index_columns)

df = df_o.replace([np.inf, -np.inf, np.nan], 0)

# Generate classification labels based on the percentage of Marek's Disease diagnoses.
bins = df["Percentage of Marek's Disease Diagnoses"].quantile([0, 0.25, 0.5, 0.75, 1]).values
labels = ['Very Low', 'Low', 'High', 'Very High']
df["Mareks_Diagnosis_Level"] = pd.cut(
    df["Percentage of Marek's Disease Diagnoses"],
    bins=bins, labels=labels,
    include_lowest=True
)

feature_columns = df.drop(columns=[
    "Percentage of Marek's Disease Diagnoses",
    'Mareks_Diagnosis_Level'
]).values
target_column = df["Mareks_Diagnosis_Level"].values
