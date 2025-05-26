import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Read dataset into a DataFrame
df = pd.read_csv("data/Avian_disease_Dataset.csv", encoding='ISO-8859-1')

# Generate classification labels based on the percentage of Marek's Disease diagnoses.
bins = df["Percentage of Marek's Disease Diagnoses"].quantile([0, 0.25, 0.5, 0.75, 1]).values
labels = ['Very Low', 'Low', 'High', 'Very High']
df["Mareks_Diagnosis_Level"] = pd.cut(
    df["Percentage of Marek's Disease Diagnoses"],
    bins=bins, labels=labels,
    include_lowest=True
)

categorical_cols = ['Region', 'Season']  # từ index nên cần reset index trước
# numerical_cols = [
#     'Found dead', 'Respiratory', 'Wasting', 'Abnormal faeces or other GIT',
#     'Musculoskeletal &/or Lame', 'Non-specific', 'Nervous', 'Recumbent',
#     'Egg drop/total', 'Skin', 'Other/Unknown',
#     'Mean temperature', 'rainfall', 'Rain days more than 1mm',
#     'Sunshine', 'Days of air frost'
# ]

numerical_cols = [
    'Found dead',
    'Respiratory',
    'Wasting',
    'Nervous',
    'Other/Unknown',
    'Mean temperature',
    'rainfall'
]

feature_columns = df[numerical_cols + categorical_cols]
target_column = df["Mareks_Diagnosis_Level"]
