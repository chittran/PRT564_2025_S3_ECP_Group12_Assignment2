import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Read dataset into a DataFrame
df = pd.read_csv("data/Avian_disease_Dataset.csv", encoding='ISO-8859-1')

# Generate classification labels based on the percentage of Marek's Disease diagnoses.
bins = df["Percentage of Marek's Disease Diagnoses"].quantile([0, 0.25, 0.5, 0.75, 1]).values
labels = ['Very Low', 'Low', 'Moderate', 'High']
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

def get_dataset(outlier_removal='none'):
    X_num = df[numerical_cols].copy()
    X_cat = df[categorical_cols].copy()
    y = target_column.copy()

    if outlier_removal == 'lof':
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        mask = lof.fit_predict(X_num) != -1
        X_num = X_num[mask]
        X_cat = X_cat[mask]
        y = y[mask].reset_index(drop=True)
    elif outlier_removal == 'isolation_forest':
        iso = IsolationForest(contamination=0.05, random_state=42)
        mask = iso.fit_predict(X_num) != -1
        X_num = X_num[mask]
        X_cat = X_cat[mask]
        y = y[mask].reset_index(drop=True)

    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    return X, y.reset_index(drop=True)
