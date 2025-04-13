import numpy as np
import statsmodels.api as sm
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, PowerTransformer
from Assignment2_Explore_relationship import *

Factors = pd.DataFrame(Factors,columns=Clinical_Sign)
def linearRegression(df: DataFrame, prefix = 'Normal', fit_transform = False, zscore_standardisation = False, detail = False):
    # Separate explanatory variables (X) from the response variable (y)
    X = Factors
    y = MarekCase

    if fit_transform:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X.values)
        X = pd.DataFrame(X_std, index=X.index, columns=X.columns)

    if zscore_standardisation:
        scaler = PowerTransformer()
        X_pow = scaler.fit_transform(X.values)
        X = pd.DataFrame(X_pow, index=X.index, columns=X.columns)

    # build the linear regression using statsmodels
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    if detail:
        print(prefix, 'MarekCase')
        print(model.summary())
    else:
        print(f'{prefix}, {'MarekCase'}, {model.rsquared}, {model.rsquared_adj}')

for variable in Clinical_Sign:
    df = Factors.copy()

    # Optimisation #1: Non-linear transformation
    for variable in Clinical_Sign:
        # Log
        df_c = df.copy()
        df_c[variable] = df_c[variable].apply(lambda x: np.log(x) if x != 0 else 0)
        linearRegression(df_c, f'Non-linear transformation (Log) - {variable}')

        # Exponential
        df_c = df.copy()
        df_c[variable] = df_c[variable].apply(np.exp)
        linearRegression(df_c, f'Non-linear transformation (Exponential) - {variable}')

        # Quandratic
        df_c = df.copy()
        df_c[variable] = df_c[variable].apply(lambda x: x**2)
        linearRegression(df_c, f'Non-linear transformation (Quandratic) - {variable}')

        # Reciprocal
        df_c = df.copy()
        df_c[variable] = df_c[variable].apply(lambda x: 1/x if x != 0 else 0)
        linearRegression(df_c, f'Non-linear transformation (Reciprocal) - {variable}')
