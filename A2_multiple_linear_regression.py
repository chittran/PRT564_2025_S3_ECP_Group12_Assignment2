from sklearn.decomposition import PCA
import statsmodels.api as sm
from enum import Enum, auto
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from A2_datawrangling import *

class ScalingType(Enum):
    NONE = auto()
    STANDARD = auto()
    POWER = auto()

def linearRegression(df: DataFrame, prefix = 'Normal', scalingType=ScalingType.NONE, use_pca = False, pca_n_components = 2, detail = False):
    # Clean df
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Separate explanatory variables (X) from the response variable (y)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    if scalingType == ScalingType.STANDARD:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X.values)
        X = pd.DataFrame(X_std, index=X.index, columns=X.columns)
    elif scalingType == ScalingType.POWER:
        scaler = PowerTransformer()
        X_pow = scaler.fit_transform(X.values)
        X = pd.DataFrame(X_pow, index=X.index, columns=X.columns)

    if use_pca:
        pca = PCA(n_components=pca_n_components)
        X_pca = pca.fit_transform(X.values)
        X = pd.DataFrame(X_pca, index=X.index, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

    # Split dataset into 60% training and 40% test sets 
    # training and test splits on original data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)

    # build the linear regression using statsmodels
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train).fit()

    X_test = sm.add_constant(X_test)
    y_pred = model.predict(X_test)

    if detail:
        print(prefix, y.name)
        print(model.summary())
    else:
        print(f'{prefix}, {y.name}, {model.rsquared}, {model.rsquared_adj}')
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmse_norm = rmse / (y_test.max() - y_test.min())  # Normalized RMSE

    # Output performance metrics
    print("Model performance:")
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("RMSE (Normalised): ", rmse_norm)

    return model

if __name__ == "__main__":
    # Optimisation #1: Non-linear transformation
    for variable in feature_columns:
        # Log
        df_c = df.copy()
        df_c[variable] = np.where(df_c[variable] != 0, np.log(df_c[variable]), 0)
        linearRegression(df_c, f'Non-linear transformation (Log) - {variable}')

        # Exponential
        df_c = df.copy()
        df_c[variable] = df_c[variable].apply(np.exp)
        linearRegression(df_c, f'Non-linear transformation (Exponential) - {variable}')

        # Quadratic
        df_c = df.copy()
        df_c[variable] = df_c[variable].apply(lambda x: x**2)
        linearRegression(df_c, f'Non-linear transformation (Quadratic) - {variable}')

        # Reciprocal
        df_c = df.copy()
        df_c[variable] = np.where(df_c[variable] != 0, 1/df_c[variable], 0)
        linearRegression(df_c, f'Non-linear transformation (Reciprocal) - {variable}')

    linearRegression(df, detail=True)
