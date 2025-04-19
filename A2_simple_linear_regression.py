import matplotlib.pyplot as plt
import pathlib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from A2_datawrangling import *

def linear_regression_analysis(df, feature_column, target_column):
    print('======',feature_column,'======')

    # Separate explanatory variables (X) from the response variable (y)
    X = df[[feature_column]].values
    y = df[target_column].values

    # Split dataset into 60% training and 40% test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    # Build and train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Use the model to predict values in the test set
    y_pred = model.predict(X_test)

    # Compute standard performance metrics
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)                 # RMSE
    y_max = y.max()
    y_min = y.min()
    rmse_norm = rmse / (y_max - y_min)  # Normalized RMSE
    r_2 = metrics.r2_score(y_test, y_pred)

    # Output performance metrics
    print("MLP performance:")
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("RMSE (Normalised): ", rmse_norm)
    print("R^2: ", r_2)

    # Output regression equation
    print(f'{target_column} = {model.intercept_:.4f}' +
            f'{" + " if model.coef_[0] >= 0 else " - "} {abs(model.coef_[0]):.4f}({feature_column})')

    # Create scatter plot
    plt.scatter(X, y)
    plt.xlabel(feature_column)
    plt.ylabel(target_column)
    plt.title(f'{feature_column} VS {target_column}')
    
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, linewidth=2)
    # plt.show()

    # Create output directory and save plot
    output_directory = pathlib.Path(f'output/simple_linear_regression/')
    output_directory.mkdir(parents=True, exist_ok=True)  
    safe_feature_name = feature_column.replace(' ', '_')
    safe_feature_name = safe_feature_name.replace('\\', '_')
    safe_feature_name = safe_feature_name.replace('/', '_')
    plt.savefig(output_directory / f'{safe_feature_name}.png')
    plt.clf()  # Clear the current figure

if __name__ == "__main__":
    # Loop through groups and perform regression analysis
    for feature_column in feature_columns:
        linear_regression_analysis(df, feature_column, target_column)
