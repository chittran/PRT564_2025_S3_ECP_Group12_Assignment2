import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Read dataset into a DataFrame
df = pd.read_csv("Avian_Disease_Dataset.csv", encoding='ISO-8859-1')

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,3:-1].values
y = df.iloc[:,-1].values
#print(x,y)

Clinical_Sign = df.columns[3:-1].values
print(Clinical_Sign)

def linear_regression_analysis(sign):
    target_col = 'No of Marek Disease Diagnoses'

    print('======',sign,'======')

    # Select relevant columns
    subset_df = df[[sign, target_col]].dropna()

    X = subset_df[[sign]].values
    y = subset_df[target_col].values

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
    rmse = metrics.mean_squared_error(y_test, y_pred)  # RMSE
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
    print(f'{sign} = {model.intercept_:.4f}' +
            f'{" + " if model.coef_[0] >= 0 else " - "} {abs(model.coef_[0]):.4f}(No of Marek Disease Diagnoses)')

    # Create scatter plot
    plt.scatter(df['No of Marek Disease Diagnoses'], df[sign])
    plt.xlabel("Clinical_Sign")
    plt.ylabel('No of Marek Disease Diagnoses')
    plt.title(f'{sign} VS No of Marek Disease Diagnoses')
    plt.show()
    # Create output directory and save plot
    # output_directory = pathlib.Path(f'output/simple_linear_regression/{group_column}{group_value}')
    # output_directory.mkdir(parents=True, exist_ok=True)  
    # plt.savefig(output_directory / f'{Clinical_Sign}.png')
    # plt.clf()  # Clear the current figure

# Loop through groups and perform regression analysis
for sign in Clinical_Sign:
    linear_regression_analysis(sign)

