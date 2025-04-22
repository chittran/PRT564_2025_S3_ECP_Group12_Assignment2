from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from A2_datawrangling import *

def scree_plot(df: pd.DataFrame, standardize = False, transform_rainfall = False):
    
    X = df.iloc[:, :-1]
    
    #standardize
    if standardize:
        X = StandardScaler().fit_transform(X)
        title = "Scree Plot (Standardized Data)"
    
    elif transform_rainfall:
        df_c = df.copy()
        df_c['rainfall'] = np.where(df_c['rainfall'] != 0, 1/df_c['rainfall'], 0)
        df = df_c

        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]

        title = "Transformed Rainfall + Standardized Data"

    else:
        title = "Scree Plot (Raw Data)"

    # Fit PCA
    pca = PCA(n_components=X.shape[1]).fit(X)
    var_ratio = pca.explained_variance_ratio_
    pcs       = np.arange(1, len(var_ratio) + 1)
    
    plt.plot(pcs, np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(title)
    plt.xticks(pcs)
    plt.grid()
    plt.show()

scree_plot(df, standardize=False, transform_rainfall=False)   #raw data
scree_plot(df, standardize=True, transform_rainfall=False)    #standardized data
scree_plot(df, standardize=False, transform_rainfall=True)    #transformed rainfall + standardized data
