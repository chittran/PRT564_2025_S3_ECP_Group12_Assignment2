from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from A2_datawrangling import *


def PCs_Visualization(df, standardize=False, transform_rainfall=False):

    pca = PCA(n_components=2)       
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    # print(X.columns)

    X_pca = pca.fit_transform(X)  
    title = "PC1 & PC2 on raw data"

    if transform_rainfall and standardize:
        df_c = df.copy()
        df_c['rainfall'] = np.where(df_c['rainfall'] != 0, 1/df_c['rainfall'], 0)
        df = df_c
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        X = StandardScaler().fit_transform(X)
        X_pca = pca.fit_transform(X)
        title = "PC1 & PC2 on Transformed Rainfall + Standardized Data"
    
    # how much variance each PC captures
    explained = pca.explained_variance_ratio_
    print(f"PC1: {explained[0]:.2%}, PC2: {explained[1]:.2%}, Total: {explained[:2].sum():.2%}")

    plt.figure(figsize=(6,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], 
                c=y,        # color by response variable y
                cmap='viridis', alpha=0.7)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.colorbar(label='% Marek’s Diagnoses') 
    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)
    plt.tight_layout()
    plt.show()

PCs_Visualization(df, standardize=False, transform_rainfall=False)   #raw data
PCs_Visualization(df, standardize=True, transform_rainfall=True)    #standardized data + transformed rainfall


def loadings_table(df: pd.DataFrame, k, standardization = False, transformed_reainfall = False, target_last=True): 

    df_c = df.copy()
    
    if transformed_reainfall and 'rainfall' in df_c.columns:
        df_c['rainfall'] = np.where(df_c['rainfall'] != 0, 1 / df_c['rainfall'], 0)

    X = df_c.iloc[:, :-1] if target_last else df_c
    
    if standardization:
        X_proc = StandardScaler().fit_transform(X)
    else:
        X_proc = X.values 
    
    #fit PCA
    pca = PCA(n_components=k).fit(X_proc)
    
    # build loadings table
    cols = [f'PC{i+1}' for i in range(k)]
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=cols
    ).round(3)
    
    print(
        f"PCA loadings (n_components={k}, "
        f"standardization={standardization}, transformed_reainfall={transformed_reainfall}):"
    )
    print(loadings, end='\n\n')

loadings_table(df, k=2, standardization=False, transformed_reainfall=False) # raw data
loadings_table(df, k=8, standardization=True, transformed_reainfall=False) #standardized
loadings_table(df, k=2, standardization=True, transformed_reainfall=True) #transformed rainfall + standardized