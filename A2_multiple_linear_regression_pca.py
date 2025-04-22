from A2_multiple_linear_regression import *

df_c = df.copy()

# Run full model
print("=============================Original multiple linear regression model=========================")
linearRegression(df_c, detail=True)

# PCA without Standardisation
print("=============================PCA without Standardisation=========================")
linearRegression(df_c, use_pca=True, pca_n_components=2, detail=True)

# PCA with Standardisation
print("=============================PCA with Standardisation=========================")
linearRegression(df_c, scalingType=ScalingType.STANDARD, use_pca=True, pca_n_components=4, detail=True)

# PCA without Standardisation
# linearRegression(df_c, use_pca=True, pca_n_components=2, detail=True)
# linearRegression(df_c, scalingType=ScalingType.STANDARD, use_pca=True, pca_n_components=2, detail=True)


# PCA with Standardisation + backward elimination + transformed Rainfall
print("================= PCA with Standardisation + backward elimination + transformed Rainfall ===========")
"Transformed rainfall "
df_c = df.copy()
df_c['rainfall'] = np.where(df_c['rainfall'] != 0, 1/df_c['rainfall'], 0)
df = df_c
linearRegression(df, scalingType=ScalingType.STANDARD, use_pca=True, pca_n_components=2, detail=True)