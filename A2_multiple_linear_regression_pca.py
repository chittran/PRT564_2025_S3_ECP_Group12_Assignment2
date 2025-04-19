from A2_multiple_linear_regression import *

df_c = df.copy()

# Run full model
linearRegression(df_c, detail=True)

# PCA without Standardisation
linearRegression(df_c, use_pca=True, detail=True)

# PCA with Standardisation
linearRegression(df_c, scalingType=ScalingType.STANDARD, use_pca=True, detail=True)