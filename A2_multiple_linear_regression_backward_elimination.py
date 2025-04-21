from A2_multiple_linear_regression import *

df_c = df.copy()

# Run full model
linearRegression(df_c, detail=True)

# Based on p-values printed, remove the WORST (highest p-value > 0.05)
df_c = df_c.drop(columns=['Sunshine'])
df_c = df_c.drop(columns=['Abnormal faeces or other GIT'])
df_c = df_c.drop(columns=['rainfall'])
df_c = df_c.drop(columns=['Skin'])
df_c = df_c.drop(columns=['Rain days more than 1mm'])

linearRegression(df_c)

df_c = df_c.drop(columns=['Recumbent'])
df_c = df_c.drop(columns=['Respiratory'])
df_c = df_c.drop(columns=['Non-specific'])
df_c = df_c.drop(columns=['Other/Unknown'])
df_c = df_c.drop(columns=['Nervous'])
df_c = df_c.drop(columns=['Musculoskeletal &/or Lame'])
df_c = df_c.drop(columns=['Egg drop/total'])
df_c = df_c.drop(columns=['Mean temperature'])
df_c = df_c.drop(columns=['Days of air frost'])


# df_c = df_c.drop(columns=['Found dead'])
# df_c = df_c.drop(columns=['Wasting'])

# linearRegression(df_c, detail=True)

