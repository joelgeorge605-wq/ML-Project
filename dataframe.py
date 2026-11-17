import pandas as pd
df=pd.read_csv('loan_approval_dataset.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())