
import pandas as pd

df = pd.read_csv("loan_approval_dataset.csv")

# Clean column names and text values
df.columns = df.columns.str.strip()

for column in ["education", "self_employed", "loan_status"]:
    df[column] = df[column].str.strip()

print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nNegative residential asset records:")
print(df[df["residential_assets_value"] < 0])

print("\nResidential asset statistics:")
print(df["residential_assets_value"].describe())

# Check loan term values
print("\nLoan term values:")
print(sorted(df["loan_term"].unique()))

# Check education categories
print("\nEducation categories:")
print(df["education"].value_counts())

# Check employment categories
print("\nEmployment categories:")
print(df["self_employed"].value_counts())

# Check loan status distribution
print("\nLoan status distribution:")
print(df["loan_status"].value_counts())

# Check how many negative residential values exist
print("\nNegative residential asset count:")
print((df["residential_assets_value"] < 0).sum())
