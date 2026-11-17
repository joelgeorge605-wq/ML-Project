
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv("loan_approval_dataset.csv")

# 2. Clean column names and categorical values
df.columns = df.columns.str.strip()

categorical_columns = [
    "education",
    "self_employed",
    "loan_status"
]

for column in categorical_columns:
    df[column] = df[column].str.strip()

# 3. Handle invalid residential asset values
df["residential_assets_value"] = (
    df["residential_assets_value"].replace(-100000, float("nan"))
)

# 4. Separate input features and target
X = df.drop(columns=["loan_id", "loan_status"])
y = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

# 5. Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

# 6. Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 7. Create Random Forest model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ))
])

# 8. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 9. Train model
model.fit(X_train, y_train)

# 10. Make predictions
y_pred = model.predict(X_test)

# 11. Evaluate model
print("\nMODEL EVALUATION")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Rejected", "Approved"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 12. Save trained model
joblib.dump(model, "loan_approval_model.pkl")

print("\nModel saved successfully!")
