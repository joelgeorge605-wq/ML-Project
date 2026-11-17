import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load dataset
df = pd.read_csv("loan_approval_dataset.csv")

# 2. Clean column names and categorical values
df.columns = df.columns.str.strip()

for column in ["education", "self_employed", "loan_status"]:
    df[column] = df[column].str.strip()

# 3. Handle the unusual residential asset values
# This treats -100000 as an invalid placeholder (a project assumption).
df["residential_assets_value"] = df[
    "residential_assets_value"
].replace(-100000, float("nan"))

# 4. Separate features and target
X = df.drop(columns=["loan_id", "loan_status"])

y = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

# 5. Identify feature types
numeric_features = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]

categorical_features = [
    "education",
    "self_employed"
]

# 6. Split the dataset (same split for all algorithms)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 7. Create preprocessing pipelines
def create_preprocessor(scale_numeric=False):

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])


# 8. Define algorithms
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    )
}

# 9. Train and evaluate each algorithm
results = []

for name, classifier in models.items():

    scale = name == "Logistic Regression"

    model = Pipeline(steps=[
        ("preprocessor", create_preprocessor(scale)),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label=1)
    recall = recall_score(y_test, predictions, pos_label=1)
    f1 = f1_score(y_test, predictions, pos_label=1)

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Rejected", "Approved"]
    ))

# 10. Display comparison
comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n\nALGORITHM COMPARISON")
print(comparison.to_string(index=False))

# 11. Save comparison results
comparison.to_csv("model_comparison.csv", index=False)

print("\nComparison saved to model_comparison.csv")