
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# 1. Load dataset
df = pd.read_csv("loan_approval_dataset.csv")

# 2. Clean dataset
df.columns = df.columns.str.strip()

for column in ["education", "self_employed", "loan_status"]:
    df[column] = df[column].str.strip()

# Treat -100000 as an invalid placeholder.
# This is a project assumption, not a confirmed dataset definition.
df["residential_assets_value"] = df[
    "residential_assets_value"
].replace(-100000, float("nan"))

# 3. Separate input and target
X = df.drop(columns=["loan_id", "loan_status"])

y = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

# 4. Identify feature columns
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

# 5. Keep 20% test data separate.
# Cross-validation will use only the remaining 80%.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Preprocessing function
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

# 7. Define algorithms
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

# 8. Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

results = []

# 9. Validate each algorithm
for name, classifier in models.items():

    scale = name == "Logistic Regression"

    model = Pipeline(steps=[
        ("preprocessor", create_preprocessor(scale)),
        ("classifier", classifier)
    ])

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    results.append({
        "Algorithm": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1 Score": scores["test_f1"].mean(),
        "F1 Std": scores["test_f1"].std()
    })

# 10. Display results
comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n5-FOLD CROSS-VALIDATION RESULTS")
print(comparison.to_string(index=False))

# 11. Save results
comparison.to_csv(
    "cross_validation_results.csv",
    index=False
)

print("\nResults saved to cross_validation_results.csv")
