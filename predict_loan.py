
import pandas as pd
import joblib

# Load the trained Random Forest model
model = joblib.load("loan_approval_model.pkl")

print("\n===== LOAN APPROVAL PREDICTION =====")

# Collect applicant details
applicant = {
    "no_of_dependents": int(input("Number of dependents: ")),

    "education": input(
        "Education (Graduate/Not Graduate): "
    ).strip(),

    "self_employed": input(
        "Self employed (Yes/No): "
    ).strip(),

    "income_annum": float(input("Annual income: ")),

    "loan_amount": float(input("Loan amount: ")),

    "loan_term": int(input("Loan term: ")),

    "cibil_score": int(input("CIBIL score: ")),

    "residential_assets_value": float(
        input("Residential assets value: ")
    ),

    "commercial_assets_value": float(
        input("Commercial assets value: ")
    ),

    "luxury_assets_value": float(
        input("Luxury assets value: ")
    ),

    "bank_asset_value": float(
        input("Bank asset value: ")
    )
}

# Convert input into a DataFrame
input_data = pd.DataFrame([applicant])

# Predict loan status
prediction = model.predict(input_data)[0]

print("\n===== PREDICTION RESULT =====")

if prediction == 1:
    print("Loan Status: APPROVED")
else:
    print("Loan Status: REJECTED")

print("\nPrediction completed!")
