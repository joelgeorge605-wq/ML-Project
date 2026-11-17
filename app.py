from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_approval_model.pkl")

model = joblib.load(MODEL_PATH)

# The algorithm that train_model.py saves into loan_approval_model.pkl
DEPLOYED_MODEL = "Random Forest"


# ---------- Display helpers (used by the templates) ----------

def inr(value, decimals=0):
    """Format a number with Indian digit grouping: 2990000 -> ₹29,90,000"""
    n = float(value)
    sign = "-" if n < 0 else ""
    whole, _, frac = f"{abs(n):.{decimals}f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"{sign}₹{whole}" + (f".{frac}" if frac else "")


def inr_short(value):
    """Short Indian form: 29900000 -> ₹2.99 crore, 2400000 -> ₹24 lakh"""
    n = abs(float(value))

    def trim(x):
        return f"{x:.2f}".rstrip("0").rstrip(".")

    if n >= 1e7:
        return f"₹{trim(n / 1e7)} crore"
    if n >= 1e5:
        return f"₹{trim(n / 1e5)} lakh"
    return inr(n)


app.jinja_env.filters["inr"] = inr
app.jinja_env.filters["inr_short"] = inr_short


# ---------- Real numbers for the home page ----------

def load_model_stats():
    """Read the algorithm comparison saved by compare_models.py."""
    path = os.path.join(BASE_DIR, "model_comparison.csv")
    try:
        df = pd.read_csv(path)
        return [
            {
                "name": row["Algorithm"],
                "accuracy": row["Accuracy"] * 100,
                "precision": row["Precision"] * 100,
                "recall": row["Recall"] * 100,
                "f1": row["F1 Score"] * 100,
            }
            for _, row in df.iterrows()
        ]
    except Exception as e:
        print("Could not load model_comparison.csv:", e)
        return []


def count_records():
    path = os.path.join(BASE_DIR, "loan_approval_dataset.csv")
    try:
        return len(pd.read_csv(path))
    except Exception as e:
        print("Could not read dataset:", e)
        return None


MODEL_STATS = load_model_stats()
RECORD_COUNT = count_records()


# Home page
@app.route("/")
def home():
    deployed = next((m for m in MODEL_STATS if m["name"] == DEPLOYED_MODEL), None)
    return render_template(
        "index.html",
        models=MODEL_STATS,
        deployed_name=DEPLOYED_MODEL,
        deployed=deployed,
        record_count=RECORD_COUNT,
    )


# Application page
@app.route("/application")
def application():
    return render_template("application.html")


# Prediction page
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = {
            "no_of_dependents": int(request.form["no_of_dependents"]),
            "education": request.form["education"],
            "self_employed": request.form["self_employed"],
            "income_annum": float(request.form["income_annum"]),
            "loan_amount": float(request.form["loan_amount"]),
            "loan_term": float(request.form["loan_term"]),
            "cibil_score": float(request.form["cibil_score"]),
            "residential_assets_value": float(request.form["residential_assets_value"]),
            "commercial_assets_value": float(request.form["commercial_assets_value"]),
            "luxury_assets_value": float(request.form["luxury_assets_value"]),
            "bank_asset_value": float(request.form["bank_asset_value"])
        }

        # Input validation
        if data["no_of_dependents"] < 0:
            raise ValueError("Dependents cannot be negative.")

        if data["income_annum"] < 0:
            raise ValueError("Income cannot be negative.")

        if data["loan_amount"] <= 0 or data["loan_term"] <= 0:
            raise ValueError("Loan amount and term must be positive.")

        if not 300 <= data["cibil_score"] <= 900:
            raise ValueError("CIBIL score must be between 300 and 900.")

        for field in [
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value"
        ]:
            if data[field] < 0:
                raise ValueError("Asset values cannot be negative.")

        # Convert input into DataFrame
        input_data = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]

        classes = model.named_steps["classifier"].classes_

        approval_probability = probabilities[
            list(classes).index(1)
        ] * 100

        rejection_probability = probabilities[
            list(classes).index(0)
        ] * 100

        if prediction == 1:
            result = "Loan Approved"
            status = "approved"
        else:
            result = "Loan Not Approved"
            status = "rejected"

        # Extra figures shown on the result page
        total_assets = (
            data["residential_assets_value"]
            + data["commercial_assets_value"]
            + data["luxury_assets_value"]
            + data["bank_asset_value"]
        )
        loan_to_income = (
            data["loan_amount"] / data["income_annum"]
            if data["income_annum"] > 0 else None
        )

        return render_template(
            "result.html",
            result=result,
            status=status,
            approval_probability=round(approval_probability, 2),
            rejection_probability=round(rejection_probability, 2),
            data=data,
            total_assets=total_assets,
            loan_to_income=loan_to_income
        )

    except (ValueError, KeyError) as e:
        return render_template(
            "application.html",
            error=str(e),
            form=request.form
        ), 400

    except Exception as e:
        print("Prediction Error:", e)
        return "Something went wrong during prediction.", 500


if __name__ == "__main__":
    app.run(debug=True)
