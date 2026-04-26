"""
app.py
------
Flask web application for the Loan Approval Prediction System.

Routes
------
GET  /            → Home / prediction form
POST /predict     → Accept form data, run model, return result
GET  /charts      → Data visualisation dashboard
GET  /about       → About the project
"""

import os
import pickle
import subprocess
import sys

import numpy as np
from flask import Flask, render_template, request, redirect, url_for

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "loan_data.csv")

# ── Auto-bootstrap on cloud: generate data + train if model doesn't exist ─────
if not os.path.exists(MODEL_PATH):
    print("🔧  model.pkl not found — running bootstrap pipeline …")
    if not os.path.exists(DATA_PATH):
        print("   Generating dataset …")
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "generate_dataset.py")],
                       check=True)
    print("   Training model …")
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "train_model.py")],
                   check=True)
    print("✅  Bootstrap complete.")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model        = bundle["model"]
scaler       = bundle["scaler"]           # None for Random Forest
FEATURE_NAMES = bundle["feature_names"]
MODEL_NAME   = bundle["model_name"]
MODEL_ACC    = bundle["accuracy"]


# ════════════════════════════════════════════════════════════════════════════════
# Encoding maps  (must mirror train_model.py's LabelEncoder order)
# ════════════════════════════════════════════════════════════════════════════════
ENCODE = {
    "Gender":        {"Female": 0, "Male": 1},
    "Married":       {"No": 0,     "Yes": 1},
    "Education":     {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"No": 0,     "Yes": 1},
    "Dependents":    {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
}


def preprocess_input(form):
    """
    Convert raw HTML form data into the numeric feature vector
    expected by the model.
    Returns (feature_array, error_message_or_None).
    """
    errors = []

    # -- Helpers --
    def get_float(field, label, min_val=0, max_val=1e9):
        raw = form.get(field, "").strip()
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"'{label}' must be a valid number.")
            return None
        if val < min_val or val > max_val:
            errors.append(f"'{label}' must be between {min_val} and {max_val}.")
            return None
        return val

    def get_cat(field, label, mapping):
        raw = form.get(field, "").strip()
        if raw not in mapping:
            errors.append(f"Invalid value for '{label}'.")
            return None
        return mapping[raw]

    # -- Collect values --
    gender      = get_cat("Gender",        "Gender",           ENCODE["Gender"])
    married     = get_cat("Married",       "Married",          ENCODE["Married"])
    dependents  = get_cat("Dependents",    "Dependents",       ENCODE["Dependents"])
    education   = get_cat("Education",     "Education",        ENCODE["Education"])
    self_emp    = get_cat("Self_Employed", "Self Employed",    ENCODE["Self_Employed"])
    prop_area   = get_cat("Property_Area", "Property Area",    ENCODE["Property_Area"])

    app_income  = get_float("ApplicantIncome",   "Applicant Income",   0,   81000)
    co_income   = get_float("CoapplicantIncome",  "Co-applicant Income", 0,  41667)
    loan_amount_raw = get_float("LoanAmount", "Loan Amount (₹)", 10000, 999999)
    loan_amount = round(loan_amount_raw / 1000, 3) if loan_amount_raw is not None else None
    loan_term   = get_float("Loan_Amount_Term",    "Loan Term (months)",   12, 480)
    credit_hist = get_float("Credit_History",      "Credit History",       0,  1)

    if errors:
        return None, errors

    # -- Build feature array in the same order as training --
    row = {
        "Gender":            gender,
        "Married":           married,
        "Dependents":        dependents,
        "Education":         education,
        "Self_Employed":     self_emp,
        "ApplicantIncome":   app_income,
        "CoapplicantIncome": co_income,
        "LoanAmount":        loan_amount,   # converted to thousands for the model
        "Loan_Amount_Term":  loan_term,
        "Credit_History":    credit_hist,
        "Property_Area":     prop_area,
    }
    features = np.array([[row[f] for f in FEATURE_NAMES]], dtype=float)
    return features, None


# ════════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    """Render the home / prediction form."""
    return render_template("index.html",
                           model_name=MODEL_NAME,
                           model_acc=f"{MODEL_ACC*100:.2f}")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission and return prediction result."""
    features, errors = preprocess_input(request.form)

    if errors:
        return render_template("index.html",
                               model_name=MODEL_NAME,
                               model_acc=f"{MODEL_ACC*100:.2f}",
                               errors=errors,
                               form_data=request.form)

    # Scale if the selected model requires it (Logistic Regression)
    X = scaler.transform(features) if scaler else features

    prediction   = model.predict(X)[0]               # 0 or 1
    proba        = model.predict_proba(X)[0]         # [prob_0, prob_1]
    confidence   = round(float(proba[prediction]) * 100, 2)
    result_label = "Approved ✅" if prediction == 1 else "Not Approved ❌"
    result_class = "approved"   if prediction == 1 else "rejected"

    # Result messages
    if prediction == 1:
        message = "🎉 Congratulations! Your loan has been approved successfully."
    else:
        message = "Your loan is not approved at this time. You can improve your chances by maintaining a good credit history and stable income."

    return render_template("result.html",
                           result_label=result_label,
                           result_class=result_class,
                           confidence=confidence,
                           message=message,
                           model_name=MODEL_NAME,
                           model_acc=f"{MODEL_ACC*100:.2f}",
                           form_data=request.form)


@app.route("/charts")
def charts():
    """Data visualisation dashboard."""
    chart_dir = os.path.join(BASE_DIR, "static", "charts")
    charts_available = {}
    chart_files = {
        "loan_status_dist":  "Loan Status Distribution",
        "model_comparison":  "Model Accuracy Comparison",
        "confusion_matrix":  "Confusion Matrix",
        "feature_importance":"Feature Importances",
        "income_dist":       "Applicant Income Distribution",
    }
    for fname, label in chart_files.items():
        path = os.path.join(chart_dir, f"{fname}.png")
        if os.path.exists(path):
            charts_available[fname] = label

    return render_template("charts.html",
                           charts=charts_available,
                           model_name=MODEL_NAME,
                           model_acc=f"{MODEL_ACC*100:.2f}")


@app.route("/about")
def about():
    return render_template("about.html",
                           model_name=MODEL_NAME,
                           model_acc=f"{MODEL_ACC*100:.2f}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n🚀  Loan Approval Prediction System")
    print(f"    Model   : {MODEL_NAME}")
    print(f"    Accuracy: {MODEL_ACC*100:.2f} %")
    print(f"    Running on http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
