"""
generate_dataset.py
-------------------
Generates a realistic synthetic Loan Prediction dataset and saves it as
'loan_data.csv' in the project root.  This mirrors the well-known Kaggle
Loan Prediction dataset (LoanTap / Analytics Vidhya) so the project runs
completely offline without any external download.
"""

import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
N = 614          # same size as the canonical Kaggle dataset

# ── Helper: introduce ~5 % NaN values in a Series ────────────────────────────
def add_missing(arr, pct=0.05):
    idx = np.random.choice(len(arr), size=int(len(arr) * pct), replace=False)
    arr = arr.astype(object)
    arr[idx] = np.nan
    return arr

# ── Categorical features ──────────────────────────────────────────────────────
loan_ids   = [f"LP{str(i).zfill(6)}" for i in range(1, N + 1)]
gender     = np.random.choice(["Male", "Female"], N, p=[0.80, 0.20])
married    = np.random.choice(["Yes", "No"],      N, p=[0.65, 0.35])
dependents = np.random.choice(["0", "1", "2", "3+"], N, p=[0.57, 0.17, 0.16, 0.10])
education  = np.random.choice(["Graduate", "Not Graduate"], N, p=[0.78, 0.22])
self_emp   = np.random.choice(["Yes", "No"], N, p=[0.14, 0.86])
prop_area  = np.random.choice(["Urban", "Semiurban", "Rural"], N, p=[0.37, 0.38, 0.25])

# ── Numerical features ────────────────────────────────────────────────────────
# Income slightly higher for Graduates
app_income = np.where(
    education == "Graduate",
    np.random.normal(5500, 3000, N),
    np.random.normal(3800, 2000, N)
).astype(int)
app_income = np.clip(app_income, 1000, 81000)

# ~35 % of applicants have no co-applicant income
_has_co = np.random.random(N) >= 0.35          # True → has co-applicant
_co_raw = np.random.normal(2000, 1500, N)
co_income = np.where(_has_co, _co_raw, 0.0).astype(float)
co_income = np.clip(co_income, 0, 41667)

loan_amount = ((app_income + co_income) * np.random.uniform(0.08, 0.35, N) / 1000).round()
loan_amount = np.clip(loan_amount, 9, 700)

loan_term = np.random.choice([12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
                              N, p=[0.01, 0.01, 0.02, 0.02, 0.04, 0.08, 0.04, 0.04, 0.69, 0.05])

# Credit history heavily influences approval
credit_hist = np.random.choice([1.0, 0.0], N, p=[0.84, 0.16])

# ── Target: Loan_Status ───────────────────────────────────────────────────────
# Approval probability depends on credit history, income, and education
base_prob = (
    0.50 * credit_hist
    + 0.15 * (app_income > 4000).astype(float)
    + 0.10 * (education == "Graduate").astype(float)
    + 0.10 * (prop_area == "Semiurban").astype(float)
    + 0.05 * (married == "Yes").astype(float)
    - 0.05 * (self_emp == "Yes").astype(float)
    + np.random.normal(0, 0.05, N)
)
base_prob = np.clip(base_prob, 0.05, 0.95)
loan_status = np.where(np.random.random(N) < base_prob, "Y", "N")

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "Loan_ID":            loan_ids,
    "Gender":             add_missing(gender.copy()),
    "Married":            add_missing(married.copy()),
    "Dependents":         add_missing(dependents.copy()),
    "Education":          education,
    "Self_Employed":      add_missing(self_emp.copy()),
    "ApplicantIncome":    app_income,
    "CoapplicantIncome":  co_income.round(2),
    "LoanAmount":         add_missing(loan_amount.copy()),
    "Loan_Amount_Term":   add_missing(loan_term.astype(object).copy()),
    "Credit_History":     add_missing(credit_hist.copy()),
    "Property_Area":      prop_area,
    "Loan_Status":        loan_status,
})

df.to_csv("loan_data.csv", index=False)
print(f"✅  Dataset generated: loan_data.csv  ({len(df)} rows, {df.shape[1]} columns)")
print(f"    Approval rate : {(df['Loan_Status']=='Y').mean():.1%}")
print(f"    Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
