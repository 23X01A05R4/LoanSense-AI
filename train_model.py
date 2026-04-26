"""
train_model.py
--------------
End-to-end ML pipeline for the Loan Approval Prediction System.

Steps
-----
1. Load & explore the dataset
2. Preprocess (handle missing values, encode categoricals, scale numerics)
3. Train Logistic Regression and Random Forest
4. Evaluate both models (accuracy, confusion matrix, classification report)
5. Select the best model and save it as model.pkl
6. Save charts for data visualisation
"""

import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── 0. Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "loan_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
CHART_DIR  = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  LOAN APPROVAL PREDICTION — MODEL TRAINING")
print("=" * 65)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Run  python generate_dataset.py  first."
    )

df = pd.read_csv(DATA_PATH)
print(f"\n📂  Dataset loaded  →  {df.shape[0]} rows × {df.shape[1]} columns")
print("\n--- First 5 rows ---")
print(df.head())
print("\n--- Info ---")
print(df.dtypes)
print("\n--- Missing values ---")
print(df.isnull().sum())


# ════════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════════
print("\n\n[STEP 2] Preprocessing …")

# Drop Loan_ID (identifier, not a feature)
df.drop(columns=["Loan_ID"], inplace=True, errors="ignore")

# --- 2a. Fill missing values ---
# pandas 3.x uses Copy-on-Write; always use assignment form (df[col] = ...)

# Categorical → mode
cat_cols = ["Gender", "Married", "Dependents", "Self_Employed"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)              # convert NaN float → str 'nan'
        df[col] = df[col].replace('nan', np.nan)   # make them real NaN again
        df[col] = df[col].fillna(df[col].mode()[0])  # fill with mode

# Credit_History is numeric (float) with NaN → coerce then fill
if "Credit_History" in df.columns:
    df["Credit_History"] = pd.to_numeric(df["Credit_History"], errors="coerce")
    df["Credit_History"] = df["Credit_History"].fillna(
        df["Credit_History"].mode()[0]
    )

# Numerical → median
num_cols = ["LoanAmount", "Loan_Amount_Term"]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

print("   Missing values after fill:", df.isnull().sum().sum())
assert df.isnull().sum().sum() == 0, "ERROR: NaN values remain after imputation!"

# --- 2b. Encode categoricals ---
# Target: Y → 1, N → 0
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Remaining categorical columns
encode_cols = ["Gender", "Married", "Education", "Self_Employed",
               "Dependents", "Property_Area"]
le = LabelEncoder()
for col in encode_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

print("   Encoding complete.")

# --- 2c. Feature / target split ---
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

FEATURE_NAMES = list(X.columns)   # save for the Flask app

# --- 2d. Train / test split (80-20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape}  |  Test: {X_test.shape}")

# --- 2e. Scale numerical features ---
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ════════════════════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════════
print("\n\n[STEP 3] Training models …")

# --- Model A: Logistic Regression ---
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_sc, y_train)

# --- Model B: Random Forest ---
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)   # RF doesn't need scaling


# ════════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ════════════════════════════════════════════════════════════════════════════════
print("\n\n[STEP 4] Evaluating models …")


def evaluate(name, model, X_t, y_t, needs_scale=False):
    """Print metrics and return accuracy."""
    Xt = X_test_sc if needs_scale else X_t
    y_pred = model.predict(Xt)
    acc    = accuracy_score(y_t, y_pred)
    cv     = cross_val_score(
        model,
        X_train_sc if needs_scale else X_train,
        y_train, cv=5, scoring="accuracy"
    ).mean()

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Test Accuracy   : {acc:.4f}  ({acc*100:.2f} %)")
    print(f"  5-Fold CV Acc   : {cv:.4f}  ({cv*100:.2f} %)")
    print("\n  Classification Report:")
    print(classification_report(y_t, y_pred,
                                target_names=["Not Approved", "Approved"]))
    cm = confusion_matrix(y_t, y_pred)
    print("  Confusion Matrix:")
    print(cm)
    return acc, y_pred


lr_acc, lr_pred = evaluate("Logistic Regression", lr_model,
                            X_test_sc, y_test, needs_scale=True)
rf_acc, rf_pred = evaluate("Random Forest",       rf_model,
                            X_test,    y_test, needs_scale=False)


# ════════════════════════════════════════════════════════════════════════════════
# 5. SELECT BEST MODEL
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n\n[STEP 5] Model comparison")
print(f"  Logistic Regression accuracy : {lr_acc:.4f}")
print(f"  Random Forest accuracy       : {rf_acc:.4f}")

if rf_acc >= lr_acc:
    best_name   = "Random Forest"
    best_model  = rf_model
    best_pred   = rf_pred
    best_scaler = None           # RF doesn't need the scaler
    best_acc    = rf_acc
else:
    best_name   = "Logistic Regression"
    best_model  = lr_model
    best_pred   = lr_pred
    best_scaler = scaler
    best_acc    = lr_acc

print(f"\n  ✅  Best model selected: {best_name}  (accuracy={best_acc:.4f})")


# ════════════════════════════════════════════════════════════════════════════════
# 6. SAVE MODEL
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n\n[STEP 6] Saving model to {MODEL_PATH} …")

model_bundle = {
    "model":         best_model,
    "scaler":        best_scaler,   # None for RF
    "feature_names": FEATURE_NAMES,
    "model_name":    best_name,
    "accuracy":      round(best_acc, 4),
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_bundle, f)

print("   model.pkl saved  ✅")


# ════════════════════════════════════════════════════════════════════════════════
# 7. VISUALISATION CHARTS  (saved to static/charts/)
# ════════════════════════════════════════════════════════════════════════════════
print("\n\n[STEP 7] Generating charts …")

sns.set_theme(style="darkgrid", palette="muted")

# --- Chart 1: Loan Status Distribution ---
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["Loan_Status"].value_counts()
ax.bar(["Approved (Y)", "Not Approved (N)"],
       [counts.get(1, 0), counts.get(0, 0)],
       color=["#4CAF50", "#F44336"], edgecolor="white", linewidth=1.5)
ax.set_title("Loan Status Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Count")
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3, int(bar.get_height()),
            ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "loan_status_dist.png"), dpi=120)
plt.close()

# --- Chart 2: Model Accuracy Comparison ---
fig, ax = plt.subplots(figsize=(6, 4))
models  = ["Logistic\nRegression", "Random\nForest"]
accs    = [lr_acc * 100, rf_acc * 100]
colors  = ["#2196F3", "#FF9800"]
bars    = ax.bar(models, accs, color=colors, edgecolor="white", linewidth=1.5, width=0.4)
ax.set_ylim(0, 110)
ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
ax.set_ylabel("Accuracy (%)")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1, f"{acc:.2f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "model_comparison.png"), dpi=120)
plt.close()

# --- Chart 3: Confusion Matrix of best model ---
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Not Approved", "Approved"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "confusion_matrix.png"), dpi=120)
plt.close()

# --- Chart 4: Feature Importances (Random Forest) ---
if hasattr(rf_model, "feature_importances_"):
    importances = pd.Series(rf_model.feature_importances_,
                            index=FEATURE_NAMES).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    importances.plot(kind="barh", ax=ax, color="#673AB7", edgecolor="white")
    ax.set_title("Feature Importances (Random Forest)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "feature_importance.png"), dpi=120)
    plt.close()

# --- Chart 5: Applicant Income Distribution ---
fig, ax = plt.subplots(figsize=(7, 4))
orig_df = pd.read_csv(DATA_PATH)
ax.hist(orig_df["ApplicantIncome"].dropna(), bins=40, color="#009688",
        edgecolor="white", linewidth=0.8)
ax.set_title("Applicant Income Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Income")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "income_dist.png"), dpi=120)
plt.close()

print("   Charts saved to static/charts/  ✅")
print("\n" + "=" * 65)
print("  TRAINING COMPLETE")
print(f"  Best model  : {best_name}")
print(f"  Accuracy    : {best_acc*100:.2f} %")
print(f"  Saved to    : {MODEL_PATH}")
print("=" * 65 + "\n")
