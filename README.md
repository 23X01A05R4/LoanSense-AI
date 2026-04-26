# 🏦 Loan Approval Prediction System

> An end-to-end Machine Learning project that predicts bank loan approval using Python, Flask, and Scikit-learn — complete with a professional dark-mode web UI, analytics dashboard, and confidence scoring.

---

## 📌 Project Overview

This system takes an applicant's financial and demographic profile as input and predicts whether their loan will be **Approved** or **Not Approved**, along with a **confidence percentage**. It is built as a complete portfolio-quality ML project suitable for:

- College / university submissions
- Resume & portfolio showcase
- Beginner-to-intermediate ML learning

---

## 🗂️ Project Structure

```
MLProject/
│
├── app.py                  # Flask web application (routes, prediction logic)
├── train_model.py          # ML pipeline: preprocessing → training → evaluation → saving
├── generate_dataset.py     # Synthetic dataset generator (mirrors Kaggle Loan dataset)
│
├── model.pkl               # Trained model bundle (auto-generated)
├── loan_data.csv           # Dataset (auto-generated)
│
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── templates/
│   ├── index.html          # Home — prediction form
│   ├── result.html         # Prediction result page
│   ├── charts.html         # Analytics dashboard
│   └── about.html          # Project info page
│
└── static/
    ├── style.css           # Dark-gradient glassmorphism stylesheet
    ├── script.js           # Client-side interactivity & validation
    └── charts/             # Auto-generated visualisation PNGs
        ├── loan_status_dist.png
        ├── model_comparison.png
        ├── confusion_matrix.png
        ├── feature_importance.png
        └── income_dist.png
```

---

## ⚙️ Technologies Used

| Layer         | Technology                          |
|---------------|-------------------------------------|
| Language      | Python 3.8+                         |
| Web Framework | Flask 2.x                           |
| ML Library    | Scikit-learn (LR + Random Forest)   |
| Data          | Pandas, NumPy                       |
| Visualisation | Matplotlib, Seaborn                 |
| Model Saving  | Pickle                              |
| Frontend      | HTML5, CSS3 (Vanilla), JavaScript   |
| Fonts         | Google Fonts — Inter                |

---

## 🚀 Quick Start

### 1. Clone / Download the project

```bash
cd MLProject
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the dataset

```bash
python generate_dataset.py
```

This creates `loan_data.csv` with 614 rows mirroring the Kaggle Loan Prediction dataset.

### 5. Train the model

```bash
python train_model.py
```

This will:
- Preprocess the data (handle nulls, encode categoricals, scale numerics)
- Train **Logistic Regression** and **Random Forest**
- Print evaluation metrics (accuracy, confusion matrix, classification report)
- Save the best model as `model.pkl`
- Generate 5 analytics charts in `static/charts/`

### 6. Run the web app

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## 🤖 Machine Learning Details

### Dataset Features

| Feature             | Type        | Description                          |
|---------------------|-------------|--------------------------------------|
| Gender              | Categorical | Male / Female                        |
| Married             | Categorical | Yes / No                             |
| Dependents          | Categorical | 0 / 1 / 2 / 3+                       |
| Education           | Categorical | Graduate / Not Graduate              |
| Self_Employed       | Categorical | Yes / No                             |
| ApplicantIncome     | Numerical   | Monthly income (₹)                   |
| CoapplicantIncome   | Numerical   | Co-applicant income (₹)              |
| LoanAmount          | Numerical   | Loan amount in ₹ thousands           |
| Loan_Amount_Term    | Numerical   | Repayment period in months           |
| Credit_History      | Binary      | 1 = good history, 0 = poor / none    |
| Property_Area       | Categorical | Urban / Semiurban / Rural            |
| **Loan_Status**     | **Target**  | **Y = Approved, N = Not Approved**   |

### Models Trained

1. **Logistic Regression** — interpretable baseline, uses StandardScaler
2. **Random Forest (200 trees)** — ensemble model, higher accuracy

The best-performing model is automatically selected and saved.

### Preprocessing Steps

1. Drop `Loan_ID` (identifier, not a feature)
2. Fill missing values: categorical → mode, numerical → median
3. Label encode all categorical features
4. Standard scale numerical features (for Logistic Regression)
5. 80/20 stratified train-test split

---

## 🌐 Web Application Routes

| Route      | Method | Description                     |
|------------|--------|---------------------------------|
| `/`        | GET    | Home — prediction form          |
| `/predict` | POST   | Submit form → get prediction    |
| `/charts`  | GET    | Analytics / visualisation page  |
| `/about`   | GET    | Project info & documentation    |

---

## 📊 Features of the Web App

- ✅ Professional dark-mode UI with glassmorphism design
- ✅ Prediction confidence percentage with animated progress bar
- ✅ Form validation (client-side JS + server-side Python)
- ✅ Submitted details summary on result page
- ✅ Actionable tips based on prediction outcome
- ✅ Analytics dashboard with 5 auto-generated charts
- ✅ About page with ML pipeline documentation
- ✅ Responsive design (mobile-friendly)

---

## 📈 Sample Output

```
Model Accuracy Comparison
─────────────────────────
Logistic Regression : ~79%
Random Forest       : ~82%

✅ Best model selected: Random Forest (accuracy=0.8211)
```

---

## 📝 License

This project is open-source and free to use for educational purposes.

---

## 👨‍💻 Author

Built as a complete ML portfolio project using Python, Flask, and Scikit-learn.
