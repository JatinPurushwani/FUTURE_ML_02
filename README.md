# 📘 Telco Customer Churn Prediction — FUTURE_ML_02
A complete, practical machine learning pipeline for predicting customer churn using the Telco Customer Churn dataset.

---

## 🔍 Overview

This project builds an end-to-end churn prediction system.  
It includes:

- Data inspection and cleaning  
- Feature engineering  
- Preprocessing pipelines  
- Model training (Logistic Regression, Random Forest, XGBoost)  
- Model comparison  
- Threshold tuning for business decisions  
- Export of churn probabilities and feature importance  
- Guidance for Power BI dashboards and SHAP explainability  

The output is directly usable by BI/CRM teams for retention workflows.

---

## 📂 Repository Structure

FUTURE_ML_02/
│
├── data/ # Raw dataset (not added to Git)
│ └── Telco-Customer-Churn.csv
│
├── notebooks/
│ ├── 01_inspect.ipynb # Data understanding
│ ├── 02_preprocess.ipynb # Cleaning + feature engineering
│ ├── 03_train_and_eval.ipynb # All ML models + evaluation
│ └── 04_shap_explain.ipynb # Explainability (optional)
│
├── output/
│ ├── churn_predictions.csv
│ ├── feature_importance.csv
│ └── best_model_xgb.joblib
│
├── src/
│ ├── data_utils.py
│ ├── preprocess.py
│ └── train.py
│
├── README.md
└── requirements.txt


---

## 🧠 Problem Definition

**Goal:** Predict whether a customer will churn (target = `Churn`).  
**Type:** Binary classification.  
**Business Impact:** Prevent churn by ranking customers by churn risk and enabling targeted retention actions.

---

## 🛠️ Methodology

### 1. Data Cleaning
- Converted `TotalCharges` from string to numeric  
- Handled missing values  
- Removed `customerID` from feature set  
- Mapped `Churn` from Yes/No → 1/0  

### 2. Feature Engineering
Created features to enhance predictive power:

- `num_services` – count of services subscribed  
- `tenure_bucket` – grouped tenure ranges  
- `contract_months` – numeric contract duration  
- `high_monthly` – charges above median  

### 3. Preprocessing
Handled by `ColumnTransformer`:

- Numerical: median imputation + scaling  
- Categorical: imputation + one-hot encoding  

### 4. Models Used
| Model | Purpose |
|-------|---------|
| Logistic Regression | Interpretable baseline |
| Random Forest | Handles non-linear interactions |
| XGBoost | High-performance final model |

### 5. Evaluation Metrics
- ROC-AUC  
- Precision/Recall  
- Average Precision (PR-AUC)  
- Confusion matrix  
- Business-driven threshold optimization  

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone <repo_url>
cd FUTURE_ML_02
