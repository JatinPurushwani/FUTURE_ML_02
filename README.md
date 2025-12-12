# Telecom Customer Churn Prediction — FUTURE_ML_02

A complete, production-oriented machine learning pipeline that predicts **customer churn** using the Telco dataset.  
The project covers data processing, feature engineering, model training, evaluation, and delivery of actionable outputs (probabilities + feature importance).

Built as a real-world ML engineering assignment with a focus on clean structure, reproducibility, and practical business use.

---

## 1. Problem Overview

Subscription businesses (telecom, SaaS, banking) lose revenue when customers churn.  
Most companies track customers in different internal systems but lack a reliable way to:

- Identify high-risk customers early  
- Understand *why* they may churn  
- Prioritize retention actions  
- Provide churn risk insights to BI/CRM systems  

Goal:

> Build a churn prediction system that outputs **per-customer churn probability** + **drivers**, supporting data-driven retention decisions.

---

## 2. High-Level Solution Design

### **Core ML Pipeline**
1. **Data Cleaning**
   - Fix inconsistent datatypes (e.g., `TotalCharges`)
   - Handle missing entries
   - Convert `Churn` into binary label (0/1)

2. **Feature Engineering**
   - Service counts (`num_services`)
   - Tenure segmentation (`tenure_bucket`)
   - Contract duration encoding (`contract_months`)
   - High-spend identification (`high_monthly`)

3. **Preprocessing**
   - Numeric → imputation + scaling  
   - Categorical → imputation + one-hot encoding  

4. **Models**
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost (final chosen model)

5. **Evaluation**
   - ROC-AUC, PR-AUC
   - Precision–Recall analysis
   - Threshold tuning for business context

6. **Deliverables**
   - `churn_predictions.csv` → probabilities for each customer
   - `feature_importance.csv` → top drivers of churn
   - Trained model artifact (`best_model_xgb.joblib`)
   - Power BI-ready outputs

---

## 3. Tech Stack

- **Language:** Python  
- **ML:** scikit-learn, XGBoost  
- **Data Handling:** pandas, numpy  
- **Explainability:** SHAP (optional)  
- **Visualization:** matplotlib / Power BI  
- **Environment:** Jupyter Notebooks + modular Python scripts  

---

## 4. Project Structure
```
FUTURE_ML_02/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   ├── 01_inspect.ipynb
│   ├── 02_preprocess.ipynb
│   ├── 03_train_and_eval.ipynb
│   └── 04_shap_explain.ipynb
│
├── output/
│   ├── churn_predictions.csv
│   ├── feature_importance.csv
│   └── best_model_xgb.joblib
│
├── src/
│   ├── data_utils.py
│   ├── preprocess.py
│   └── train.py
│
├── requirements.txt
└── README.md

```

## 5. Output Files

### **`churn_predictions.csv`**
Includes:
- `customerID`
- `actual_churn`
- `predicted_probability`  

Used in:
- Power BI dashboards  
- CRM retention pipelines (call lists)  
- Marketing segmentation

---

### **`feature_importance.csv`**
Global feature importance from XGBoost — used by BI/ops teams to understand churn drivers.

---

### **`best_model_xgb.joblib`**
Serialized ML model for inference, API deployment, or scheduled batch scoring.

---

## 6. Power BI Integration

**Visuals:**

- KPI → Avg predicted churn  
- Bar chart → Feature importance  
- Scatter → Monthly Charges vs Tenure (colored by churn probability)  
- Table → Top-risk customers sorted by probability  
- Filters → Contract type, Payment method, Tenure bucket  

This completes the ML → BI pipeline for stakeholder decision-making.

---

## 7. Reproducibility & Development Workflow

### **Setup**
```bash
git clone https://github.com/JatinPurushwani/FUTURE_ML_02
cd FUTURE_ML_02
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt

