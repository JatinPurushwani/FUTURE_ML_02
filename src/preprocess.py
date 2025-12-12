# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from typing import Tuple, List

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create recommended engineered features in-place and return df copy.
    Assumes raw Telco dataset columns are present.
    """
    df = df.copy()

    # Convert TotalCharges to numeric (some rows may be blank)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # number of services (count of columns not equal to 'No' or 'No phone service')
    service_cols = [
        "PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
        "DeviceProtection","TechSupport","StreamingTV","StreamingMovies","InternetService"
    ]
    # Some datasets include InternetService with values; count only positive services
    def _is_service_val(val):
        return False if pd.isna(val) else (str(val).lower() not in ("no", "no phone service", "none"))

    if set(service_cols).intersection(df.columns):
        df["num_services"] = df[service_cols].apply(lambda row: sum(_is_service_val(v) for v in row), axis=1)
    else:
        df["num_services"] = 0

    # tenure bucket
    if "tenure" in df.columns:
        df["tenure_bucket"] = pd.cut(df["tenure"], bins=[-1, 6, 12, 24, 48, 1000], labels=["0-6","7-12","13-24","25-48","48+"])

    # contract months numeric
    if "Contract" in df.columns:
        df["contract_months"] = df["Contract"].map({"Month-to-month": 0, "One year": 12, "Two year": 24})
        df["contract_months"] = df["contract_months"].fillna(-1)

    # high monthly flag
    if "MonthlyCharges" in df.columns:
        median_mc = df["MonthlyCharges"].median()
        df["high_monthly"] = (df["MonthlyCharges"] > median_mc).astype(int)

    return df


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Create a ColumnTransformer for numeric and categorical features.
    Returns (preprocessor, numeric_cols, categorical_cols)
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transform = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # FIXED: use sparse_output=False for newer sklearn
    categorical_transform = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transform, numeric_cols),
        ("cat", categorical_transform, categorical_cols)
    ], remainder="drop")

    return preprocessor, numeric_cols, categorical_cols
