import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import joblib

from data_utils import load_data, save_csv, save_model
from preprocess import engineer_features, build_preprocessor

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train")

def evaluate_model(pipe, X_test, y_test):
    y_proba = pipe.predict_proba(X_test)[:,1]
    y_pred = pipe.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    cr = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, (y_proba >= 0.5).astype(int))
    return {"auc": auc, "ap": ap, "report": cr, "confusion_matrix": cm, "y_proba": y_proba, "y_pred": y_pred}

def main(args):
    # paths
    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load
    df = load_data(data_path)

    # check for Churn column
    if "Churn" not in df.columns:
        raise RuntimeError("The loaded dataset does not contain 'Churn' column. Use original Telco dataset with label present.")

    # save customer ids
    customer_col = "customerID" if "customerID" in df.columns else None
    customer_ids = df[customer_col] if customer_col else pd.Series(range(len(df)), name="customer_index")

    # preprocess & feature engineering
    df = engineer_features(df)

    # prepare X, y
    y = df["Churn"].map({"Yes":1, "No":0}) if df["Churn"].dtype == object or df["Churn"].dtype == "O" else df["Churn"]
    X = df.drop(columns=[c for c in ["Churn", customer_col] if c in df.columns])

    # train/test split
    X_train, X_test, y_train, y_test, cust_train, cust_test = train_test_split(
        X, y, customer_ids, test_size=0.2, stratify=y, random_state=42
    )

    # build preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    logger.info("Numeric cols: %s", num_cols)
    logger.info("Categorical cols: %s", cat_cols)

    # models
    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42),
        "xgb": XGBClassifier(n_estimators=300, use_label_encoder=False, eval_metric="logloss", random_state=42)
    }

    results = {}
    pipelines = {}

    for name, clf in models.items():
        pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
        logger.info("Training %s...", name)
        pipe.fit(X_train, y_train)
        res = evaluate_model(pipe, X_test, y_test)
        results[name] = res
        pipelines[name] = pipe
        logger.info("%s AUC: %.4f  AP: %.4f", name, res["auc"], res["ap"])

    # pick best by AUC
    best_name = max(results.keys(), key=lambda k: results[k]["auc"])
    best_pipe = pipelines[best_name]
    logger.info("Selected best model: %s (AUC=%.4f)", best_name, results[best_name]["auc"])

    # final outputs: predictions on full dataset (using best pipeline)
    logger.info("Generating predictions for full dataset")
    probs_full = best_pipe.predict_proba(X)[:,1]
    preds_df = pd.DataFrame({
        "customerID": customer_ids.reset_index(drop=True),
        "actual_churn": y.reset_index(drop=True),
        "predicted_probability": probs_full
    })

    # feature importance if available
    fi_df = pd.DataFrame()
    try:
        clf_final = best_pipe.named_steps["clf"]
        if hasattr(clf_final, "feature_importances_"):
            # need feature names; get from preprocessor
            try:
                feature_names = []
                # numeric names
                if hasattr(preprocessor, "transformers_"):
                    for name_tr, trans, cols in preprocessor.transformers_:
                        if name_tr == "num":
                            feature_names.extend(cols)
                        elif name_tr == "cat":
                            # OneHotEncoder inside pipeline -> get categories
                            ohe = trans.named_steps["onehot"]
                            cats = []
                            if hasattr(ohe, "get_feature_names_out"):
                                # scikit-learn >=1.0
                                cats = list(ohe.get_feature_names_out(cols))
                            else:
                                # fallback
                                cats = [f"{c}_{i}" for c in cols for i in range(1)]
                            feature_names.extend(cats)
                importances = clf_final.feature_importances_
                fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
                fi_df = fi_df.sort_values("importance", ascending=False)
            except Exception as e:
                logger.warning("Could not extract feature names for importance: %s", e)
    except Exception:
        logger.exception("Feature importance extraction failed.")

    # Save outputs
    save_csv(preds_df, out_dir, "churn_predictions.csv")
    if not fi_df.empty:
        save_csv(fi_df, out_dir, "feature_importance.csv")

    # save model
    save_model(best_pipe, out_dir, name=f"best_model_{best_name}.joblib")

    # save metrics summary
    metrics_records = []
    for name, r in results.items():
        metrics_records.append({
            "model": name,
            "auc": r["auc"],
            "average_precision": r["ap"],
            "precision_macro": r["report"]["macro avg"]["precision"],
            "recall_macro": r["report"]["macro avg"]["recall"],
            "f1_macro": r["report"]["macro avg"]["f1-score"]
        })
    metrics_df = pd.DataFrame(metrics_records)
    save_csv(metrics_df, out_dir, "model_metrics_summary.csv")

    logger.info("All outputs saved to %s", out_dir.resolve())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn models and export predictions.")
    parser.add_argument("--data", required=True, help="Path to Telco-Customer-Churn.csv")
    parser.add_argument("--out", required=True, help="Output directory (will be created)")
    args = parser.parse_args()
    main(args)
