# src/eval_and_report.py
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, classification_report
)
from fpdf import FPDF

OUT = Path("output")
OUT.mkdir(parents=True, exist_ok=True)

# --- load predictions and feature importance
preds = pd.read_csv(OUT / "churn_predictions.csv")
# use model_metrics_summary.csv if present
metrics_df = pd.read_csv(OUT / "model_metrics_summary.csv") if (OUT / "model_metrics_summary.csv").exists() else None
fi = pd.read_csv(OUT / "feature_importance.csv") if (OUT / "feature_importance.csv").exists() else None

# Ensure columns
assert {"predicted_probability", "actual_churn"}.issubset(preds.columns), "churn_predictions.csv missing columns"

y_true = preds["actual_churn"].values
y_proba = preds["predicted_probability"].values
y_pred = (y_proba >= 0.5).astype(int)

# ---------- Metrics & text summary ----------
auc = roc_auc_score(y_true, y_proba)
ap = average_precision_score(y_true, y_proba)
cr = classification_report(y_true, y_pred)
metrics_text = f"ROC-AUC: {auc:.4f}\nAverage Precision (PR-AUC): {ap:.4f}\n\nClassification Report:\n{cr}\n"
(OUT / "metrics.txt").write_text(metrics_text)

# ---------- Confusion matrix plot ----------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(5,5))
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
ax.set_title("Confusion Matrix (threshold=0.5)")
fig.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=200)
plt.close(fig)

# ---------- ROC curve ----------
fpr, tpr, _ = roc_curve(y_true, y_proba)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
ax.plot([0,1],[0,1],'--', color='gray')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "roc_curve.png", dpi=200)
plt.close(fig)

# ---------- Precision-Recall curve ----------
prec, rec, _ = precision_recall_curve(y_true, y_proba)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(rec, prec, label=f"AP={ap:.3f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "pr_curve.png", dpi=200)
plt.close(fig)

# ---------- Feature importance plot (if available) ----------
if fi is not None and not fi.empty:
    topn = min(15, len(fi))
    fi_top = fi.sort_values("importance", ascending=False).head(topn)
    fig, ax = plt.subplots(figsize=(6, max(3, topn*0.3)))
    ax.barh(fi_top["feature"][::-1], fi_top["importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances (XGBoost)")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=200)
    plt.close(fig)

# ---------- Small textual summary for PDF -----------
# pick top drivers (from fi) and some numbers
top_drivers = fi["feature"].head(5).tolist() if (fi is not None and not fi.empty) else []
summary_lines = [
    "Churn Model Summary",
    f"Rows scored: {len(preds)}",
    f"ROC-AUC: {auc:.4f}",
    f"PR-AUC (AP): {ap:.4f}",
    "Top feature drivers: " + ", ".join(top_drivers) if top_drivers else "Top feature drivers: N/A",
    "",
    "Business recommendation highlights:",
    " - Target month-to-month users with early-tenure offers",
    " - Convert electronic-check users to auto-pay",
    " - Offer support/security bundles to at-risk customers"
]

# ---------- Build PDF --------------
pdf = FPDF(unit="pt", format="A4")
pdf.set_auto_page_break(margin=36)
pdf.add_page()
pdf.set_font("Helvetica", size=14)
pdf.cell(0, 18, "Churn Model Report", ln=1)
pdf.set_font("Helvetica", size=10)
for line in summary_lines:
    pdf.multi_cell(0, 14, line)
pdf.ln(6)

# Insert images if exist
def insert_image_if_exists(path, w=450):
    if (OUT / path).exists():
        pdf.image(str(OUT / path), w= w, h= w*0.6)
        pdf.ln(8)

insert_image_if_exists("confusion_matrix.png", w=250)
insert_image_if_exists("roc_curve.png", w=350)
insert_image_if_exists("pr_curve.png", w=350)
insert_image_if_exists("feature_importance.png", w=400)

pdf.output(OUT / "Churn_Report.pdf")
print("Saved outputs to", OUT)
