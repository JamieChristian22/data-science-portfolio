from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

from src.generate_data import generate_dataset
from src.modeling import FEATURES, TARGET, candidate_models, classification_metrics, choose_threshold, cv_summary

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "telecom_churn.csv"
FIG = ROOT / "figures"
REP = ROOT / "reports"
MOD = ROOT / "models"
for p in (FIG, REP, MOD): p.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    generate_dataset().to_csv(DATA, index=False)

df = pd.read_csv(DATA)
X, y = df[FEATURES], df[TARGET]
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=.20, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=.25, stratify=y_trainval, random_state=42)

comparison = []
models = candidate_models()
for name, model in models.items():
    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    val = classification_metrics(y_val, val_prob, .5)
    cv = cv_summary(model, X_trainval, y_trainval)
    comparison.append({"model": name, **val, **cv})

comparison_df = pd.DataFrame(comparison).sort_values(["pr_auc", "roc_auc"], ascending=False)
comparison_df.to_csv(REP / "model_comparison.csv", index=False)

# Favor interpretable logistic regression when discrimination is competitive.
logit = models["Logistic Regression"]
logit.fit(X_train, y_train)
val_prob = logit.predict_proba(X_val)[:, 1]
threshold, threshold_df = choose_threshold(y_val, val_prob, beta=2, min_precision=.22)
threshold_df.to_csv(REP / "threshold_analysis.csv", index=False)

# Refit selected model using all non-test data after threshold selection.
logit.fit(X_trainval, y_trainval)
test_prob = logit.predict_proba(X_test)[:, 1]
metrics_default = classification_metrics(y_test, test_prob, .50)
metrics_business = classification_metrics(y_test, test_prob, threshold)

artifact = {"model": logit, "threshold": threshold, "features": FEATURES,
            "metadata": {"target": TARGET, "model_name": "Logistic Regression", "seed": 42}}
joblib.dump(artifact, MOD / "churn_model.joblib")

metrics = {
    "dataset": {"rows": int(len(df)), "churn_rate": float(y.mean()), "test_rows": int(len(y_test))},
    "selected_model": "Logistic Regression",
    "default_threshold_metrics": metrics_default,
    "business_threshold_metrics": metrics_business,
    "business_threshold": threshold,
}
(REP / "metrics.json").write_text(json.dumps(metrics, indent=2))

# ROC and PR curves
RocCurveDisplay.from_predictions(y_test, test_prob)
plt.title("ROC Curve — Holdout Test Set"); plt.tight_layout(); plt.savefig(FIG / "roc_curve.png", dpi=180); plt.close()
PrecisionRecallDisplay.from_predictions(y_test, test_prob)
plt.title("Precision–Recall Curve — Holdout Test Set"); plt.tight_layout(); plt.savefig(FIG / "precision_recall_curve.png", dpi=180); plt.close()
ConfusionMatrixDisplay.from_predictions(y_test, (test_prob >= threshold).astype(int), values_format='d')
plt.title(f"Confusion Matrix — Threshold {threshold:.2f}"); plt.tight_layout(); plt.savefig(FIG / "confusion_matrix.png", dpi=180); plt.close()

# Threshold tradeoff chart
plt.figure(figsize=(8,5))
plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
plt.plot(threshold_df["threshold"], threshold_df["f2"], label="F2")
plt.axvline(threshold, linestyle="--", label=f"Selected {threshold:.2f}")
plt.xlabel("Decision threshold"); plt.ylabel("Score"); plt.title("Threshold Tradeoffs")
plt.legend(); plt.tight_layout(); plt.savefig(FIG / "threshold_tradeoffs.png", dpi=180); plt.close()

# Logistic coefficients / odds-ratio style interpretation
prep = logit.named_steps["prep"]
feature_names = prep.get_feature_names_out()
coefs = logit.named_steps["model"].coef_[0]
coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs, "odds_ratio": np.exp(coefs)})
coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
coef_df.sort_values("abs_coefficient", ascending=False).to_csv(REP / "logistic_coefficients.csv", index=False)

# Model-agnostic permutation importance on holdout set
perm = permutation_importance(logit, X_test, y_test, scoring="average_precision", n_repeats=15, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({"feature": FEATURES, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
perm_df.to_csv(REP / "permutation_importance.csv", index=False)
plt.figure(figsize=(8,5))
top = perm_df.head(9).sort_values("importance_mean")
plt.barh(top["feature"], top["importance_mean"])
plt.xlabel("Decrease in PR-AUC when permuted"); plt.title("Permutation Feature Importance")
plt.tight_layout(); plt.savefig(FIG / "feature_importance.png", dpi=180); plt.close()

# Business simulation: compare threshold to no-model and default threshold.
def business_case(m, contact_cost=12, retained_value=450, save_rate=.30):
    contacted = m["tp"] + m["fp"]
    expected_saves = m["tp"] * save_rate
    value_saved = expected_saves * retained_value
    campaign_cost = contacted * contact_cost
    net = value_saved - campaign_cost
    return {"contacted_customers": int(contacted), "true_churners_reached": int(m["tp"]),
            "expected_saves": float(expected_saves), "gross_value_saved": float(value_saved),
            "campaign_cost": float(campaign_cost), "estimated_net_value": float(net)}

business = {
    "assumptions": {"contact_cost_usd": 12, "retained_customer_value_usd": 450, "save_rate_if_true_churner_contacted": .30},
    "default_threshold_0_50": business_case(metrics_default),
    "selected_business_threshold": business_case(metrics_business),
}
(REP / "business_case.json").write_text(json.dumps(business, indent=2))

# Score the full customer base for an action-oriented deliverable.
full_prob = logit.predict_proba(X)[:,1]
scored = df[["customer_id"]].copy()
scored["churn_probability"] = full_prob
scored["risk_band"] = pd.cut(full_prob, [-.001,.20,.40,.60,.80,1.0], labels=["Very Low","Low","Medium","High","Very High"])
scored["retention_priority"] = np.where(full_prob >= threshold, "Contact", "Monitor")
scored.sort_values("churn_probability", ascending=False).to_csv(REP / "customer_risk_scores.csv", index=False)

print(json.dumps(metrics, indent=2))
print("Artifacts written to models/, figures/, and reports/.")
