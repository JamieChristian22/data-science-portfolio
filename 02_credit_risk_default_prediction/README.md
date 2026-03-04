# Credit Risk: Loan Default Prediction (End-to-End)

## Business Goal
Predict probability of default to reduce losses and improve approval/pricing decisions.

## Dataset
File: `data/credit_risk.csv`  
Rows: **5,000** | Default rate: **26.7%**

## Model
Gradient Boosting with:
- One-hot encoding (`loan_purpose`)
- Standard scaling (numeric)

## Results (Holdout Test)
- ROC AUC: **0.630**
- Accuracy: **0.729**
- Precision: **0.467**
- Recall: **0.105**
- F1: **0.171**

Artifacts:
- Model: `models/credit_risk_gb.joblib`
- Metrics: `reports/metrics.json`
- Charts: `figures/roc_curve.png`, `figures/precision_recall.png`

## Business Actions
1. Probability bands → approve / review / decline thresholds.
2. Monitor default rate by band monthly (drift).
3. Use top drivers for policy reviews (DTI, utilization, delinquencies).
