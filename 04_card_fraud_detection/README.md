# Credit Card Fraud Detection (Supervised + Unsupervised)

## Business Goal
Detect fraudulent transactions for real-time review and loss prevention.

## Dataset
File: `data/transactions.csv`  
Rows: **30,000** | Fraud rate: **0.46%**

## Models
- **Supervised**: Logistic Regression (class-weight balanced) for real-time scoring
- **Unsupervised**: Isolation Forest for anomaly scoring when labels are incomplete

## Results (Holdout)
Supervised (thr=0.35):
- ROC AUC: **0.695**
- Precision: **0.005**
- Recall: **0.929**
- F1: **0.010**

Unsupervised:
- ROC AUC (anomaly score): **0.632**

Artifacts:
- Models: `models/fraud_logreg.joblib`, `models/fraud_isolation_forest.joblib`
- Metrics: `reports/metrics_supervised.json`, `reports/metrics_unsupervised.json`
- Charts: `figures/precision_recall.png`, `figures/roc_curve.png`, `figures/confusion_matrix.png`
