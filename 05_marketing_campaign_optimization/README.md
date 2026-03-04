# Marketing Campaign Optimization (Response Prediction + Lift)

## Business Goal
Predict which customers are most likely to respond so outreach focuses where conversion is highest.

## Dataset
File: `data/campaign_contacts.csv`  
Rows: **9,000** | Response rate: **3.3%**

## Model
Logistic Regression with one-hot encoding + scaling.

## Results (Holdout)
- ROC AUC: **0.629**
- Precision: **0.000**
- Recall: **0.000**
- F1: **0.000**

Lift:
- `reports/lift_by_decile.csv`
- `figures/lift_curve.png`

Artifacts:
- Model: `models/campaign_response_logreg.joblib`
- Metrics: `reports/metrics.json`
- Charts: `figures/roc_curve.png`, `figures/precision_recall.png`, `figures/lift_curve.png`
