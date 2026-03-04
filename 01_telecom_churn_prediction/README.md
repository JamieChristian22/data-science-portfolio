# Telecom Customer Churn Prediction (End-to-End)

## Business Goal
Predict which customers are likely to churn so retention teams can intervene.

## Dataset
File: `data/telecom_churn.csv`  
Rows: **3,500** | Churn rate: **15.0%**

## Model
Logistic Regression with:
- One-hot encoding for contract/service/billing
- Standard scaling for numeric features

## Results (Holdout Test)
- ROC AUC: **0.771**
- Accuracy: **0.858**
- Precision: **0.636**
- Recall: **0.122**
- F1: **0.204**

Artifacts:
- Model: `models/churn_logreg.joblib`
- Metrics: `reports/metrics.json`
- Charts: `figures/roc_curve.png`, `figures/confusion_matrix.png`

## Business Actions
1. Weekly churn scoring → prioritize top risk decile.
2. Retention offers for month-to-month + high charges + repeated support calls.
3. CX improvements for low satisfaction customers.
