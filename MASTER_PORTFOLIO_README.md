
# Data Science Portfolio (6 End-to-End Case Studies) 

This repository contains **6 production-style data science case studies**. Each project includes:
- CSV dataset (`data/`)
- **EDA notebook** (`notebooks/01_eda.ipynb`)
- **Modeling notebook** (`notebooks/02_modeling.ipynb`)
- Reproducible training pipeline (`run_pipeline.py`)
- Saved model artifacts (`models/`)
- Metrics + business-ready reports (`reports/`)
- Visualizations (`figures/`)
- **Working Streamlit demo app** (`deployment/app_streamlit.py`)

## Projects
01. Telecom Churn Prediction (classification)
02. Credit Risk Default Prediction (classification + policy bands)
03. Retail Demand Forecasting (time-series features + baseline comparison)
04. Credit Card Fraud Detection (supervised + anomaly scoring)
05. Marketing Campaign Optimization (lift + ROI simulation)
06. E-Commerce Recommendation System (collaborative filtering + Precision@K evaluation)

## Run any project locally
```bash
cd 01_telecom_churn_prediction
pip install -r requirements.txt
python run_pipeline.py
```

## Run the Streamlit demo
```bash
cd 01_telecom_churn_prediction/deployment
pip install -r ../requirements.txt
streamlit run app_streamlit.py
```

## What this portfolio proves
- You can turn messy business questions into measurable ML outcomes.
- You can evaluate models with the *right metrics* (ROC-AUC, PR curves, MAE/MAPE, Precision@K).
- You can package work like a professional: reproducible code, saved models, and demos.
