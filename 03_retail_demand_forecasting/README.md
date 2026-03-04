# Retail Demand Forecasting (Time Series + ML)

## Business Goal
Forecast daily demand to reduce stockouts and overstocking.

## Dataset
File: `data/daily_sales.csv`  
Date range: **2024-01-01 → 2025-10-11**

## Feature Engineering
- Lags: 1, 7, 14
- 7-day rolling mean
- Calendar: day-of-week, month
- Known drivers: promo, holiday season, price index

## Model
RandomForestRegressor with time-based split (no leakage).

## Results (Holdout)
- MAE: **18.56 units**
- MAPE: **7.63%**

Artifacts:
- Model: `models/demand_forecast_rf.joblib`
- Metrics: `reports/metrics.json`
- Charts: `figures/actual_vs_predicted.png`, `figures/feature_importance.png`
