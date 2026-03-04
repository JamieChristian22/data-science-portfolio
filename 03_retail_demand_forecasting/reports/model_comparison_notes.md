
# Forecast Model Comparison Notes

This project includes a baseline and a RandomForest forecast.

## Baseline
A simple baseline for daily demand forecasting is the same day last week (lag_7).

## Why compare?
- Baselines establish whether ML is worth maintaining.
- If ML improvement is small, prioritize simpler strategies.

## Recommendation
If RandomForest reduces MAPE by ≥ 10–15% vs baseline in production backtests, adopt ML and retrain monthly.
