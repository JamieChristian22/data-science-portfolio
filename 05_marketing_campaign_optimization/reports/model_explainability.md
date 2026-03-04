# Model Explainability (Practical)

This project includes **interpretable artifacts** that connect model behavior to business decisions.

What to review:
- **Metrics**: see `metrics.json` (and any notebook metrics in `baseline_metrics_notebook.json`).
- **Drivers**:
  - For classification notebooks, see `baseline_coefficients.csv` for top positive/negative drivers.
  - For forecasting, see `top_features.csv` / feature importance charts.
  - For recommender, see `recommender_eval_precision_at_k.json`.

How to use this:
1. Identify top drivers (e.g., churn drivers like month-to-month contracts + frequent support calls).
2. Translate drivers into levers (retention offers, policy updates, inventory rules, fraud thresholds).
3. Monitor drift: re-check driver magnitudes and outcome rates monthly.
