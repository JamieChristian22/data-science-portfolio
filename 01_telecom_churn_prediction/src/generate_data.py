from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "telecom_churn.csv"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_dataset(n=3500, seed=42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.56, 0.27, 0.17])
    internet = rng.choice(["DSL", "Fiber optic", "None"], n, p=[0.36, 0.49, 0.15])
    paperless = rng.choice(["Yes", "No"], n, p=[0.61, 0.39])
    base_charge = np.select([internet == "None", internet == "DSL", internet == "Fiber optic"], [35, 65, 90])
    monthly = np.clip(base_charge + rng.normal(0, 19, n) + (paperless == "Yes") * 2.0, 20, 140).round(2)
    support = np.clip(rng.poisson(1.25 + (contract == "Month-to-month") * .25, n), 0, 8)
    late = np.clip(rng.poisson(0.55 + (paperless == "No") * .15, n), 0, 5)
    satisfaction = np.clip(rng.normal(3.65, 0.78, n) - support * .12 - late * .08, 1, 5).round(2)
    total = np.maximum(0, monthly * np.maximum(tenure, 1) * rng.normal(1.0, .055, n)).round(2)

    logit = (-2.65
             + 1.05 * (contract == "Month-to-month")
             + 0.42 * (internet == "Fiber optic")
             + 0.22 * (paperless == "Yes")
             + 0.20 * support
             + 0.28 * late
             + 0.010 * (monthly - 70)
             - 0.015 * tenure
             - 0.58 * (satisfaction - 3.0))
    p = sigmoid(logit)
    churn = rng.binomial(1, p)

    df = pd.DataFrame({
        "customer_id": [f"C{100000+i}" for i in range(n)],
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "contract_type": contract,
        "internet_service": internet,
        "paperless_billing": paperless,
        "support_calls_last_90d": support,
        "late_payments_last_12m": late,
        "satisfaction_score_1to5": satisfaction,
        "churned": churn,
    })
    return df

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows to {OUT}")
    print(f"Churn rate: {df['churned'].mean():.1%}")

if __name__ == "__main__":
    main()
