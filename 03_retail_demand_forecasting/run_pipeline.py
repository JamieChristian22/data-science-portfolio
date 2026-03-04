import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT/"data"/"daily_sales.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

for lag in [1,7,14]:
    df[f"lag_{lag}"] = df["units_sold"].shift(lag)
df["roll_mean_7"] = df["units_sold"].shift(1).rolling(7).mean()
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df = df.dropna().reset_index(drop=True)

X = df.drop(columns=["units_sold","date"])
y = df["units_sold"].astype(float)

split = int(len(df)*0.8)
Xtr, Xte = X.iloc[:split], X.iloc[split:]
ytr, yte = y.iloc[:split], y.iloc[split:]

model = RandomForestRegressor(n_estimators=220, random_state=42, min_samples_leaf=2)
model.fit(Xtr, ytr)
pred = model.predict(Xte)

metrics = {
    "mae_units": float(mean_absolute_error(yte, pred)),
    "mape": float(mean_absolute_percentage_error(yte, pred))
}
(ROOT/"models").mkdir(exist_ok=True)
(ROOT/"reports").mkdir(exist_ok=True)
joblib.dump(model, ROOT/"models"/"demand_forecast_rf_retrained.joblib")
(Path(ROOT/"reports"/"metrics_recomputed.json")).write_text(json.dumps(metrics, indent=2))
print("Done.")
