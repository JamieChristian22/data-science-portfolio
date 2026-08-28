
from pathlib import Path
import pandas as pd, numpy as np, joblib, json
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parents[1]
def make_features(d):
    x=d.copy(); x["date"]=pd.to_datetime(x["date"]); x=x.sort_values("date")
    for l in [1,7,14,28]: x[f"lag_{l}"]=x.sales_units.shift(l)
    x["rolling_7"]=x.sales_units.shift(1).rolling(7).mean(); x["rolling_28"]=x.sales_units.shift(1).rolling(28).mean()
    x["dow"]=x.date.dt.dayofweek; x["month"]=x.date.dt.month; x["weekofyear"]=x.date.dt.isocalendar().week.astype(int)
    return x.dropna().reset_index(drop=True)
def main():
    df=pd.read_csv(ROOT/"data/daily_sales.csv"); x=make_features(df); tr,te=x.iloc[:-90],x.iloc[-90:]
    f=[c for c in x.columns if c not in ["date","sales_units"]]
    m=HistGradientBoostingRegressor(max_iter=350,learning_rate=.05,max_leaf_nodes=24,l2_regularization=1,random_state=42)
    m.fit(tr[f],tr.sales_units); joblib.dump(m,ROOT/"models/demand_forecast_model_retrained.joblib"); print("ok")
if __name__=="__main__": main()
