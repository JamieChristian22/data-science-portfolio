import pandas as pd
from src.forecast_pipeline import make_features

def test_features():
 x=make_features(pd.read_csv('data/daily_sales.csv')); assert not x.isna().any().any()
