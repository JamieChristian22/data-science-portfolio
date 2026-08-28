
import pandas as pd
from src.credit_risk_pipeline import load_data, choose_threshold

def test_dataset_schema():
    df = load_data()
    assert len(df) == 5000
    assert set(df["defaulted"].unique()).issubset({0,1})
    assert df["application_id"].is_unique

def test_probability_inputs():
    df = load_data()
    assert df["debt_to_income"].between(0,1).all()
    assert df["credit_utilization"].between(0,1).all()
    assert df["credit_score"].between(300,850).all()

def test_threshold_helper():
    t = choose_threshold([0,0,1,1], [0.1,0.2,0.7,0.8])
    assert 0.10 <= t <= 0.70
