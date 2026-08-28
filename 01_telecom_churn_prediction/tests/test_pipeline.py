from pathlib import Path
import pandas as pd
from src.generate_data import generate_dataset
from src.modeling import FEATURES, candidate_models, classification_metrics


def test_generated_schema_and_size():
    df = generate_dataset(n=500, seed=7)
    assert len(df) == 500
    assert set(FEATURES + ["customer_id", "churned"]).issubset(df.columns)
    assert set(df["churned"].unique()).issubset({0, 1})


def test_logistic_pipeline_predicts_probabilities():
    df = generate_dataset(n=800, seed=8)
    X, y = df[FEATURES], df["churned"]
    model = candidate_models(seed=8)["Logistic Regression"]
    model.fit(X, y)
    p = model.predict_proba(X.iloc[:20])[:, 1]
    assert len(p) == 20
    assert ((p >= 0) & (p <= 1)).all()


def test_metrics_keys():
    m = classification_metrics([0,0,1,1], [0.1,0.4,0.6,0.9], .5)
    for key in ["accuracy","precision","recall","f1","f2","roc_auc","pr_auc","tn","fp","fn","tp"]:
        assert key in m
