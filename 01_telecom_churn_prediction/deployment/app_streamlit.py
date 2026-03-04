
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="01_telecom_churn_prediction Demo", layout="centered")
st.title("01 Telecom Churn Prediction — Model Demo")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telecom_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_logreg.joblib"

st.caption("Upload a CSV to score rows using the trained model artifact stored in /models.")

with st.expander("Preview the dataset schema"):
    df_schema = pd.read_csv(DATA_PATH).head(10)
    st.dataframe(df_schema)

uploaded = st.file_uploader("Upload CSV (same columns as dataset, excluding target).", type=["csv"])

df_full = pd.read_csv(DATA_PATH)
target_col = "churned"

if uploaded is not None:
    df_in = pd.read_csv(uploaded)
else:
    df_in = df_full.drop(columns=[target_col]).head(20)

st.write("### Input")
st.dataframe(df_in.head(50))

if st.button("Predict"):
    model = joblib.load(MODEL_PATH)

    # Try scoring with the raw frame first; if that fails, drop common ID columns.
    try:
        preds = model.predict(df_in)
        proba = model.predict_proba(df_in)[:, 1] if hasattr(model, "predict_proba") else None
    except Exception:
        id_cols = [c for c in df_in.columns if c.endswith("_id") or c in ["customer_id","application_id","transaction_id"]]
        X = df_in.drop(columns=id_cols) if id_cols else df_in
        preds = model.predict(X)
        proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    out = pd.DataFrame({"prediction": preds})
    if proba is not None:
        out["probability"] = proba

    st.write("### Output")
    st.dataframe(out)
