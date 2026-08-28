from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = joblib.load(ROOT / "models" / "churn_model.joblib")
model = ARTIFACT["model"]
threshold = ARTIFACT["threshold"]

st.set_page_config(page_title="Telecom Churn Risk", page_icon="📉", layout="centered")
st.title("Telecom Customer Churn Risk")
st.caption("Portfolio demo — probability scoring plus business-threshold decisioning")

c1, c2 = st.columns(2)
with c1:
    tenure = st.slider("Tenure (months)", 0, 72, 18)
    monthly = st.number_input("Monthly charges ($)", 20.0, 140.0, 85.0, 1.0)
    contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "None"])
with c2:
    support = st.slider("Support calls — last 90 days", 0, 8, 2)
    late = st.slider("Late payments — last 12 months", 0, 5, 1)
    satisfaction = st.slider("Satisfaction score", 1.0, 5.0, 3.0, .1)
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])

total = monthly * max(tenure, 1)
row = pd.DataFrame([{
    "tenure_months": tenure, "monthly_charges": monthly, "total_charges": total,
    "contract_type": contract, "internet_service": internet, "paperless_billing": paperless,
    "support_calls_last_90d": support, "late_payments_last_12m": late,
    "satisfaction_score_1to5": satisfaction,
}])

if st.button("Score churn risk", type="primary"):
    p = float(model.predict_proba(row)[:,1][0])
    st.metric("Predicted churn probability", f"{p:.1%}")
    st.progress(min(max(p,0.0),1.0))
    if p >= threshold:
        st.error(f"Retention action recommended (business threshold = {threshold:.2f}).")
    else:
        st.success(f"Monitor — below business threshold ({threshold:.2f}).")
    st.caption("This demonstration is for portfolio use and should not be used as a production customer decision system without monitoring, validation, fairness review, and governance.")
