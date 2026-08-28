
from pathlib import Path
import joblib, pandas as pd, streamlit as st

ROOT = Path(__file__).resolve().parents[1]
bundle = joblib.load(ROOT / "models" / "credit_risk_model.joblib")
model, threshold = bundle["model"], bundle["threshold"]

st.set_page_config(page_title="Credit Risk Decision Support", page_icon="📊", layout="centered")
st.title("Credit Risk Decision Support Demo")
st.caption("Portfolio demonstration only — not a real lending decision system.")

with st.form("application"):
    annual_income = st.number_input("Annual income", 18000.0, 250000.0, 65000.0, step=1000.0)
    loan_amount = st.number_input("Loan amount", 1500.0, 50000.0, 15000.0, step=500.0)
    loan_term_months = st.selectbox("Loan term (months)", [24,36,48,60], index=1)
    interest_rate_pct = st.slider("Interest rate (%)", 4.0, 29.0, 11.0, .1)
    employment_years = st.slider("Employment years", 0.0, 25.0, 5.0, .5)
    credit_score = st.slider("Credit score", 500, 850, 680)
    debt_to_income = st.slider("Debt-to-income ratio", 0.02, .72, .30, .01)
    delinquencies_last_2y = st.slider("Delinquencies in last 2 years", 0, 8, 0)
    credit_utilization = st.slider("Credit utilization", .03, .99, .40, .01)
    loan_purpose = st.selectbox("Loan purpose", ["debt_consolidation","home_improvement","medical","auto","small_business","other"])
    submitted = st.form_submit_button("Estimate default risk")

if submitted:
    row = pd.DataFrame([{
        "annual_income":annual_income,"loan_amount":loan_amount,"loan_term_months":loan_term_months,
        "interest_rate_pct":interest_rate_pct,"employment_years":employment_years,"credit_score":credit_score,
        "debt_to_income":debt_to_income,"delinquencies_last_2y":delinquencies_last_2y,
        "credit_utilization":credit_utilization,"loan_purpose":loan_purpose
    }])
    pd_value = float(model.predict_proba(row)[:,1][0])
    if pd_value < .10: band = "Low"
    elif pd_value < .20: band = "Moderate"
    elif pd_value < .30: band = "Elevated"
    elif pd_value < .45: band = "High"
    else: band = "Very High"
    st.metric("Estimated probability of default", f"{pd_value:.1%}")
    st.write(f"**Risk band:** {band}")
    st.write(f"**Model operating threshold:** {threshold:.2f}")
    st.warning("This app is an educational portfolio demo. Real credit decisions require validated data, legal/compliance review, fairness testing, adverse-action processes, monitoring, and human governance.")
