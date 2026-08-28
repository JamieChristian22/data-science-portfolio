import streamlit as st,pandas as pd
st.title('Fraud Investigation Queue Demo')
st.caption('Synthetic portfolio data; not for real payment authorization.')
df=pd.read_csv('reports/holdout_scored_transactions.csv').sort_values('fraud_probability',ascending=False)
st.dataframe(df.head(100))
