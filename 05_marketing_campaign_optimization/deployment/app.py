import streamlit as st,pandas as pd
st.title('Campaign Targeting & Uplift Demo')
st.caption('Synthetic educational campaign data.')
st.dataframe(pd.read_csv('reports/holdout_customer_scores.csv').sort_values('response_score',ascending=False).head(100))
