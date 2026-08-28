import streamlit as st,pandas as pd
st.title('Retail Demand Forecasting Dashboard')
st.line_chart(pd.read_csv('reports/holdout_forecast.csv').set_index('date'))
