# Streamlit Deployment

From the project root:

```bash
pip install -r requirements.txt
python run_pipeline.py
streamlit run deployment/app_streamlit.py
```

The app loads the versioned `models/churn_model.joblib` artifact, uses the validation-selected business threshold, and returns both a churn probability and retention action.
