# Data Dictionary

The included dataset is **deterministic synthetic portfolio data** created to mirror the schema and approximate 26.7% default rate documented in the original project. It contains no real applicants or personal data.

| Field | Type | Description |
|---|---|---|
| application_id | string | Synthetic unique application identifier |
| annual_income | numeric | Annual applicant income in USD |
| loan_amount | numeric | Requested principal in USD |
| loan_term_months | integer | Requested term |
| interest_rate_pct | numeric | Demonstration interest rate |
| employment_years | numeric | Years of employment |
| credit_score | integer | Synthetic credit score |
| debt_to_income | numeric | Debt-to-income ratio |
| delinquencies_last_2y | integer | Delinquency count |
| credit_utilization | numeric | Revolving utilization ratio |
| loan_purpose | categorical | Loan purpose |
| defaulted | binary | Synthetic outcome; 1 = default, 0 = no default |
