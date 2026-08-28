# Data Dictionary

| Field | Type | Description |
|---|---|---|
| customer_id | string | Synthetic unique customer identifier |
| tenure_months | integer | Months the customer has been active |
| monthly_charges | float | Current monthly service charge in USD |
| total_charges | float | Approximate cumulative charges in USD |
| contract_type | category | Month-to-month, One year, or Two year |
| internet_service | category | DSL, Fiber optic, or None |
| paperless_billing | category | Whether paperless billing is enabled |
| support_calls_last_90d | integer | Support contacts in the prior 90 days |
| late_payments_last_12m | integer | Late payments in the prior 12 months |
| satisfaction_score_1to5 | float | Customer satisfaction score from 1 to 5 |
| churned | binary | 1 = churned, 0 = retained |
