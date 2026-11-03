# 🚀 Data Science Portfolio

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge)
![Data Science](https://img.shields.io/badge/Data%20Science-1E90FF?style=for-the-badge)
![SQL](https://img.shields.io/badge/SQL-336791?style=for-the-badge&logo=postgresql&logoColor=white)

[![Sponsor on GitHub](https://img.shields.io/badge/Sponsor%20this%20project-GitHub%20Sponsors-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/JamieChristian22)

End-to-end **data science and machine learning portfolio projects** demonstrating predictive modeling, forecasting, and classification. A recommendation-system concept is outlined as future work.

These projects simulate **real-world business problems** across industries including telecom, finance, retail, marketing, and ecommerce.

---

## Five runnable project folders · Reviewer quick start

[![Project tests](https://github.com/JamieChristian22/data-science-portfolio/actions/workflows/portfolio-tests.yml/badge.svg)](https://github.com/JamieChristian22/data-science-portfolio/actions/workflows/portfolio-tests.yml)
[![View portfolio website](https://img.shields.io/badge/Portfolio-Website-0f766e)](https://jamiechristian22.github.io/)

| Project | Business decision | Evidence |
| --- | --- | --- |
| [01 · Telecom churn](01_telecom_churn_prediction/README.md) | Identify retention opportunities | [Model card](01_telecom_churn_prediction/reports/model_card.md) · [PR curve](01_telecom_churn_prediction/figures/precision_recall_curve.png) · [Tests](01_telecom_churn_prediction/tests/test_pipeline.py) |
| [02 · Credit risk](02_credit_risk_default_prediction/README.md) | Assess default probabilities and threshold trade-offs | [Cross-validation](02_credit_risk_default_prediction/reports/cross_validation.json) · [Tests](02_credit_risk_default_prediction/tests/test_pipeline.py) |
| [03 · Retail demand forecast](03_retail_demand_forecasting/README.md) | Reduce forecast error for inventory decisions | [Rolling backtest](03_retail_demand_forecasting/reports/rolling_backtest.csv) · [Tests](03_retail_demand_forecasting/tests/test_pipeline.py) |
| [04 · Card fraud detection](04_card_fraud_detection/README.md) | Prioritize costly and imbalanced alerts | [PR curve](04_card_fraud_detection/figures/precision_recall.png) · [Tests](04_card_fraud_detection/tests/test_pipeline.py) |
| [05 · Marketing optimization](05_marketing_campaign_optimization/README.md) | Rank outreach by predicted response | [Lift by decile](05_marketing_campaign_optimization/reports/lift_by_decile.csv) · [Tests](05_marketing_campaign_optimization/tests/test_pipeline.py) |

### Reproduce a project locally

Each project has its own dependencies and execution steps. From the repository root, for example:

```bash
cd 01_telecom_churn_prediction
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest tests -q
python run_pipeline.py
```

Use the same pattern for the other projects and consult each project README. The [GitHub Actions workflow](.github/workflows/portfolio-tests.yml) now runs the five existing test suites independently; **do not assume they pass until the Actions page confirms it**.

### Evidence and model-quality boundaries

- Datasets are simulation / learning materials unless specifically stated otherwise. Model results are not real production business outcomes.
- Compare useful baselines and held-out evaluation sets. For forecasting, preserve chronological splits and inspect rolling backtests.
- For fraud and churn, distinguish ROC-AUC from PR-AUC, and discuss false positives, calibration and class imbalance.
- Inspect notebooks, saved models, metrics, data dictionaries and scripts. A screenshot alone does not establish model quality.
- **Project inventory:** Five project directories exist. A recommendation-engine concept is described below as a future extension, not a runnable sixth project.
- Generated Python cache files have been removed from version control. Existing ignore rules help prevent them from being committed again.

---

# 📊 Portfolio Projects

## 📉 Telecom Customer Churn Prediction

**Business Problem**

Telecom companies lose millions in revenue due to customer churn.
This project predicts which customers are most likely to cancel their service so retention teams can intervene early.

**Tech Stack**

Python
Pandas
Scikit-learn
Matplotlib

**Model**

Logistic Regression Classification

**Key Outputs**

Customer churn probability predictions
ROC curve evaluation
Confusion matrix analysis

---

## 💳 Credit Risk Default Prediction

**Business Problem**

Financial institutions must evaluate borrower risk before issuing loans.

This project predicts whether a loan applicant is likely to default.

**Tech Stack**

Python
Pandas
Scikit-learn

**Model**

Gradient Boosting Classifier

**Key Outputs**

Loan default probability predictions
Risk segmentation modeling
Model performance evaluation

---

## 🛒 Retail Demand Forecasting

**Business Problem**

Retail companies must forecast demand accurately to avoid stockouts and excess inventory.

This project predicts product demand using historical sales data.

**Tech Stack**

Python
Pandas
Scikit-learn
Matplotlib

**Model**

Random Forest Regression

**Key Outputs**

Daily product demand forecasts
Forecast visualizations
Mean Absolute Error performance metrics

---

## 🕵️ Credit Card Fraud Detection

**Business Problem**

Banks must detect fraudulent transactions quickly to prevent financial loss.

This project builds a machine learning model to identify suspicious transactions.

**Tech Stack**

Python
Pandas
Scikit-learn

**Model**

Logistic Regression Fraud Detection Model

**Key Outputs**

Fraud probability scoring
Precision-recall evaluation
Fraud detection threshold tuning

---

## 📢 Marketing Campaign Optimization

**Business Problem**

Marketing teams want to target customers who are most likely to respond to campaigns.

This project predicts campaign response probability to improve marketing ROI.

**Tech Stack**

Python
Pandas
Scikit-learn

**Model**

Logistic Regression

**Key Outputs**

Customer response prediction
Lift curve evaluation
Campaign targeting recommendations

---

## 🧠 Future extension: ecommerce recommendation system (not yet implemented)

**Business Problem**

Ecommerce platforms increase sales through personalized product recommendations.

**Planned concept, not a completed project in this repository:** a recommendation engine using collaborative filtering.

**Tech Stack**

Python
Pandas
Scikit-learn

**Model**

Matrix Factorization using Truncated SVD

**Key Outputs**

Personalized product recommendations
Precision@K evaluation metrics
User-item interaction modeling

---

# 🧰 Tools & Technologies

🐍 Python
📊 Pandas
🔢 NumPy
🤖 Scikit-learn
📈 Matplotlib
🗄 SQL
📉 Machine Learning
📊 Predictive Analytics

---

# 🧠 Skills Demonstrated

Data Cleaning and Preparation
Exploratory Data Analysis
Feature Engineering
Machine Learning Modeling
Model Evaluation and Metrics
Data Visualization
Business Problem Translation

---

# 👨‍💻 About Me

**Jamie Christian**

Business and technology professional building independent data analytics and predictive-modeling case studies.

🔗 LinkedIn
https://www.linkedin.com/in/jamiechristian2/

💻 GitHub
https://github.com/JamieChristian22

---

# 🎯 Portfolio Goal

This repository demonstrates practical **data science and machine learning solutions applied to real-world business problems** across multiple industries.

The goal is to showcase **end-to-end analytics workflows including data preparation, modeling, evaluation, and business insight generation.**

---

## 💖 Support This Project

If the data science and machine-learning projects in this portfolio are useful to you, consider [supporting independent development through GitHub Sponsors](https://github.com/sponsors/JamieChristian22).

[![Become a GitHub Sponsor](https://img.shields.io/badge/Become%20a-GitHub%20Sponsor-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/JamieChristian22)

Sponsorship is entirely optional and helps support documentation, educational experiments, testing, and maintenance. You can also help by starring the repository, sharing it, or submitting constructive issues and contributions. Sponsorship does not guarantee model features, outcomes, or influence evaluation results.

## License and third-party materials

See [LICENSE](LICENSE) for the repository license. Third-party code, datasets, images, and other assets retain their respective terms.

## Reviewer guide and asset provenance

For a guided project review and sample dataset/media provenance checks, see [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md). Asset listings do not establish ownership or license clearance.
