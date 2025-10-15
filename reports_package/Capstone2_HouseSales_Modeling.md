# 🏡 Capstone 2: House Sales Price Modeling

**Dataset:** house_sales.csv  
**Tools:** Python, Pandas, Matplotlib, Scikit-Learn  

### Objective
Develop a predictive model to estimate housing prices based on living area, bedrooms, and property features.

### Key Insights
- Strong correlation between `sqft_living` and `price` (r = 0.72).
- Homes with `waterfront = 1` have 35% higher average prices.
- Feature `price_per_sqft` improves model stability and interpretability.

### Visuals
- Regression Plot (`house_price_regression.png`)
- Feature Importance Bar Chart (`feature_importance.png`)

### Next Steps
- Test tree-based models (RandomForest, XGBoost).
- Add SHAP summary for explainability.
