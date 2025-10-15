# 🌦️ Capstone 1: Weather to Rain Prediction

**Dataset:** weatherAUS.csv  
**Tools:** Python, Pandas, Matplotlib, Seaborn  

### Objective
Analyze weather patterns to predict rainfall likelihood and temperature trends across major Australian cities.

### Key Insights
- **Humidity3pm** shows the strongest positive correlation (r = 0.68) with rainfall.
- Temperature range (MaxTemp - MinTemp) narrows on days with rainfall.
- Sydney and Brisbane exhibit higher rainfall frequencies during January–March.

### Visuals
- Temperature Distribution Plot (`temp_distribution.png`)
- Correlation Heatmap (`weather_corr.png`)

### Next Steps
- Build logistic regression model for RainToday classification.
- Add time-series decomposition for seasonal rainfall trends.
