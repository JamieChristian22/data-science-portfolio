# 📓 Notebooks — Data Science Portfolio

This folder showcases the Jupyter notebooks used throughout my data science journey — from EDA and SQL to machine learning and API projects.  
Each notebook is fully runnable in **JupyterLab**, **VS Code**, or **Google Colab**, and designed to demonstrate real-world data analytics workflows.

---

## 🗂 Notebook Index

| Filename | Description | Skills / Tools |
|-----------|--------------|----------------|
| **DV0101EN-Final-Assign-Part1.ipynb** | IBM Data Science course final — EDA, descriptive stats, and visualization fundamentals. | Python, Pandas, Matplotlib |
| **DataScienceEcosystem.ipynb** | Overview of tools, languages, and libraries used in modern data science. | Markdown, Documentation |
| **Final Assignment-2.ipynb** | Continuation of course capstone; applies data wrangling and SQL queries to a business dataset. | SQL, Pandas, Joins |
| **FinalProject_AUSWeather.ipynb** | Full project predicting Australian weather outcomes based on meteorological data. | ML (Pandas, Scikit-Learn, Seaborn) |
| **House_Sales_in_King_Count_USA.ipynb** | Regression modeling on U.S. house prices dataset — data cleaning, feature engineering, prediction. | Linear Regression, EDA, Matplotlib |
| **SpaceX_Machine Learning Prediction_Part_5.ipynb** | Machine learning pipeline predicting Falcon 9 landing success with visualization and model tuning. | Logistic Regression, SVM, Decision Tree, KNN, Elastic-Net |
| **edadataviz (1).ipynb** | Exploratory data analysis and visualization mini-project with plots and trend analysis. | Pandas, Seaborn |
| **jupyter-labs-eda-sql-coursera_sqllite (1).ipynb** | SQL Lite practice notebook — querying, filtering, and aggregating datasets within Python. | SQL, SQLite3 |
| **jupyter-labs-spacex-data-collection-api.ipynb** | Collecting live data from the SpaceX API and preparing it for analysis. | API Requests, JSON, Data Wrangling |
| **jupyter-labs-webscraping.ipynb** | Web scraping with Python — extracting HTML tables and saving clean datasets. | BeautifulSoup, Requests |
| **labs-jupyter-spacex-Data wrangling.ipynb** | Cleaning and merging datasets for the SpaceX analysis pipeline. | Data Wrangling, ETL |

---

## ⚙️ How to Run

1. Activate your virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r ../requirements.txt
   ```
2. Launch Jupyter:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
3. Open any notebook under `notebooks/` to explore the analysis.

---

## 🧭 Best Practices

- Use relative paths (`../data_files/...`) for datasets and visuals.  
- Each notebook begins with a **Problem Statement → Data → Method → Results → Insights** layout.  
- Keep outputs under 5 MB for smooth GitHub preview.  
- Add visuals from the `Visuals/` folder using  
  ```markdown
  ![Confusion Matrix](../Visuals/confusion_matrix_svm.png)
  ```  
- Random seeds and environment versions are pinned for reproducibility.

---

## 💡 Next Steps

- [ ] Add a unified notebook index with links from the main README.  
- [ ] Export top projects (`SpaceX`, `AUSWeather`, `House Sales`) to PDF summaries.  
- [ ] Add runtime badges (e.g., Python 3.11 | JupyterLab 4).  

---

> *“Each notebook tells a story — from messy data to measurable insight.”*  
> — **Jamie Christian**
