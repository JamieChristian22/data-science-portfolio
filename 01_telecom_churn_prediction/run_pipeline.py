import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "telecom_churn.csv")

X = df.drop(columns=["churned"])
y = df["churned"]

cat_cols = ["contract_type","internet_service","paperless_billing"]
num_cols = [c for c in X.columns if c not in cat_cols + ["customer_id"]]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

pipe = Pipeline([("prep", pre), ("model", LogisticRegression(max_iter=1000))])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.22, random_state=42, stratify=y)
pipe.fit(Xtr, ytr)
proba = pipe.predict_proba(Xte)[:,1]

(ROOT/"models").mkdir(exist_ok=True)
(ROOT/"reports").mkdir(exist_ok=True)
joblib.dump(pipe, ROOT/"models"/"churn_logreg_retrained.joblib")
(Path(ROOT/"reports"/"metrics_recomputed.json")).write_text(json.dumps({"roc_auc": float(roc_auc_score(yte, proba))}, indent=2))
print("Done.")
