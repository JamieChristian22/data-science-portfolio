import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT/"data"/"transactions.csv")

X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]
num_cols = [c for c in X.columns if c != "transaction_id"]

pre = ColumnTransformer([("num", StandardScaler(), num_cols)])
pipe = Pipeline([("prep", pre), ("model", LogisticRegression(max_iter=1200, class_weight="balanced"))])

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
pipe.fit(Xtr,ytr)
proba = pipe.predict_proba(Xte)[:,1]

(ROOT/"models").mkdir(exist_ok=True)
(ROOT/"reports").mkdir(exist_ok=True)
joblib.dump(pipe, ROOT/"models"/"fraud_logreg_retrained.joblib")
(Path(ROOT/"reports"/"metrics_recomputed.json")).write_text(json.dumps({"roc_auc": float(roc_auc_score(yte, proba))}, indent=2))
print("Done.")
