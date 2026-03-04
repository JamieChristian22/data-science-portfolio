import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "credit_risk.csv")

X = df.drop(columns=["defaulted"])
y = df["defaulted"]

cat_cols = ["loan_purpose"]
num_cols = [c for c in X.columns if c not in cat_cols + ["application_id"]]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

pipe = Pipeline([("prep", pre), ("model", GradientBoostingClassifier(random_state=42))])
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
pipe.fit(Xtr,ytr)
proba = pipe.predict_proba(Xte)[:,1]

(ROOT/"models").mkdir(exist_ok=True)
(ROOT/"reports").mkdir(exist_ok=True)
joblib.dump(pipe, ROOT/"models"/"credit_risk_gb_retrained.joblib")
(Path(ROOT/"reports"/"metrics_recomputed.json")).write_text(json.dumps({"roc_auc": float(roc_auc_score(yte, proba))}, indent=2))
print("Done.")
