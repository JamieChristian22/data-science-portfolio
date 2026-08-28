
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, f1_score, brier_score_loss, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]

def load_data(path=None):
    path = path or ROOT / "data" / "credit_risk.csv"
    df = pd.read_csv(path)
    required = {"application_id","annual_income","loan_amount","loan_term_months","interest_rate_pct",
                "employment_years","credit_score","debt_to_income","delinquencies_last_2y",
                "credit_utilization","loan_purpose","defaulted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["defaulted"].isna().any():
        raise ValueError("Target contains missing values.")
    return df

def make_preprocessor(X):
    cat = ["loan_purpose"]
    num = [c for c in X.columns if c not in cat + ["application_id"]]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", StandardScaler(), num)
    ], verbose_feature_names_out=False)

def make_candidates(X):
    pre = make_preprocessor(X)
    return {
        "Logistic Regression": Pipeline([("prep", pre), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))]),
        "Random Forest": Pipeline([("prep", pre), ("model", RandomForestClassifier(n_estimators=350, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1))]),
        "HistGradientBoosting": Pipeline([("prep", pre), ("model", HistGradientBoostingClassifier(max_iter=350, learning_rate=.04, max_leaf_nodes=20, l2_regularization=.8, random_state=42))])
    }

def choose_threshold(y_true, proba, fn_cost=5, fp_cost=1):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    best = None
    for t in np.arange(.10, .71, .01):
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        cost = fn*fn_cost + fp*fp_cost
        row = (cost, float(t), tn, fp, fn, tp)
        if best is None or row[0] < best[0]:
            best = row
    return best[1]

def main():
    df = load_data()
    X, y = df.drop(columns="defaulted"), df["defaulted"]
    Xtv, Xtest, ytv, ytest = train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    Xtr, Xval, ytr, yval = train_test_split(Xtv,ytv,test_size=.25,random_state=42,stratify=ytv)
    candidates = make_candidates(X)
    scores = {}
    fitted = {}
    for name, pipe in candidates.items():
        pipe.fit(Xtr.drop(columns=["application_id"]), ytr)
        p = pipe.predict_proba(Xval.drop(columns=["application_id"]))[:,1]
        scores[name] = {"roc_auc": roc_auc_score(yval,p), "pr_auc": average_precision_score(yval,p)}
        fitted[name] = pipe
    selected_name = max(scores, key=lambda k: scores[k]["pr_auc"])
    val_cal = CalibratedClassifierCV(fitted[selected_name], method="sigmoid", cv=3)
    val_cal.fit(Xtr.drop(columns=["application_id"]), ytr)
    pval = val_cal.predict_proba(Xval.drop(columns=["application_id"]))[:,1]
    threshold = choose_threshold(yval, pval)

    final = CalibratedClassifierCV(fitted[selected_name], method="sigmoid", cv=5)
    final.fit(Xtv.drop(columns=["application_id"]), ytv)
    ptest = final.predict_proba(Xtest.drop(columns=["application_id"]))[:,1]
    pred = (ptest >= threshold).astype(int)

    metrics = {
        "selected_model": selected_name,
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(ytest,ptest)),
        "pr_auc": float(average_precision_score(ytest,ptest)),
        "precision": float(precision_score(ytest,pred)),
        "recall": float(recall_score(ytest,pred)),
        "f1": float(f1_score(ytest,pred)),
        "brier": float(brier_score_loss(ytest,ptest))
    }
    (ROOT/"models").mkdir(exist_ok=True)
    (ROOT/"reports").mkdir(exist_ok=True)
    joblib.dump({"model":final,"threshold":threshold}, ROOT/"models"/"credit_risk_model.joblib")
    (ROOT/"reports"/"metrics_recomputed.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
