
from pathlib import Path
import pandas as pd, numpy as np, joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[1]
F=["amount","merchant_risk","transactions_last_hour","distance_from_home_km","card_present","foreign_transaction"]
def top_k_alerts(scores,pct=.01):
    s=np.asarray(scores); k=max(1,int(len(s)*pct)); a=np.zeros(len(s),int); a[np.argsort(-s)[:k]]=1; return a
def main():
    df=pd.read_csv(ROOT/"data/transactions.csv"); split=int(len(df)*.8); tr,te=df.iloc[:split],df.iloc[split:]
    m=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(class_weight="balanced",max_iter=2000,random_state=42))]);m.fit(tr[F],tr.fraud)
    joblib.dump(m,ROOT/"models/fraud_supervised_retrained.joblib");print("ok")
if __name__=="__main__":main()
