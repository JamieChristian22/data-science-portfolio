
from pathlib import Path
import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[1]
F=["age","income","prior_purchases","email_engagement","tenure_months","channel","discount_pct"]
def top_fraction(scores,frac=.1):
    s=np.asarray(scores); return np.argsort(-s)[:max(1,int(len(s)*frac))]
def main():
    df=pd.read_csv(ROOT/"data/campaign_contacts.csv");tr,te=train_test_split(df,test_size=.2,random_state=42,stratify=df.response)
    prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),["channel"]),("num",StandardScaler(),[x for x in F if x!="channel"])])
    m=Pipeline([("prep",prep),("model",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42))]);m.fit(tr[F],tr.response)
    joblib.dump(m,ROOT/"models/campaign_response_model_retrained.joblib");print("ok")
if __name__=="__main__":main()
