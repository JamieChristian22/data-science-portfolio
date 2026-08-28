import pandas as pd
from src.fraud_pipeline import top_k_alerts

def test_rate():
 df=pd.read_csv('data/transactions.csv'); assert 0<df.fraud.mean()<.02

def test_topk():
 a=top_k_alerts([.1,.9,.3,.8],.5); assert a.sum()==2 and a[1]==1 and a[3]==1
