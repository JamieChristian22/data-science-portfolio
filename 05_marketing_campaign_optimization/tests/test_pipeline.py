import pandas as pd
from src.marketing_pipeline import top_fraction

def test_rate():
 df=pd.read_csv('data/campaign_contacts.csv'); assert 0<df.response.mean()<.15

def test_top():
 assert list(top_fraction([.1,.9,.3,.8],.5))==[1,3]
