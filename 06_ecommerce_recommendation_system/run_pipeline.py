import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parent
purchases = pd.read_csv(ROOT/"data"/"purchases.csv")

ui = pd.pivot_table(purchases, index="user_id", columns="product_id", values="quantity", aggfunc="sum", fill_value=0)

svd = TruncatedSVD(n_components=28, random_state=42)
user_emb = svd.fit_transform(ui.values)
item_emb = svd.components_.T

(ROOT/"models").mkdir(exist_ok=True)
(ROOT/"reports").mkdir(exist_ok=True)
joblib.dump({"svd": svd, "user_index": ui.index.tolist(), "item_columns": ui.columns.tolist()}, ROOT/"models"/"svd_model_retrained.joblib")
np.save(ROOT/"models"/"user_embeddings_retrained.npy", user_emb)
np.save(ROOT/"models"/"item_embeddings_retrained.npy", item_emb)

(Path(ROOT/"reports"/"metrics_recomputed.json")).write_text(json.dumps({
    "n_users": int(ui.shape[0]),
    "n_products": int(ui.shape[1]),
    "sparsity": float(1 - (ui.values > 0).mean())
}, indent=2))
print("Done.")
