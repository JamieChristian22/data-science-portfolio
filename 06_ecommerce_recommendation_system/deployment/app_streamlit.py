
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Recommender Demo", layout="centered")
st.title("E-Commerce Recommendation System — Demo")

ROOT = Path(__file__).resolve().parents[1]
PURCHASES = pd.read_csv(ROOT / "data" / "purchases.csv")
PRODUCTS = pd.read_csv(ROOT / "data" / "products.csv")
MODEL_PATH = ROOT / "models" / "svd_model.joblib"

st.caption("Select a user and generate top-N recommendations using the trained SVD model.")

user_ids = sorted(PURCHASES["user_id"].unique().tolist())
user = st.selectbox("User ID", user_ids)
topn = st.slider("Top N recommendations", 5, 20, 10)

if st.button("Recommend"):
    model_bundle = joblib.load(MODEL_PATH)
    svd = model_bundle["svd"]
    user_index = model_bundle["user_index"]
    item_cols = model_bundle["item_columns"]

    # Build user-item matrix consistent with training
    ui = pd.pivot_table(PURCHASES, index="user_id", columns="product_id", values="quantity", aggfunc="sum", fill_value=0)
    # Align columns
    ui = ui.reindex(columns=item_cols, fill_value=0)

    if user not in ui.index:
        st.error("User not found in matrix.")
        st.stop()

    # Compute embeddings on-the-fly (fast enough for demo)
    user_emb = svd.transform(ui.values)
    item_emb = svd.components_.T

    u_idx = list(ui.index).index(user)
    uvec = user_emb[u_idx]
    scores = item_emb @ uvec

    already = set(ui.columns[ui.loc[user].values > 0])
    ranked = [ui.columns[i] for i in np.argsort(scores)[::-1] if ui.columns[i] not in already][:topn]

    rec_df = pd.DataFrame({"recommended_product_id": ranked})
    rec_df = rec_df.merge(PRODUCTS, left_on="recommended_product_id", right_on="product_id", how="left").drop(columns=["product_id"])
    st.dataframe(rec_df)
