# E-Commerce Recommendation System (Collaborative Filtering)

## Business Goal
Recommend products customers are likely to buy next to increase conversion and AOV.

## Dataset
- Purchases: `data/purchases.csv`
- Products: `data/products.csv`
Users: **600** | Products: **320** | Sparsity: **95.7%**

## Method
- User×item matrix from purchase quantities (implicit feedback)
- TruncatedSVD embeddings
- Score items via dot product and filter already-purchased items

## Output
- `reports/sample_recommendations.csv` (10 recommendations for 8 sample users)

Artifacts:
- Model: `models/svd_model.joblib`
- Embeddings: `models/user_embeddings.npy`, `models/item_embeddings.npy`
- Charts: `figures/purchases_per_user_hist.png`, `figures/catalog_category_counts.png`
