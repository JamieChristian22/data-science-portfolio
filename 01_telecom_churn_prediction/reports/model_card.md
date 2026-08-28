# Model Card — Telecom Churn Classifier

## Intended use
Portfolio demonstration of customer churn risk scoring and retention prioritization.

## Model
Interpretable logistic regression with standardized numeric features, one-hot encoded categorical features, and class balancing. A business threshold is selected on a validation set using F2 while enforcing a minimum precision constraint.

## Why logistic regression
The project intentionally values interpretability, stable probabilities, and transparent business communication. Tree-based alternatives are benchmarked and retained in the comparison report.

## Evaluation
A stratified train/validation/test split is used. Threshold selection occurs only on validation data; final metrics are reported on the untouched test set. Five-fold stratified cross-validation is also reported for model comparison.

## Limitations
The dataset is synthetic and does not represent a real telecom customer population. No fairness claim should be inferred. In a production setting, monitoring would be required for drift, calibration, subgroup performance, retention-offer effects, and data quality.

## Governance
This model should support human decision-making rather than automatically deny, restrict, or materially alter customer service.
