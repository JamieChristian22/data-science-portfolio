# Model Card — Credit Risk Default Prediction

## Intended use
Educational portfolio demonstration of probability-of-default modeling and risk decision support.

## Not intended for
Real underwriting, pricing, adverse action, or automated lending decisions.

## Selected model
**Logistic Regression**, selected using validation PR-AUC, then probability-calibrated using sigmoid calibration.

## Holdout performance
- ROC-AUC: **0.720**
- PR-AUC: **0.483**
- Recall at business threshold: **0.848**
- Precision at business threshold: **0.365**
- F1: **0.510**
- F2: **0.670**
- Brier score: **0.174**
- KS statistic: **0.348**
- Threshold: **0.19**, selected on validation data using a 5:1 false-negative:false-positive cost weighting.

## Key safeguards
Protected characteristics are intentionally absent. Production use would still require fair-lending review, explainability, model risk management, stability tests, outcome monitoring, data quality controls, adverse-action processes, and documented human oversight.

## Limitations
The dataset is synthetic. Performance therefore demonstrates workflow competence, not production lending validity or expected real-world performance.
