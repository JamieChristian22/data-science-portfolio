from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             fbeta_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_validate

TARGET = "churned"
ID_COL = "customer_id"
NUMERIC = ["tenure_months", "monthly_charges", "total_charges",
           "support_calls_last_90d", "late_payments_last_12m", "satisfaction_score_1to5"]
CATEGORICAL = ["contract_type", "internet_service", "paperless_billing"]
FEATURES = NUMERIC + CATEGORICAL


def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], remainder="drop")


def candidate_models(seed=42):
    return {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor()),
            ("model", RandomForestClassifier(n_estimators=450, min_samples_leaf=4,
                                             class_weight="balanced_subsample", random_state=seed, n_jobs=-1)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", preprocessor()),
            ("model", HistGradientBoostingClassifier(max_iter=250, learning_rate=.045,
                                                      max_leaf_nodes=15, l2_regularization=1.0,
                                                      random_state=seed)),
        ]),
    }


def classification_metrics(y_true, prob, threshold=.5):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(y_true, prob, beta=2, min_precision=.25):
    rows = []
    for t in np.arange(.10, .81, .01):
        m = classification_metrics(y_true, prob, threshold=float(t))
        rows.append(m)
    df = pd.DataFrame(rows)
    eligible = df[df["precision"] >= min_precision]
    best = (eligible if not eligible.empty else df).sort_values(["f2", "recall"], ascending=False).iloc[0]
    return float(best["threshold"]), df


def cv_summary(model, X, y, seed=42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_validate(model, X, y, cv=cv,
                            scoring={"roc_auc":"roc_auc", "pr_auc":"average_precision", "recall":"recall", "f1":"f1"},
                            n_jobs=-1)
    return {k.replace("test_", "cv_"): float(np.mean(v)) for k, v in scores.items() if k.startswith("test_")}
