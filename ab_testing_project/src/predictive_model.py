"""
Predictive modeling: can we predict whether a player will still be
active 7 days after install, based on their early behavior?

This is the "predictive modeling" half of the resume bullet, distinct
from the A/B test itself. The A/B test tells us WHETHER the gate change
mattered; this model tells us WHAT early signals predict retention,
which is useful even outside the context of this one experiment.

Model: Logistic Regression (interpretable baseline) and Random Forest
(usually stronger, gives feature importances) predicting retention_7
from version + sum_gamerounds + retention_1.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame({
        "is_gate_40": (df["version"] == "gate_40").astype(int),
        "sum_gamerounds": df["sum_gamerounds"],
        "retention_1": df["retention_1"].astype(int),
    })
    y = df["retention_7"].astype(int)
    return X, y


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }


def train_and_evaluate(df: pd.DataFrame) -> dict:
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)  # tree models don't need scaling

    results = {
        "logistic_regression": evaluate(log_reg, X_test_scaled, y_test),
        "random_forest": evaluate(rf, X_test, y_test),
        "random_forest_feature_importance": dict(
            zip(X.columns, rf.feature_importances_)
        ),
        "logistic_regression_coefficients": dict(
            zip(X.columns, log_reg.coef_[0])
        ),
        "baseline_positive_rate": y.mean(),  # what you'd get by always predicting "no"
    }
    return results


if __name__ == "__main__":
    df = pd.read_csv("data/cookie_cats_clean.csv")
    results = train_and_evaluate(df)
    for k, v in results.items():
        print(f"\n{k}: {v}")
