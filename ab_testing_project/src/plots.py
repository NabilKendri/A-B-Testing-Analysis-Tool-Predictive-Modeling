"""
Generates the figures used in the README / report.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

from predictive_model import build_features

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})

COLOR_30 = "#4C72B0"
COLOR_40 = "#DD8452"


def plot_retention(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ["retention_1", "retention_7"]
    labels = ["Day-1 retention", "Day-7 retention"]
    x = np.arange(len(metrics))
    width = 0.35

    for i, version in enumerate(["gate_30", "gate_40"]):
        sub = df[df["version"] == version]
        rates = [sub[m].mean() for m in metrics]
        ns = [len(sub)] * len(metrics)
        errs = [1.96 * np.sqrt(r * (1 - r) / n) for r, n in zip(rates, ns)]
        ax.bar(
            x + (i - 0.5) * width, rates, width,
            yerr=errs, capsize=4,
            label=version, color=[COLOR_30, COLOR_40][i],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Retention rate")
    ax.set_title("Retention by gate version (95% CI)")
    ax.legend()
    # annotate significance on day-7
    ax.annotate("p = 0.0016 **", xy=(1, max(df[df.version=="gate_30"].retention_7.mean(),
                                             df[df.version=="gate_40"].retention_7.mean()) + 0.015),
                ha="center", fontsize=10, color="#333333")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gamerounds_distribution(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    capped = df["sum_gamerounds"].clip(upper=200)  # tail is long; cap for readability
    for version, color in [("gate_30", COLOR_30), ("gate_40", COLOR_40)]:
        sub = capped[df["version"] == version]
        ax.hist(sub, bins=40, alpha=0.55, label=version, color=color, density=True)
    ax.set_xlabel("Rounds played in first 14 days (capped at 200 for display)")
    ax.set_ylabel("Density")
    ax.set_title("Engagement distribution by gate version")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(df: pd.DataFrame, out_path: Path):
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(importances.index, importances.values, color="#55A868")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest feature importance\n(predicting 7-day retention)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(df: pd.DataFrame, out_path: Path):
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42).fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, model, X_te in [
        ("Logistic Regression", log_reg, X_test_s),
        ("Random Forest", rf, X_test),
    ]:
        probs = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve: predicting 7-day retention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_all(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_retention(df, out_dir / "retention_comparison.png")
    plot_gamerounds_distribution(df, out_dir / "gamerounds_distribution.png")
    plot_feature_importance(df, out_dir / "feature_importance.png")
    plot_roc_curves(df, out_dir / "roc_curve.png")


if __name__ == "__main__":
    df = pd.read_csv("data/cookie_cats_clean.csv")
    generate_all(df, Path("reports/figures"))
    print("Saved figures to reports/figures/")
