"""
Data cleaning for the Cookie Cats A/B test dataset.

Why this step matters (resume bullet: "processing and cleaning of large
datasets to ensure the integrity and accuracy of experimental results"):
raw experiment data almost always has outliers, duplicates, or invalid
rows that can silently bias a statistical test if you don't check for them.
"""

import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV exactly as collected."""
    return pd.read_csv(path)


def _gap_cutoff(values: pd.Series, top_n: int = 20) -> float:
    """
    Find a natural cutoff above the top-N sorted values by locating the
    single largest relative jump between consecutive values.

    This is more defensible here than a blind z-score or IQR rule: this
    distribution is heavy-tailed by nature (a few very engaged players
    genuinely rack up 1,000-3,000 rounds), so a fixed statistical rule
    either misses the true anomaly or wrongly discards real "whale"
    players. Looking for the one jump that dwarfs the rest isolates
    values that don't belong to the same distribution as everything else.
    """
    top = values.sort_values(ascending=False).head(top_n).to_numpy()
    gaps = top[:-1] - top[1:]
    jump_idx = gaps.argmax()
    # Only treat it as an anomaly if the jump is far bigger than the
    # typical spacing between the other top values.
    typical_gap = gaps[gaps != gaps[jump_idx]].mean() if len(gaps) > 1 else 0
    if gaps[jump_idx] > 10 * max(typical_gap, 1):
        return top[jump_idx + 1]  # cutoff = highest "normal" value
    return float("inf")  # no clear anomaly found


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean the dataset and return (clean_df, report).

    Steps:
    1. Drop duplicate userids (each player should appear once).
    2. Drop rows with nulls in key columns.
    3. Remove extreme outliers in sum_gamerounds using a gap-based rule
       (see _gap_cutoff): a single user with 49,854 rounds in 14 days is
       physically implausible and sits in a class of its own, far past
       the next-heaviest real player (2,961 rounds). Left in, it would
       inflate the mean and variance for every downstream test.
    """
    report = {"rows_in": len(df)}

    # 1. Duplicates
    n_dupes = df["userid"].duplicated().sum()
    df = df.drop_duplicates(subset="userid", keep="first")
    report["duplicates_removed"] = int(n_dupes)

    # 2. Missing values
    n_before = len(df)
    df = df.dropna(subset=["userid", "version", "sum_gamerounds", "retention_1", "retention_7"])
    report["nulls_removed"] = int(n_before - len(df))

    # 3. Outliers (gap-based cutoff on sum_gamerounds)
    cutoff = _gap_cutoff(df["sum_gamerounds"])
    outlier_mask = df["sum_gamerounds"] > cutoff
    report["outlier_cutoff"] = cutoff
    report["outliers_removed"] = int(outlier_mask.sum())
    report["outlier_values"] = df.loc[outlier_mask, "sum_gamerounds"].tolist()
    df = df.loc[~outlier_mask].copy()

    # Types
    df["version"] = df["version"].astype("category")
    df["retention_1"] = df["retention_1"].astype(bool)
    df["retention_7"] = df["retention_7"].astype(bool)

    report["rows_out"] = len(df)
    return df, report


if __name__ == "__main__":
    raw = load_raw("data/cookie_cats.csv")
    clean_df, report = clean(raw)
    clean_df.to_csv("data/cookie_cats_clean.csv", index=False)
    print(report)
