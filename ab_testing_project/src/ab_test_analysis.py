"""
Statistical analysis of the Cookie Cats A/B test.

Question: does moving the first gate from level 30 (control) to level 40
(treatment) change player engagement?

We test two different kinds of metric, so we need two different tests:
- retention_1 / retention_7 are proportions (came back: yes/no)
  -> two-proportion z-test (or equivalently chi-square test of independence)
- sum_gamerounds is a skewed count (most players play few rounds, a few
  play thousands) -> Mann-Whitney U test, which compares distributions
  without assuming normality (a t-test's assumption would be violated here)
"""

import numpy as np
import pandas as pd
from scipy import stats


def proportion_test(df: pd.DataFrame, metric: str, group_col: str = "version"):
    """
    Two-proportion z-test for a boolean metric (e.g. retention_1).
    Returns a dict with each group's rate, the z-statistic, p-value,
    and a 95% confidence interval for the difference in proportions.
    """
    groups = df[group_col].unique()
    assert len(groups) == 2, "Expected exactly two groups"
    g1, g2 = sorted(groups)

    n1 = df[df[group_col] == g1].shape[0]
    n2 = df[df[group_col] == g2].shape[0]
    x1 = df[df[group_col] == g1][metric].sum()
    x2 = df[df[group_col] == g2][metric].sum()

    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # 95% CI for the difference (unpooled SE, standard approach for a CI)
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    ci_low, ci_high = diff - 1.96 * se_diff, diff + 1.96 * se_diff

    return {
        "metric": metric,
        f"{g1}_rate": p1, f"{g1}_n": n1,
        f"{g2}_rate": p2, f"{g2}_n": n2,
        "diff (g1-g2)": diff,
        "z_stat": z,
        "p_value": p_value,
        "95%_CI": (ci_low, ci_high),
        "significant_at_0.05": p_value < 0.05,
    }


def mann_whitney_test(df: pd.DataFrame, metric: str, group_col: str = "version"):
    """
    Mann-Whitney U test comparing distributions of a skewed numeric metric
    (e.g. sum_gamerounds) between the two groups.
    """
    groups = df[group_col].unique()
    g1, g2 = sorted(groups)
    x1 = df[df[group_col] == g1][metric]
    x2 = df[df[group_col] == g2][metric]

    u_stat, p_value = stats.mannwhitneyu(x1, x2, alternative="two-sided")

    return {
        "metric": metric,
        f"{g1}_median": x1.median(), f"{g1}_mean": x1.mean(),
        f"{g2}_median": x2.median(), f"{g2}_mean": x2.mean(),
        "u_stat": u_stat,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }


def run_all_tests(df: pd.DataFrame) -> dict:
    return {
        "retention_1": proportion_test(df, "retention_1"),
        "retention_7": proportion_test(df, "retention_7"),
        "sum_gamerounds": mann_whitney_test(df, "sum_gamerounds"),
    }


if __name__ == "__main__":
    df = pd.read_csv("data/cookie_cats_clean.csv")
    results = run_all_tests(df)
    for name, res in results.items():
        print(f"\n--- {name} ---")
        for k, v in res.items():
            print(f"  {k}: {v}")
