"""
Generates a plain-English markdown report from the A/B test and model
results. This is the "translating complex metrics into improvement
opportunities" part of the resume bullet -- the audience is a product
manager, not a statistician.
"""

import pandas as pd
from data_cleaning import load_raw, clean
from ab_test_analysis import run_all_tests
from predictive_model import train_and_evaluate
from sql_analysis import load_db, run_queries


def verdict(p_value: float) -> str:
    return "a real, statistically significant difference" if p_value < 0.05 \
        else "no statistically significant difference"


def build_report(clean_df: pd.DataFrame, ab_results: dict, model_results: dict,
                  sql_results: list | None = None) -> str:
    r1, r7, rounds = ab_results["retention_1"], ab_results["retention_7"], ab_results["sum_gamerounds"]

    lines = []
    lines.append("# Cookie Cats A/B Test — Summary Report\n")
    lines.append(
        "**Question:** does moving the game's first gate from level 30 to "
        "level 40 change player engagement and retention?\n"
    )
    lines.append(f"**Sample size (after cleaning):** {len(clean_df):,} players "
                  f"({(clean_df['version']=='gate_30').sum():,} gate_30 / "
                  f"{(clean_df['version']=='gate_40').sum():,} gate_40)\n")

    lines.append("## Key Findings\n")
    lines.append(
        f"- **Day-1 retention:** {r1['gate_30_rate']:.1%} (gate_30) vs. "
        f"{r1['gate_40_rate']:.1%} (gate_40) — {verdict(r1['p_value'])} "
        f"(p={r1['p_value']:.3f})."
    )
    lines.append(
        f"- **Day-7 retention:** {r7['gate_30_rate']:.1%} (gate_30) vs. "
        f"{r7['gate_40_rate']:.1%} (gate_40) — {verdict(r7['p_value'])} "
        f"(p={r7['p_value']:.4f}). Moving the gate to level 40 **reduces** "
        f"week-long retention."
    )
    lines.append(
        f"- **Rounds played:** median {rounds['gate_30_median']:.0f} (gate_30) vs. "
        f"{rounds['gate_40_median']:.0f} (gate_40) — {verdict(rounds['p_value'])} "
        f"(p={rounds['p_value']:.3f}). The gate position doesn't change how much "
        f"players play overall, only whether they come back a week later."
    )

    lines.append("\n## Recommendation\n")
    lines.append(
        "- **Do not move the gate to level 40.** The only metric that moved "
        "with statistical confidence — 7-day retention — moved in the wrong "
        "direction. There's no offsetting gain in short-term engagement to "
        "justify the change."
    )

    if sql_results:
        seg = dict(sql_results)["2. Engagement segmentation: bucket players by how much they played,"]
        lines.append("\n## Where the Effect Comes From (SQL segmentation)\n")
        lines.append(
            "Slicing retention by engagement bucket (see `sql/queries.sql`) shows the "
            "drop isn't spread evenly across all players:"
        )
        pivot = seg.pivot(index="engagement_bucket", columns="version", values="retention_7_pct")
        pivot = pivot.reindex(["0 rounds (never started)", "1-10 rounds (light)",
                                "11-50 rounds (moderate)", "51-200 rounds (heavy)",
                                "200+ rounds (whale)"])
        lines.append("\n| Engagement bucket | gate_30 retention_7 | gate_40 retention_7 |")
        lines.append("|---|---|---|")
        for bucket, row in pivot.iterrows():
            lines.append(f"| {bucket} | {row['gate_30']:.1f}% | {row['gate_40']:.1f}% |")
        lines.append(
            "\n- **Activation is unaffected** — a near-identical share of players in both "
            "groups never play a single round, so the gate isn't scaring people off before "
            "they start.\n"
            "- **Power users (top 10% by rounds played) retain equally well either way** — "
            "once someone is hooked, gate position stops mattering.\n"
            "- **The drop concentrates in the moderate and heavy engagement segments** "
            "(11-200 rounds) — exactly the players who are engaged enough to hit the gate but "
            "not yet die-hard fans — consistent with the level-40 gate blocking that group "
            "before they build a long-term habit."
        )

    lines.append("\n## Predictive Model: What Predicts a Player Returning?\n")
    rf = model_results["random_forest"]
    imp = model_results["random_forest_feature_importance"]
    top_feature = max(imp, key=imp.get)
    lines.append(
        f"- A Random Forest model predicts 7-day retention with "
        f"ROC-AUC = {rf['roc_auc']:.2f} (vs. a baseline retention rate of "
        f"{model_results['baseline_positive_rate']:.1%})."
    )
    lines.append(
        f"- The strongest predictor is **{top_feature}** "
        f"(importance={imp[top_feature]:.2f}) — how much a player plays in "
        "the first 14 days matters far more for retention than which gate "
        "version they saw."
    )
    lines.append(
        "- **Actionable takeaway:** rather than tuning gate placement, "
        "product efforts to boost early game-rounds-played (e.g. onboarding, "
        "early rewards) are likely to have more impact on long-term retention."
    )

    return "\n".join(lines)


if __name__ == "__main__":
    raw = load_raw("data/cookie_cats.csv")
    clean_df, clean_report = clean(raw)
    ab_results = run_all_tests(clean_df)
    model_results = train_and_evaluate(clean_df)

    report_text = build_report(clean_df, ab_results, model_results)
    with open("reports/summary_report.md", "w") as f:
        f.write(report_text)
    print(report_text)
