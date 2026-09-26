"""
Run the full A/B testing pipeline: clean -> test -> model -> report.

Usage: python3 main.py   (run from the project root)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_cleaning import load_raw, clean
from ab_test_analysis import run_all_tests
from predictive_model import train_and_evaluate
from sql_analysis import load_db, run_queries
from plots import generate_all
from report import build_report


def main():
    root = Path(__file__).parent
    raw = load_raw(root / "data" / "cookie_cats.csv")

    print("Cleaning data...")
    clean_df, clean_report = clean(raw)
    clean_df.to_csv(root / "data" / "cookie_cats_clean.csv", index=False)
    print(f"  {clean_report}")

    print("\nRunning statistical tests...")
    ab_results = run_all_tests(clean_df)
    for name, res in ab_results.items():
        sig = "SIGNIFICANT" if res["significant_at_0.05"] else "not significant"
        print(f"  {name}: p={res['p_value']:.4f} ({sig})")

    print("\nTraining predictive model...")
    model_results = train_and_evaluate(clean_df)
    print(f"  Random Forest ROC-AUC: {model_results['random_forest']['roc_auc']:.3f}")

    print("\nLoading into SQLite and running SQL analysis...")
    conn = load_db(
        root / "data" / "cookie_cats_clean.csv",
        root / "sql" / "schema.sql",
        root / "data" / "cookie_cats.db",
    )
    sql_results = run_queries(conn, root / "sql" / "queries.sql")
    conn.close()
    for label, sql_df in sql_results:
        print(f"  {label}")

    print("\nGenerating figures...")
    generate_all(clean_df, root / "reports" / "figures")

    print("\nWriting report...")
    report_text = build_report(clean_df, ab_results, model_results, sql_results)
    report_path = root / "reports" / "summary_report.md"
    report_path.write_text(report_text)
    print(f"  Saved to {report_path}")


if __name__ == "__main__":
    main()
