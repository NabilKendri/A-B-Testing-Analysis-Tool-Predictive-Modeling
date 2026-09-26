"""
Loads the cleaned dataset into a SQLite database and runs the analysis
queries in sql/queries.sql.

Why SQL here, when we already have the data in pandas: in a real
experimentation pipeline, event data usually lives in a warehouse and
the first analysis pass happens in SQL, before anything is pulled into
Python for statistical testing or modeling. This module reproduces that
step and demonstrates segment-level slicing (by engagement bucket, by
percentile) that would be awkward to hand-roll in pandas but is natural
in SQL.
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd


def load_db(clean_csv: Path, schema_sql: Path, db_path: Path) -> sqlite3.Connection:
    """(Re)build a SQLite database from the cleaned CSV."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(schema_sql.read_text())

    df = pd.read_csv(clean_csv)
    df.to_sql("players", conn, if_exists="append", index=False)
    conn.commit()
    return conn


def _split_statements(sql_text: str) -> list[str]:
    """Split a .sql file into individual statements on top-level semicolons,
    skipping comment-only lines so each numbered query runs separately."""
    statements = [s.strip() for s in sql_text.split(";")]
    return [s for s in statements if s and not all(
        line.strip().startswith("--") or not line.strip()
        for line in s.splitlines()
    )]


def run_queries(conn: sqlite3.Connection, queries_sql: Path) -> list[tuple[str, pd.DataFrame]]:
    """Run every query in the .sql file, paired with its leading comment as a label."""
    text = queries_sql.read_text()
    # Split into blocks separated by blank lines before a numbered "-- N." comment
    blocks = re.split(r"\n(?=-- \d+\. )", text)
    results = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        comment_lines = [l[3:] for l in block.splitlines() if l.startswith("--")]
        label = comment_lines[0] if comment_lines else "query"
        sql = "\n".join(l for l in block.splitlines() if not l.startswith("--")).strip()
        if not sql:
            continue  # header/comment-only block, no actual query
        df = pd.read_sql_query(sql, conn)
        results.append((label, df))
    return results


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    conn = load_db(
        root / "data" / "cookie_cats_clean.csv",
        root / "sql" / "schema.sql",
        root / "data" / "cookie_cats.db",
    )
    for label, df in run_queries(conn, root / "sql" / "queries.sql"):
        print(f"\n--- {label} ---")
        print(df.to_string(index=False))
    conn.close()
