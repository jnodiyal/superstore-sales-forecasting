"""Store cleaned orders in SQLite and aggregate monthly sales with SQL."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

MONTHLY_SQL = """
SELECT strftime('%Y-%m-01', order_date) AS month,
       ROUND(SUM(sales), 2)             AS sales,
       COUNT(*)                         AS order_lines
FROM orders
GROUP BY 1
ORDER BY 1;
"""


def monthly_sales_from_sql(clean_df: pd.DataFrame, db_path: str | Path) -> pd.Series:
    """Load cleaned orders into SQLite, run the aggregation query, return a monthly series."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    df = clean_df.copy()
    df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
    with sqlite3.connect(db_path) as conn:
        df.to_sql("orders", conn, if_exists="replace", index=False)
        monthly = pd.read_sql_query(MONTHLY_SQL, conn, parse_dates=["month"])
    series = monthly.set_index("month")["sales"].astype(float)
    series.index = series.index.to_period("M").to_timestamp()
    return series.asfreq("MS")
