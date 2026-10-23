"""Load the Superstore data, run data-quality checks, and clean it."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ALIASES = {
    "order_date": ["order date", "order_date", "orderdate"],
    "sales": ["sales", "sales amount", "revenue"],
    "order_id": ["order id", "order_id", "orderid"],
    "category": ["category"],
    "region": ["region"],
}


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:  # the public Superstore CSVs are often latin-1
        return pd.read_csv(path, encoding="latin-1")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to order_date / sales / order_id / category / region."""
    lookup = {c.strip().lower(): c for c in df.columns}
    rename = {}
    for canon, options in ALIASES.items():
        for opt in options:
            if opt in lookup:
                rename[lookup[opt]] = canon
                break
    out = df.rename(columns=rename)
    # Row ID is a unique counter; drop it so true duplicate orders can be detected
    out = out.drop(columns=[c for c in out.columns if c.strip().lower() == "row id"])
    missing = [c for c in ("order_date", "sales") if c not in out.columns]
    if missing:
        raise ValueError(f"Required column(s) not found: {missing}. Found: {list(df.columns)}")
    return out


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse dates, choosing day-first or month-first by whichever fails less."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        month_first = pd.to_datetime(series, errors="coerce", dayfirst=False)
        day_first = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return day_first if day_first.isna().sum() < month_first.isna().sum() else month_first


def run_quality_checks(df: pd.DataFrame) -> dict:
    """Profile the raw data. Returns a JSON-serialisable report."""
    dates = _parse_dates(df["order_date"])
    sales = pd.to_numeric(df["sales"], errors="coerce")
    q1, q3 = sales.quantile([0.25, 0.75])
    upper = q3 + 3 * (q3 - q1)
    return {
        "rows": int(len(df)),
        "missing_values": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "unparseable_dates": int(dates.isna().sum()),
        "non_numeric_sales": int(sales.isna().sum() - df["sales"].isna().sum()),
        "non_positive_sales": int((sales <= 0).sum()),
        "future_dates": int((dates > pd.Timestamp.today()).sum()),
        "sales_outliers_iqr3": int((sales > upper).sum()),
        "date_min": str(dates.min().date()) if dates.notna().any() else None,
        "date_max": str(dates.max().date()) if dates.notna().any() else None,
    }


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply cleaning rules and return (clean_df, report with before/after counts)."""
    report = {"before": run_quality_checks(df), "actions": {}}
    out = df.copy()
    out["order_date"] = _parse_dates(out["order_date"])
    out["sales"] = pd.to_numeric(out["sales"], errors="coerce")

    n = len(out)
    out = out.drop_duplicates()
    report["actions"]["dropped_duplicate_rows"] = n - len(out)

    n = len(out)
    out = out.dropna(subset=["order_date", "sales"])
    report["actions"]["dropped_missing_date_or_sales"] = n - len(out)

    n = len(out)
    out = out[out["sales"] > 0]
    report["actions"]["dropped_non_positive_sales"] = n - len(out)

    n = len(out)
    out = out[out["order_date"] <= pd.Timestamp.today()]
    report["actions"]["dropped_future_dates"] = n - len(out)

    out = out.sort_values("order_date").reset_index(drop=True)
    report["actions"]["note"] = "Outliers are flagged but kept (large orders are real sales)."

    months = out["order_date"].dt.to_period("M").nunique()
    report["months_of_history"] = int(months)
    if months < 24:
        raise ValueError(f"Only {months} months of data after cleaning; need at least 24 to forecast.")
    report["after"] = run_quality_checks(out)
    return out, report


def save_report(report: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_quality_report.json").write_text(json.dumps(report, indent=2))
    b, a = report["before"], report["after"]
    lines = [
        "# Data quality report", "",
        f"- Rows before / after cleaning: **{b['rows']} / {a['rows']}**",
        f"- Missing values (before): {b['missing_values'] or 'none'}",
        f"- Duplicate rows (before): {b['duplicate_rows']}",
        f"- Unparseable dates (before): {b['unparseable_dates']}",
        f"- Non-positive sales (before): {b['non_positive_sales']}",
        f"- Sales outliers flagged (kept): {b['sales_outliers_iqr3']}",
        f"- Date range after cleaning: {a['date_min']} to {a['date_max']} "
        f"({report['months_of_history']} months)", "",
        "## Cleaning actions",
    ]
    lines += [f"- {k}: {v}" for k, v in report["actions"].items()]
    (out_dir / "data_quality_report.md").write_text("\n".join(lines) + "\n")
