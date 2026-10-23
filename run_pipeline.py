"""End-to-end pipeline: load -> quality checks/clean -> SQL aggregate -> models -> outputs.

  python run_pipeline.py --data data/superstore.csv
  python run_pipeline.py --data data/sample_superstore.csv      # synthetic demo data
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import data_quality as dq
from src.forecasting import MODELS, compare_models, future_forecast
from src.sql_layer import monthly_sales_from_sql


def run(data_path: str, out_dir: str = "outputs", horizon: int = 6) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = dq.standardize_columns(dq.load_raw(data_path))
    clean_df, report = dq.clean(raw)
    dq.save_report(report, out)

    series = monthly_sales_from_sql(clean_df, out / "superstore.db")
    train, test, preds, info, table = compare_models(series, horizon)
    best = table.iloc[0]["model"]
    future = future_forecast(series, best, horizon)

    table.to_csv(out / "metrics.csv", index=False)
    (out / "model_details.json").write_text(json.dumps(
        {"best_model_by_rmse": best, "holdout_months": horizon,
         "details": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in info.items()}}, indent=2))

    # wide table for Power BI
    result = pd.DataFrame({"month": series.index, "actual": series.values})
    for name in MODELS:
        result[name] = result["month"].map(preds[name])
    future_rows = pd.DataFrame({"month": future.index, "future_forecast": future.values})
    result = pd.concat([result, future_rows], ignore_index=True)
    result["best_model"] = best
    result.to_csv(out / "forecast_results.csv", index=False)

    # chart
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(series.index, series.values, color="black", label="Actual")
    for name in MODELS:
        ax.plot(preds[name].index, preds[name].values, "--", label=f"{name} (hold-out)")
    ax.plot(future.index, future.values, color="tab:red", linewidth=2.5,
            label=f"Future forecast ({best})")
    ax.axvline(test.index[0], color="grey", linestyle=":")
    ax.set_title("Monthly sales: actual vs forecasts")
    ax.set_ylabel("Sales")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "forecast_chart.png", dpi=150)
    plt.close(fig)

    print("Data quality:", json.dumps(report["actions"], indent=2))
    print(f"\nHold-out = last {horizon} months. Metrics:\n{table.to_string(index=False)}")
    print(f"\nBest model by RMSE: {best}\nOutputs written to {out}/")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/superstore.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--horizon", type=int, default=6)
    args = ap.parse_args()
    if not Path(args.data).exists():
        sys.exit(f"{args.data} not found. Put the Superstore file there, or run "
                 "`python data/generate_sample_data.py` and use --data data/sample_superstore.csv")
    run(args.data, args.out, args.horizon)
