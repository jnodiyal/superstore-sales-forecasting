import numpy as np
import pandas as pd
import pytest

from data.generate_sample_data import generate
from run_pipeline import run
from src import data_quality as dq
from src.forecasting import mape, rmse, seasonal_naive


def test_metrics_known_values():
    assert rmse([100, 200], [110, 190]) == pytest.approx(10.0)
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert mape([0, 100], [5, 110]) == pytest.approx(10.0)  # zero actuals skipped


def test_seasonal_naive_repeats_last_year():
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    s = pd.Series(np.arange(24, dtype=float), index=idx)
    fc = seasonal_naive(s, 3)
    assert list(fc.values) == [12.0, 13.0, 14.0]


def test_cleaning_removes_problems(tmp_path):
    path = tmp_path / "s.csv"
    raw = generate(str(path))
    df = dq.standardize_columns(dq.load_raw(path))
    clean_df, report = dq.clean(df)
    assert report["before"]["duplicate_rows"] > 0
    assert report["after"]["duplicate_rows"] == 0
    assert clean_df["sales"].gt(0).all()
    assert clean_df["order_date"].notna().all()


def test_missing_required_column_raises():
    with pytest.raises(ValueError):
        dq.standardize_columns(pd.DataFrame({"foo": [1]}))


def test_pipeline_end_to_end(tmp_path):
    data = tmp_path / "s.csv"
    generate(str(data))
    table = run(str(data), str(tmp_path / "out"), horizon=6)
    assert set(table["model"]) == {"seasonal_naive", "sarima", "gradient_boosting"}
    assert (tmp_path / "out" / "forecast_results.csv").exists()
    assert (tmp_path / "out" / "forecast_chart.png").exists()
    assert (tmp_path / "out" / "data_quality_report.md").exists()
