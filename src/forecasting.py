"""Baseline, SARIMA and gradient-boosting forecasters + evaluation metrics."""
from __future__ import annotations

import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

LAGS = (1, 2, 3, 12)


# ------------------------------------------------------------------ metrics
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, in percent (zero actuals are skipped)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# --------------------------------------------------------------- baseline
def seasonal_naive(train: pd.Series, horizon: int) -> pd.Series:
    """Forecast each month with the value from the same month one year earlier."""
    idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
    vals = [train.iloc[len(train) - 12 + (i % 12)] for i in range(horizon)]
    return pd.Series(vals, index=idx)


# ------------------------------------------------------------------ SARIMA
def sarima_forecast(train: pd.Series, horizon: int) -> tuple[pd.Series, dict]:
    """Pick (p,d,q)(P,D,Q,12) from a small grid by AIC, then forecast."""
    orders = [(1, 1, 1), (0, 1, 1), (1, 1, 0)]
    seasonals = [(0, 0, 0, 0), (1, 1, 0, 12), (0, 1, 1, 12)]
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order, seas in product(orders, seasonals):
            try:
                res = SARIMAX(train, order=order, seasonal_order=seas,
                              enforce_stationarity=False, enforce_invertibility=False
                              ).fit(disp=False)
            except Exception:
                continue
            if np.isfinite(res.aic) and (best is None or res.aic < best[0]):
                best = (res.aic, order, seas, res)
    if best is None:
        raise RuntimeError("No SARIMA configuration converged.")
    _, order, seas, res = best
    fc = res.forecast(horizon)
    return fc.clip(lower=0), {"order": order, "seasonal_order": seas, "aic": round(best[0], 1)}


# --------------------------------------------------- gradient boosting (ML)
def _features(history: list[float], t_index: int, month: int) -> list[float]:
    row = [history[-lag] for lag in LAGS]
    row += [np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12), t_index]
    return row


def gb_forecast(train: pd.Series, horizon: int) -> pd.Series:
    """Gradient boosting on lag + calendar features, forecasting recursively."""
    values = train.values.astype(float)
    X, y = [], []
    for t in range(max(LAGS), len(values)):
        X.append(_features(list(values[:t]), t, train.index[t].month))
        y.append(values[t])
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0
    ).fit(np.array(X), np.array(y))

    history = list(values)
    idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
    preds = []
    for i, ts in enumerate(idx):
        p = float(model.predict(np.array([_features(history, len(values) + i, ts.month)]))[0])
        p = max(p, 0.0)
        preds.append(p)
        history.append(p)
    return pd.Series(preds, index=idx)


# --------------------------------------------------------------- compare
MODELS = ("seasonal_naive", "sarima", "gradient_boosting")


def run_model(name: str, train: pd.Series, horizon: int) -> tuple[pd.Series, dict]:
    if name == "seasonal_naive":
        return seasonal_naive(train, horizon), {}
    if name == "sarima":
        return sarima_forecast(train, horizon)
    if name == "gradient_boosting":
        return gb_forecast(train, horizon), {}
    raise ValueError(name)


def compare_models(series: pd.Series, horizon: int = 6):
    """Hold out the last `horizon` months, fit every model, score with MAPE and RMSE."""
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    preds, info, rows = {}, {}, []
    for name in MODELS:
        p, meta = run_model(name, train, horizon)
        preds[name], info[name] = p, meta
        rows.append({"model": name, "MAPE_%": round(mape(test, p), 2),
                     "RMSE": round(rmse(test, p), 2)})
    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return train, test, preds, info, table


def future_forecast(series: pd.Series, best_model: str, horizon: int = 6) -> pd.Series:
    """Refit the chosen model on all history and forecast the next `horizon` months."""
    fc, _ = run_model(best_model, series, horizon)
    return fc
