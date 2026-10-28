"""Pseudo out-of-sample evaluation with an expanding window (rolling origin)."""
import numpy as np
import pandas as pd
from scipy import stats

from data import build_panel
from models import DirectML, Sarima, random_walk, select_sarima_order

HORIZONS = (1, 3, 6)
TEST_START, TEST_END = "2021-01-01", None          # targets from Jan 2021 to the last month
SARIMA_SELECTION_END = "2020-12-01"                 # order chosen on pre-test data only
PERIODS = {"2021-22 (inflation surge)": ("2021-01-01", "2022-12-01"),
           "2023-25 (disinflation)": ("2023-01-01", None)}


def run_backtest(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    order, seasonal = select_sarima_order(np.log(panel.loc[:SARIMA_SELECTION_END, "cpi_general"]))
    models = {
        "Random walk": random_walk,
        "SARIMA": Sarima(order, seasonal, max_h=max(HORIZONS)),
        "Ridge": DirectML("Ridge"),
        "Random Forest": DirectML("Random Forest"),
        "Gradient Boosting": DirectML("Gradient Boosting"),
    }
    targets = panel.loc[TEST_START:TEST_END].index
    rows = []
    for h in HORIZONS:
        for target in targets:
            origin = target - pd.DateOffset(months=h)
            hist = panel.loc[:origin]
            row = {"horizon": h, "origin": origin, "target": target,
                   "actual": panel.at[target, "infl_general"]}
            for name, model in models.items():
                row[name] = model(hist, h)
            rows.append(row)
    fc = pd.DataFrame(rows)
    # Pre-specified forecast combination: equal-weight mean of the four non-naive models.
    fc["Average of models"] = fc[["SARIMA", "Ridge", "Random Forest", "Gradient Boosting"]].mean(axis=1)
    info = {"sarima_order": order, "sarima_seasonal_order": seasonal,
            "rf_model": models["Random Forest"]}
    return fc, info


def model_names(fc: pd.DataFrame) -> list[str]:
    return [c for c in fc.columns if c not in ("horizon", "origin", "target", "actual")]


def dm_test(e_model: np.ndarray, e_bench: np.ndarray, h: int) -> float:
    """Diebold-Mariano test with Harvey et al. (1997) correction; squared-error loss.
    Returns the two-sided p-value. Negative mean loss differential favours the model."""
    d = e_model ** 2 - e_bench ** 2
    n = len(d)
    d_bar = d.mean()
    gamma = [np.mean((d[k:] - d_bar) * (d[:n - k] - d_bar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan
    dm = d_bar / np.sqrt(var)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(2 * stats.t.sf(abs(dm), df=n - 1))


def score(fc: pd.DataFrame) -> pd.DataFrame:
    out = []
    for h, g in fc.groupby("horizon"):
        rw_err = (g["Random walk"] - g["actual"]).to_numpy()
        for name in model_names(fc):
            err = (g[name] - g["actual"]).to_numpy()
            rmse = np.sqrt(np.mean(err ** 2))
            out.append({
                "horizon": h, "model": name, "n": len(err),
                "RMSE": rmse, "MAE": np.mean(np.abs(err)),
                "relative RMSE vs random walk": rmse / np.sqrt(np.mean(rw_err ** 2)),
                "DM p-value vs random walk": np.nan if name == "Random walk" else dm_test(err, rw_err, h),
            })
    return pd.DataFrame(out)


def score_by_period(fc: pd.DataFrame) -> pd.DataFrame:
    out = []
    for label, (a, b) in PERIODS.items():
        sub = fc[(fc["target"] >= a) & ((b is None) | (fc["target"] <= (b or fc["target"].max())))]
        for h, g in sub.groupby("horizon"):
            for name in model_names(fc):
                out.append({"period": label, "horizon": h, "model": name,
                            "RMSE": np.sqrt(np.mean((g[name] - g["actual"]) ** 2))})
    return pd.DataFrame(out).pivot_table(index=["period", "model"], columns="horizon", values="RMSE")


if __name__ == "__main__":
    panel = build_panel()
    fc, info = run_backtest(panel)
    print(info["sarima_order"], info["sarima_seasonal_order"])
    print(score(fc).round(3).to_string())
    print(score_by_period(fc).round(3))
