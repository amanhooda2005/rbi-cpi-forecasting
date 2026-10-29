"""Forecasting models for India's headline CPI inflation (YoY %).

Every model has the same interface:
    forecast(panel_up_to_origin, h) -> predicted YoY inflation for origin + h months
Only data available at the forecast origin is used (no look-ahead).
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

SEED = 42


# ---------------------------------------------------------------- benchmarks
def random_walk(panel: pd.DataFrame, h: int) -> float:
    """No-change forecast: inflation h months ahead equals inflation today."""
    return float(panel["infl_general"].iloc[-1])


# ---------------------------------------------------------------- SARIMA
def select_sarima_order(log_cpi: pd.Series):
    """Pick (p,d,q)(P,D,Q,12) by AIC on the initial training window only."""
    best, best_aic = None, np.inf
    for p, q, P, Q in itertools.product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(log_cpi, order=(p, 1, q), seasonal_order=(P, 1, Q, 12),
                              trend="n").fit(disp=False)
            if fit.aic < best_aic:
                best, best_aic = ((p, 1, q), (P, 1, Q, 12)), fit.aic
        except Exception:
            continue
    return best


class Sarima:
    """Seasonal ARIMA on log CPI; inflation is derived from the forecast index path."""

    def __init__(self, order, seasonal_order, max_h: int = 12):
        self.order, self.seasonal_order, self.max_h = order, seasonal_order, max_h
        self._cache = {}        # one fit per origin, reused across horizons

    def _path(self, panel: pd.DataFrame) -> pd.Series:
        origin = panel.index[-1]
        if origin not in self._cache:
            log_cpi = np.log(panel["cpi_general"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(log_cpi, order=self.order, seasonal_order=self.seasonal_order,
                              trend="n").fit(disp=False)
            self._cache[origin] = np.exp(fit.forecast(steps=self.max_h))
        return self._cache[origin]

    def __call__(self, panel: pd.DataFrame, h: int) -> float:
        path = self._path(panel).iloc[:h]
        target_date = panel.index[-1] + pd.DateOffset(months=h)
        base_date = target_date - pd.DateOffset(months=12)
        base = panel["cpi_general"].get(base_date, np.nan)
        if np.isnan(base):                       # only happens if h > 12
            base = path[base_date]
        return float((path.iloc[-1] / base - 1) * 100)


# ---------------------------------------------------------------- ML (direct)
SURVEY_COLS = ["iesh_exp_current", "iesh_exp_3m", "iesh_exp_1y", "iesh_d_exp_1y", "iesh_d_exp_3m"]


def make_features(panel: pd.DataFrame, survey: bool = False) -> pd.DataFrame:
    """Predictors observed at each month t. With survey=True, adds RBI household inflation
    expectations as known at t (see survey.py for the publication lag)."""
    f = pd.DataFrame(index=panel.index)
    if survey:
        for c in SURVEY_COLS:
            f[c] = panel[c]
    for lag in range(0, 12):
        f[f"infl_lag{lag}"] = panel["infl_general"].shift(lag)
    for lag in range(0, 3):
        f[f"food_lag{lag}"] = panel["infl_food"].shift(lag)
        f[f"fuel_lag{lag}"] = panel["infl_fuel"].shift(lag)
        f[f"misc_lag{lag}"] = panel["infl_misc"].shift(lag)   # proxy for core-type items
    for lag in range(0, 3):
        f[f"mom_lag{lag}"] = panel["mom_general"].shift(lag)
        f[f"mom_food_lag{lag}"] = panel["mom_food"].shift(lag)
    return f


def _direct_xy(panel: pd.DataFrame, h: int, survey: bool = False):
    """Train on (features at t) -> (inflation at t+h), plus target-month seasonality."""
    X = make_features(panel, survey=survey)
    target_month = (panel.index + pd.DateOffset(months=h)).month
    for m in range(2, 13):
        X[f"tmonth_{m}"] = (target_month == m).astype(float)
    y = panel["infl_general"].shift(-h)
    train = X.notna().all(axis=1) & y.notna()
    return X[train], y[train], X.iloc[[-1]]


class DirectML:
    """Direct multi-step regression: a separate model per horizon, refit at every origin."""

    def __init__(self, name: str, survey: bool = False):
        self.name, self.survey = name, survey

    def _model(self):
        if self.name == "Ridge":
            return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
        if self.name == "Random Forest":
            return RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                         max_features=0.5, random_state=SEED, n_jobs=-1)
        if self.name == "Gradient Boosting":
            return GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=2,
                                             subsample=0.8, random_state=SEED)
        raise ValueError(self.name)

    def __call__(self, panel: pd.DataFrame, h: int) -> float:
        X, y, x_now = _direct_xy(panel, h, survey=self.survey)
        model = self._model().fit(X, y)
        self.last_model_, self.feature_names_ = model, list(X.columns)
        return float(model.predict(x_now)[0])
