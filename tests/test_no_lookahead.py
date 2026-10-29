"""Checks that forecasts use only data available at the forecast origin, and that the
processed data matches MOSPI's published inflation rates. Run: python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import build_panel  # noqa: E402
from models import DirectML, Sarima, random_walk  # noqa: E402
from survey import load_rounds, monthly_known  # noqa: E402

PANEL = build_panel().pipe(lambda p: p.join(monthly_known(p.index)))
ORIGIN = pd.Timestamp("2022-06-01")


def _forecast_with_future(future_shift: float, model, h: int) -> float:
    """Forecast at ORIGIN after corrupting every month after ORIGIN."""
    p = PANEL.copy()
    p.loc[p.index > ORIGIN, :] += future_shift
    return model(p.loc[:ORIGIN], h)


def test_forecasts_ignore_future_data():
    for model in (random_walk, DirectML("Ridge"), DirectML("Random Forest"),
                  DirectML("Ridge", survey=True), DirectML("Random Forest", survey=True),
                  Sarima((0, 1, 1), (0, 1, 1, 12))):
        for h in (1, 3, 6):
            clean = _forecast_with_future(0.0, model, h)
            if isinstance(model, Sarima):
                model._cache.clear()
            corrupted = _forecast_with_future(50.0, model, h)
            if isinstance(model, Sarima):
                model._cache.clear()
            assert np.isclose(clean, corrupted), (model, h)


def test_inflation_matches_mospi():
    gap = (PANEL["infl_general"] - PANEL["inflation_official"]).abs().max()
    assert gap < 0.01
    # Spot checks against MOSPI press releases (CPI Combined, base 2012).
    for month, rate in {"2024-10-01": 6.21, "2024-12-01": 5.22, "2025-01-01": 4.31}.items():
        assert round(PANEL.at[pd.Timestamp(month), "infl_general"], 2) == rate


def test_panel_is_complete_monthly():
    assert PANEL.index.is_monotonic_increasing
    assert len(pd.date_range(PANEL.index[0], PANEL.index[-1], freq="MS")) == len(PANEL)


def test_survey_uses_only_published_rounds():
    """A round surveyed in month m must not appear in the data before month m + 1."""
    rounds = load_rounds().set_index("survey_month")
    known = monthly_known(PANEL.index)
    may_2025 = rounds.loc[pd.Timestamp("2025-05-01")]
    # In Dec 2024 the latest public round is Nov 2024; in Nov 2024 it is still Sep 2024.
    assert not np.isclose(known.loc["2024-12-01", "iesh_exp_1y"], may_2025["exp_1y"])
    nov_2024 = rounds.loc[pd.Timestamp("2024-11-01")]
    assert np.isclose(known.loc["2024-12-01", "iesh_exp_1y"], nov_2024["exp_1y"])
    assert np.isclose(known.loc["2024-11-01", "iesh_exp_1y"],
                      rounds.loc[pd.Timestamp("2024-09-01"), "exp_1y"])


def test_survey_matches_published_release():
    """May 2025 round medians as printed in RBI's June 2025 IESH release (Table 3)."""
    r = load_rounds().set_index("survey_month").loc[pd.Timestamp("2025-05-01")]
    assert round(r["exp_current"], 1) == 7.7
    assert round(r["exp_3m"], 1) == 8.9
    assert round(r["exp_1y"], 1) == 9.5
