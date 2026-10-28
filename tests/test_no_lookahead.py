"""Checks that forecasts use only data available at the forecast origin, and that the
processed data matches MOSPI's published inflation rates. Run: python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import build_panel  # noqa: E402
from models import DirectML, Sarima, random_walk  # noqa: E402

PANEL = build_panel()
ORIGIN = pd.Timestamp("2022-06-01")


def _forecast_with_future(future_shift: float, model, h: int) -> float:
    """Forecast at ORIGIN after corrupting every month after ORIGIN."""
    p = PANEL.copy()
    p.loc[p.index > ORIGIN, :] += future_shift
    return model(p.loc[:ORIGIN], h)


def test_forecasts_ignore_future_data():
    for model in (random_walk, DirectML("Ridge"), DirectML("Random Forest"),
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
