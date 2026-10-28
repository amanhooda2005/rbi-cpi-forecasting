"""Load and prepare MOSPI CPI (base 2012) data for All-India, Combined sector."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mospi_cpi_base2012_all_india_groups.csv"
PROCESSED = ROOT / "data" / "processed" / "cpi_all_india_combined_monthly.csv"

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}

GROUPS = {
    "General": "cpi_general",
    "Consumer Food Price": "cpi_food",          # CFPI, RBI's food inflation measure
    "Fuel and Light": "cpi_fuel",
    "Miscellaneous": "cpi_misc",
}


def load_raw(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["state"] == "All India") & (df["sector"] == "Combined") & df["subgroup"].isna()]
    df = df[df["group"].isin(GROUPS)]
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"].map(MONTHS), day=1))
    return df


def build_panel(path: Path = RAW) -> pd.DataFrame:
    """Monthly panel: CPI indices, official headline inflation, and derived YoY / MoM rates."""
    df = load_raw(path)
    idx = df.pivot_table(index="date", columns="group", values="index").rename(columns=GROUPS)
    official = (df[df["group"] == "General"].set_index("date")["inflation"]
                .rename("inflation_official"))
    panel = idx.join(official).sort_index().asfreq("MS")

    # Sanity checks: no gaps, no duplicates.
    if panel[list(GROUPS.values())].isna().any().any():
        raise ValueError("Missing index values in the CPI panel")

    for col in GROUPS.values():
        name = col.replace("cpi_", "")
        panel[f"infl_{name}"] = (panel[col] / panel[col].shift(12) - 1) * 100   # YoY %
        panel[f"mom_{name}"] = np.log(panel[col]).diff() * 100                  # MoM log %

    # Recomputed headline inflation must match MOSPI's published rate (rounded to 2 dp).
    gap = (panel["infl_general"] - panel["inflation_official"]).abs().max()
    if gap > 0.01:
        raise ValueError(f"Recomputed inflation differs from MOSPI by {gap:.3f} pp")
    return panel


if __name__ == "__main__":
    p = build_panel()
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    p.to_csv(PROCESSED, float_format="%.4f")
    print(f"Saved {len(p)} months ({p.index.min():%b %Y} to {p.index.max():%b %Y}) -> {PROCESSED}")
