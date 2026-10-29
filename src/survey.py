"""RBI Inflation Expectations Survey of Households (IESH): median expectations by survey round,
converted to a monthly series of what was *publicly known* at each forecast origin."""
from pathlib import Path

import openpyxl
import pandas as pd

from data import ROOT

IESH_FILE = ROOT / "data" / "raw" / "IESH05122025.xlsx"

# A round's results are published with the RBI monetary policy statement, usually in the first
# week or two of the following month. CPI for month t is released around the 12th of month t+1,
# so a round surveyed in month m is normally public before a forecast made with CPI data for m.
# To stay safe against late releases, we assume a round is usable only from month m + 1.
PUBLICATION_LAG_MONTHS = 1


def load_rounds(path: Path = IESH_FILE) -> pd.DataFrame:
    """One row per survey round: median current perception, 3-month and 1-year expectations."""
    ws = openpyxl.load_workbook(path, data_only=True, read_only=True)["T2 Household IE"]
    rows = []
    for r in ws.iter_rows(min_row=5, values_only=True):
        if r[2] is None or not hasattr(r[2], "year"):
            continue
        rows.append({"round": str(r[1]), "survey_month": pd.Timestamp(r[2].year, r[2].month, 1),
                     "exp_current": r[4], "exp_3m": r[7], "exp_1y": r[10]})
    df = pd.DataFrame(rows).dropna(subset=["exp_3m", "exp_1y"])
    df = df.sort_values("survey_month").drop_duplicates("survey_month", keep="last")
    return df.reset_index(drop=True)


def monthly_known(index: pd.DatetimeIndex, path: Path = IESH_FILE) -> pd.DataFrame:
    """For each month t: the latest survey results public at t, and the change from the round before."""
    rounds = load_rounds(path)
    rounds["available_from"] = rounds["survey_month"] + pd.DateOffset(months=PUBLICATION_LAG_MONTHS)
    rounds["d_exp_1y"] = rounds["exp_1y"].diff()
    rounds["d_exp_3m"] = rounds["exp_3m"].diff()
    cols = ["exp_current", "exp_3m", "exp_1y", "d_exp_1y", "d_exp_3m"]
    known = (rounds.set_index("available_from")[cols]
             .reindex(rounds["available_from"].tolist() + list(index)).sort_index())
    known = known[~known.index.duplicated(keep="first")].ffill()
    return known.reindex(index).add_prefix("iesh_")
