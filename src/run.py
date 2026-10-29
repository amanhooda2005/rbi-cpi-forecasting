"""Run the full study: data -> backtest -> metrics -> figures. Usage: python src/run.py"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import PROCESSED, ROOT, build_panel
from evaluate import HORIZONS, run_backtest, score, score_by_period, score_survey

RES, FIG = ROOT / "results", ROOT / "figures"
COLORS = {"Actual": "#1a1a1a", "Random walk": "#9aa0a6", "SARIMA": "#1f5fbf",
          "Ridge": "#e8710a", "Random Forest": "#188038", "Gradient Boosting": "#a142f4",
          "Average of models": "#d93025"}
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def fig_history(panel):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    p = panel.loc["2014":]
    fit_start = pd.Timestamp("2016-08-01")      # flexible inflation targeting notified Aug 2016
    band = p.loc[fit_start:].index
    ax.fill_between(band, 2, 6, color="#1f5fbf", alpha=0.08, label="RBI tolerance band 2–6% (since Aug 2016)")
    ax.plot(band, [4] * len(band), color="#1f5fbf", lw=0.8, ls="--", label="4% target")
    ax.plot(p.index, p["infl_general"], color="#1a1a1a", lw=1.8, label="Headline CPI inflation")
    ax.plot(p.index, p["infl_food"], color="#e8710a", lw=1.1, alpha=0.8, label="Food inflation (CFPI)")
    ax.axvline(pd.Timestamp("2021-01-01"), color="grey", lw=0.8)
    ax.text(pd.Timestamp("2021-02-01"), ax.get_ylim()[1] * 0.92, "test period →", color="grey")
    ax.set(title="India CPI inflation, year-on-year (%)", ylabel="%")
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "01_inflation_history.png"); plt.close(fig)


def fig_forecasts(fc):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, h in zip(axes, (1, 3)):
        g = fc[fc["horizon"] == h]
        ax.plot(g["target"], g["actual"], color=COLORS["Actual"], lw=2.2, label="Actual")
        for m in ("Random walk", "SARIMA", "Random Forest"):
            ax.plot(g["target"], g[m], color=COLORS[m], lw=1.3, label=m,
                    ls="--" if m == "Random walk" else "-")
        ax.set(title=f"{h}-month-ahead forecasts", ylabel="YoY inflation (%)")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG / "02_forecasts_vs_actual.png"); plt.close(fig)


def fig_relative_rmse(scores):
    s = scores[(scores["model"] != "Random walk") & ~scores["model"].str.endswith("+ survey")]
    models = list(dict.fromkeys(s["model"]))
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        v = s[s["model"] == m].set_index("horizon").loc[list(HORIZONS), "relative RMSE vs random walk"]
        ax.bar([x + i * w for x in range(len(HORIZONS))], v, width=w, color=COLORS[m], label=m)
    ax.axhline(1, color="black", lw=1)
    ax.set_xticks([x + w * (len(models) - 1) / 2 for x in range(len(HORIZONS))],
                  [f"{h}-month ahead" for h in HORIZONS])
    ax.set(ylabel="RMSE relative to random walk", title="Forecast accuracy (below 1 = beats the no-change benchmark)")
    ax.legend(frameon=False, fontsize=8.5, ncol=3)
    fig.tight_layout(); fig.savefig(FIG / "03_relative_rmse.png"); plt.close(fig)


def fig_importance(rf):
    imp = pd.Series(rf.last_model_.feature_importances_, index=rf.feature_names_)
    groups = {"Headline inflation lags": "infl_lag", "Food (CFPI) inflation": "food_lag",
              "Fuel & light inflation": "fuel_lag", "Miscellaneous (core-type)": "misc_lag",
              "Month-on-month momentum": "mom", "Calendar month": "tmonth"}
    g = pd.Series({k: imp[[c for c in imp.index if c.startswith(p)]].sum() for k, p in groups.items()})
    g = g.sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(g.index, g.values, color=COLORS["Random Forest"])
    ax.set(xlabel="Share of importance", title="What drives the Random Forest (6-month horizon, last origin)")
    fig.tight_layout(); fig.savefig(FIG / "04_rf_feature_importance.png"); plt.close(fig)


def fig_survey(survey_scores, panel):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), gridspec_kw={"width_ratios": [1.15, 1]})
    p = panel.loc["2014":]
    ax = axes[0]
    ax.plot(p.index, p["iesh_exp_1y"], color="#a142f4", lw=1.6, label="Households: 1-year-ahead expectation (median)")
    ax.plot(p.index, p["iesh_exp_3m"], color="#a142f4", lw=1, ls="--", label="Households: 3-month-ahead expectation")
    ax.plot(p.index, p["infl_general"], color="#1a1a1a", lw=1.6, label="Actual CPI inflation")
    ax.set(title="Household expectations vs actual inflation (%)", ylabel="%")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    models = [m for m in dict.fromkeys(survey_scores["model"])]
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        v = survey_scores[survey_scores["model"] == m].set_index("horizon").loc[list(HORIZONS), "change (%)"]
        ax.bar([x + i * w for x in range(len(HORIZONS))], v, width=w, color=COLORS[m], label=m)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks([x + w * (len(models) - 1) / 2 for x in range(len(HORIZONS))],
                  [f"{h}-month ahead" for h in HORIZONS])
    ax.set(ylabel="Change in RMSE (%)", title="Effect of adding the survey (below 0 = better)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "05_household_expectations.png"); plt.close(fig)


def main():
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True); PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    panel.to_csv(PROCESSED, float_format="%.4f")
    fc, info = run_backtest(panel)
    scores, by_period, survey_scores = score(fc), score_by_period(fc), score_survey(fc)
    survey_scores.to_csv(RES / "survey_ablation.csv", index=False, float_format="%.4f")
    fc.to_csv(RES / "forecasts.csv", index=False, float_format="%.4f")
    scores.to_csv(RES / "metrics.csv", index=False, float_format="%.4f")
    by_period.to_csv(RES / "rmse_by_period.csv", float_format="%.4f")
    meta = {"data_start": f"{panel.index.min():%Y-%m}", "data_end": f"{panel.index.max():%Y-%m}",
            "test_targets": f"{fc['target'].min():%Y-%m} to {fc['target'].max():%Y-%m}",
            "n_test_per_horizon": int((fc["horizon"] == 1).sum()),
            "sarima_order": info["sarima_order"], "sarima_seasonal_order": info["sarima_seasonal_order"]}
    (RES / "run_info.json").write_text(json.dumps(meta, indent=2, default=str))
    fig_history(panel); fig_forecasts(fc); fig_relative_rmse(scores); fig_importance(info["rf_model"])
    fig_survey(survey_scores, info["panel"])
    print(json.dumps(meta, indent=2, default=str))
    print(scores.round(3).to_string(index=False))
    print(survey_scores.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
