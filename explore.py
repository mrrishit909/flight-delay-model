"""Step 2: look before modeling: how the disruption rate moves with hour, the plane's leg of the day, and month.

    ./venv/bin/python explore.py

charts/01_hour.png, 02_leg.png, 03_month.png and results/explore.json (numbers the report quotes).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).parent
INK, DIM, GRID, BG, SERIES = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5"


def style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 14,
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 14, "lines.linewidth": 2,
    })


def load():
    return pd.read_csv(HERE / "data" / "flights.csv.gz", dtype={"leg": "Int64", "turn_min": "Int64"})


def bars(rates, labels, title, xlabel, path, overall):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(labels, rates, color=SERIES, width=0.72)
    ax.axhline(overall, color=DIM, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1.01, overall, f"all\n{overall:.0%}", color=DIM, va="center", ha="left", fontsize=10,
            transform=blended_transform_factory(ax.transAxes, ax.transData))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    style()
    d = load()
    (HERE / "charts").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    overall = d["disrupted"].mean()

    hour = d[d["dep_hour"] >= 5].groupby("dep_hour")["disrupted"].mean()      # hour 0 has a single flight
    bars(hour.values, [str(h) for h in hour.index], "Share of flights disrupted, by scheduled departure hour",
         "scheduled departure hour (local)", HERE / "charts" / "01_hour.png", overall)

    leg = d.groupby(d["leg"].clip(upper=6))["disrupted"].mean()
    bars(leg.values, ["1st", "2nd", "3rd", "4th", "5th", "6th+"], "Share disrupted, by the plane's flight of the day",
         "which flight of the day this is for the aircraft", HERE / "charts" / "02_leg.png", overall)

    month = d.groupby(d["date"].str[:7])["disrupted"].mean()
    labels = pd.to_datetime(month.index).strftime("%b\n%y")
    bars(month.values, list(labels), "Share disrupted, by month (the model is tested on Jun-Jul 2026)",
         "", HERE / "charts" / "03_month.png", overall)

    facts = {
        "flights": len(d), "disrupted_pct": round(100 * overall, 1),
        "hour_low": {"hour": int(hour.idxmin()), "pct": round(100 * hour.min(), 1)},
        "hour_high": {"hour": int(hour.idxmax()), "pct": round(100 * hour.max(), 1)},
        "leg_pct": {str(k): round(100 * v, 1) for k, v in leg.items()},
        "month_pct": {k: round(100 * v, 1) for k, v in month.items()},
        "tight_turn_pct": round(100 * d.loc[d["turn_min"] <= 30, "disrupted"].mean(), 1),
        "normal_turn_pct": round(100 * d.loc[d["turn_min"].between(31, 180), "disrupted"].mean(), 1),
    }
    (HERE / "results" / "explore.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
