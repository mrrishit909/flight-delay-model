"""Step 4: what the test months say about the models: ranking, calibration by month, and what drives the risk.

    ./venv/bin/python evaluate.py

charts/04_top10.png, 05_deciles.png, 06_importance.png and results/evaluation.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.inspection import permutation_importance

from explore import BG, DIM, INK, SERIES, style
from train import FEATURES, TARGET, gradient_boosting, load, split

HERE = Path(__file__).parent
PREDICTED = "#d95926"
GB = "Gradient boosting"
NAMES = {"dep_hour": "departure hour", "arr_hour": "arrival hour", "dow": "day of week", "leg": "plane's leg of the day",
         "turn_min": "scheduled turn time", "distance": "distance", "sched_min": "scheduled flight time",
         "carrier": "airline", "origin": "origin airport", "dest": "destination airport"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / "charts" / name, dpi=150)
    plt.close(fig)


def main():
    style()
    scores = pd.read_csv(HERE / "results" / "scores.csv")
    test_scores = scores[scores["period"] == "test"].set_index("model")
    pred = pd.read_csv(HERE / "results" / "test_predictions.csv.gz")
    y, p = pred[TARGET].values, pred[GB].values
    overall = y.mean()

    # 04: how good is each method's "riskiest 10%"?
    fig, ax = plt.subplots(figsize=(9, 3.8))
    names = list(test_scores.index)
    vals = test_scores["top10_rate"].values
    ax.barh(names[::-1], vals[::-1], color=SERIES, height=0.6)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", color=INK, fontsize=10)
    ax.axvline(overall, color=DIM, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(overall, -0.55, f" all test flights {overall:.0%}", color=DIM, fontsize=10, va="top")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-1.1, len(names) - 0.5)
    ax.set_title("Each method's riskiest 10% of flights: share actually disrupted")
    ax.set_xlabel("test months, Jun-Jul 2026")
    save(fig, "04_top10.png")

    # 05: ten risk groups: predicted vs actual (ranking and calibration in one picture)
    group = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False) + 1
    dec = pd.DataFrame({"group": group, "y": y, "p": p}).groupby("group").agg(actual=("y", "mean"), predicted=("p", "mean"), flights=("y", "size"))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(dec.index, dec["actual"], color=SERIES, width=0.72, label="actually disrupted")
    ax.plot(dec.index, dec["predicted"], color=PREDICTED, marker="o", markersize=8, label="model's average prediction")
    ax.set_xticks(dec.index)
    ax.set_xlabel("risk group (1 = the 10% of flights the model rated safest, 10 = riskiest)")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    ax.set_title("Gradient boosting on the test months: predicted vs actual, by risk group")
    save(fig, "05_deciles.png")

    # calibration by month: the ranking holds, the level moves with the season
    pred["month"] = pred["date"].str[:7]
    by_month = pred.groupby("month").agg(actual=(TARGET, "mean"), predicted=(GB, "mean"), flights=(TARGET, "size"))

    # 06: what drives the risk: shuffle one feature at a time and see how much the ranking (AUC) drops
    d = load()
    train, valid, test = split(d)
    seen = pd.concat([train, valid])
    settings = json.loads((HERE / "results" / "settings.json").read_text())["gb_chosen"]
    model = gradient_boosting(**settings).fit(seen[FEATURES], seen[TARGET])
    assert np.allclose(model.predict_proba(test[FEATURES])[:, 1].round(6), p), "refit model differs from train.py's"
    sample = test.sample(30_000, random_state=0)
    imp = permutation_importance(model, sample[FEATURES], sample[TARGET], scoring="roc_auc", n_repeats=5, random_state=0)
    importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.barh([NAMES[f] for f in importance.index], importance.values, color=SERIES, height=0.6)
    for i, v in enumerate(importance.values):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", color=INK, fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_title("Drop in AUC when one feature is scrambled (bigger = the model leans on it more)")
    save(fig, "06_importance.png")

    out = {
        "deciles": dec.round(4).reset_index().to_dict(orient="records"),
        "by_month": by_month.round(4).reset_index().to_dict(orient="records"),
        "importance_auc_drop": importance.sort_values(ascending=False).round(4).to_dict(),
    }
    (HERE / "results" / "evaluation.json").write_text(json.dumps(out, indent=2) + "\n")
    print(dec.round(3).to_string())
    print(by_month.round(3).to_string())
    print(importance.sort_values(ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
