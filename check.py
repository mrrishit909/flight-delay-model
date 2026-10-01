"""The project's check: fails if a leak creeps in, the split overlaps, or a reported number doesn't recompute.

    ./venv/bin/python check.py

Scores are recomputed from the saved test predictions with plain numpy/pandas (AUC from ranks), not sklearn.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train import FEATURES, TARGET, TEST_START, VALID_START

HERE = Path(__file__).parent
# Known only on the day or after: none of these may be a feature or even sit in the modeling file.
DAY_OF = {"DepTime", "DepDelay", "DepDel15", "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn", "ArrTime", "ArrDelay",
          "ArrDel15", "Cancelled", "CancellationCode", "Diverted", "ActualElapsedTime", "AirTime", "CarrierDelay",
          "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "Tail_Number"}


def auc(y, p):
    """Probability a random disrupted flight scores higher than a random on-time one (ties count half)."""
    r = pd.Series(p).rank(method="average").values
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    d = pd.read_csv(HERE / "data" / "flights.csv.gz")
    assert not DAY_OF & set(d.columns) and not DAY_OF & set(FEATURES), "a day-of column is in the data"
    assert set(FEATURES) | {TARGET, "date"} == set(d.columns), "unexpected columns"
    assert d["leg"].notna().all(), "flights with no aircraft are back (they leak cancellations)"
    assert d[TARGET].isin([0, 1]).all()
    assert len(d) == 635_459 - 1_738

    train, test = d[d["date"] < VALID_START], d[d["date"] >= TEST_START]
    assert train["date"].max() < VALID_START <= TEST_START <= test["date"].min(), "periods overlap"
    assert test["date"].min() == "2026-06-01" and test["date"].max() == "2026-07-31"

    pred = pd.read_csv(HERE / "results" / "test_predictions.csv.gz")
    assert len(pred) == len(test) and (pred[TARGET].values == test[TARGET].values).all(), "predictions misaligned"
    scores = pd.read_csv(HERE / "results" / "scores.csv").query("period == 'test'").set_index("model")
    y = pred[TARGET].values
    for model, row in scores.iterrows():
        p = pred[model].values
        assert abs(auc(y, p) - row["auc"]) < 1e-4, (model, auc(y, p), row["auc"])
        assert abs(np.mean((p - y) ** 2) - row["brier"]) < 1e-4, model
        assert abs(p.mean() - row["mean_predicted"]) < 1e-5 and abs(y.mean() - row["actual_rate"]) < 1e-6, model

    gb = scores.loc["Gradient boosting"]
    assert gb["auc"] > scores.loc["Pivot table (airline x hour)", "auc"] > scores.loc["Base rate", "auc"]
    assert gb["auc"] > scores.loc["Gradient boosting, no aircraft features", "auc"]

    ev = json.loads((HERE / "results" / "evaluation.json").read_text())
    group = pd.qcut(pd.Series(pred["Gradient boosting"]).rank(method="first"), 10, labels=False) + 1
    actual = pd.Series(y).groupby(group).mean()
    assert all(abs(actual[r["group"]] - r["actual"]) < 1e-4 for r in ev["deciles"]), "risk-group table doesn't recompute"
    months = pred.assign(m=pred["date"].str[:7]).groupby("m")[TARGET].mean()
    assert all(abs(months[r["month"]] - r["actual"]) < 1e-4 for r in ev["by_month"])

    print(f"OK: no day-of columns, {len(d):,} flights, periods don't overlap, "
          f"{len(scores)} models' test AUC/Brier recompute, gradient boosting AUC {gb['auc']:.3f} beats every baseline")


if __name__ == "__main__":
    main()
