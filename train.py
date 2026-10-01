"""Step 3: train on the past, tune on the next two months, test once on the last two.

    ./venv/bin/python train.py

    train  Aug 2025 - Mar 2026     fit every model
    valid  Apr - May 2026          pick the gradient-boosting settings
    test   Jun - Jul 2026          touched once, after everything is chosen (models refit on train + valid)

Writes results/scores.csv, results/settings.json and results/test_predictions.csv.gz.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
CATEGORIES = ["carrier", "origin", "dest"]
NUMBERS = ["dep_hour", "arr_hour", "dow", "leg", "turn_min", "distance", "sched_min"]
FEATURES = CATEGORIES + NUMBERS          # check.py asserts nothing measured on the day sneaks in here
TARGET = "disrupted"
VALID_START, TEST_START = "2026-04-01", "2026-06-01"
GB_SETTINGS = [                          # small, deliberate grid; chosen on validation log loss
    {"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 300},
    {"learning_rate": 0.1, "max_leaf_nodes": 63, "max_iter": 300},
    {"learning_rate": 0.05, "max_leaf_nodes": 63, "max_iter": 600},
]


def load():
    d = pd.read_csv(HERE / "data" / "flights.csv.gz")
    for c in CATEGORIES:
        d[c] = d[c].astype("category")
    return d


def split(d):
    return d[d["date"] < VALID_START], d[(d["date"] >= VALID_START) & (d["date"] < TEST_START)], d[d["date"] >= TEST_START]


class BaseRate:
    """Every flight gets the training average. Ranks nothing: AUC 0.5."""
    def fit(self, X, y):
        self.p = y.mean()
        return self

    def predict_proba(self, X):
        return np.column_stack([1 - np.full(len(X), self.p), np.full(len(X), self.p)])


class PivotTable:
    """What an analyst would do in a spreadsheet: disruption rate by airline x departure hour,
    shrunk toward the overall rate when a cell has few flights (50 pseudo-flights)."""
    def fit(self, X, y, k=50):
        self.p = y.mean()
        g = pd.DataFrame({"c": X["carrier"].astype(str), "h": X["dep_hour"], "y": y}).groupby(["c", "h"])["y"]
        self.rate = ((g.sum() + k * self.p) / (g.count() + k)).to_dict()
        return self

    def predict_proba(self, X):
        p = np.array([self.rate.get(key, self.p) for key in zip(X["carrier"].astype(str), X["dep_hour"])])
        return np.column_stack([1 - p, p])


def logistic():
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), CATEGORIES + ["dep_hour", "dow"]),
        ("num", make_pipeline(StandardScaler()), ["leg", "distance", "sched_min"]),
        ("turn", "passthrough", ["tight_turn", "no_turn"]),
    ])
    return make_pipeline(prep, LogisticRegression(max_iter=2000))


def with_turn_flags(X):
    """Logistic regression can't take blanks: turn time becomes 'tight (<=30 min)' and 'no earlier leg' flags."""
    X = X.copy()
    X["no_turn"] = X["turn_min"].isna().astype(int)
    X["tight_turn"] = (X["turn_min"] <= 30).astype(int)
    X["leg"] = X["leg"].fillna(1).clip(upper=8)
    return X


def gradient_boosting(**settings):
    return HistGradientBoostingClassifier(categorical_features="from_dtype", early_stopping=False, random_state=0, **settings)


def top_tenth(p, seed=0):
    """Exactly 10% of flights with the highest risk; ties broken at random (a pivot table has many equal scores)."""
    order = np.lexsort((np.random.default_rng(seed).random(len(p)), -p))
    top = np.zeros(len(p), dtype=bool)
    top[order[: len(p) // 10]] = True
    return top


def scores(y, p):
    top = top_tenth(p)
    return {
        "auc": roc_auc_score(y, p), "avg_precision": average_precision_score(y, p),
        "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p),
        "mean_predicted": p.mean(), "actual_rate": y.mean(),
        "top10_rate": y[top].mean(), "top10_share_of_disruptions": y[top].sum() / y.sum(),
    }


def models(best_gb):
    return {
        "Base rate": (BaseRate, lambda X: X),
        "Pivot table (airline x hour)": (PivotTable, lambda X: X),
        "Logistic regression": (logistic, with_turn_flags),
        "Gradient boosting": (lambda: gradient_boosting(**best_gb), lambda X: X),
        # BTS records the aircraft that actually flew; airlines swap some on the day. This one ignores the aircraft
        # (leg, turn time) to show how much those two features add, in case swaps leak day-of information.
        "Gradient boosting, no aircraft features": (lambda: gradient_boosting(**best_gb), lambda X: X.drop(columns=["leg", "turn_min"])),
    }


def main():
    d = load()
    train, valid, test = split(d)
    print(f"train {len(train):,}  valid {len(valid):,}  test {len(test):,}")

    # 1. choose gradient-boosting settings on validation
    tried = []
    for s in GB_SETTINGS:
        p = gradient_boosting(**s).fit(train[FEATURES], train[TARGET]).predict_proba(valid[FEATURES])[:, 1]
        tried.append({**s, "valid_log_loss": log_loss(valid[TARGET], p), "valid_auc": roc_auc_score(valid[TARGET], p)})
        print("  GB", tried[-1])
    best = min(tried, key=lambda t: t["valid_log_loss"])
    best_gb = {k: best[k] for k in GB_SETTINGS[0]}

    # 2. score every model on validation (fit on train), then on test (refit on train + valid)
    rows, preds = [], {"date": test["date"].values, TARGET: test[TARGET].values}
    seen = pd.concat([train, valid])
    for name, (make, prep) in models(best_gb).items():
        p = make().fit(prep(train[FEATURES]), train[TARGET]).predict_proba(prep(valid[FEATURES]))[:, 1]
        rows.append({"model": name, "period": "valid", **scores(valid[TARGET].values, p)})
        p = make().fit(prep(seen[FEATURES]), seen[TARGET]).predict_proba(prep(test[FEATURES]))[:, 1]
        rows.append({"model": name, "period": "test", **scores(test[TARGET].values, p)})
        preds[name] = p.round(6)

    out = pd.DataFrame(rows)
    (HERE / "results").mkdir(exist_ok=True)
    out.round(6).to_csv(HERE / "results" / "scores.csv", index=False)
    pd.DataFrame(preds).to_csv(HERE / "results" / "test_predictions.csv.gz", index=False)
    (HERE / "results" / "settings.json").write_text(json.dumps(
        {"features": FEATURES, "valid_start": VALID_START, "test_start": TEST_START,
         "gb_tried": tried, "gb_chosen": best_gb}, indent=2) + "\n")
    print(out[out["period"] == "test"].drop(columns="period").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
