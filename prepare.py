"""Step 1: one row per Florida departure, with only facts known the night before + the outcome.

    ./venv/bin/python prepare.py [path/to/raw]     # default: ~/florida-flight-delays/data/raw

Reads the 12 monthly BTS On-Time zips (Aug 2025 - Jul 2026; download.py in the florida-flight-delays repo
fetches them) and writes data/flights.csv.gz.

Allowed features are things on the schedule: airline, airports, scheduled times, distance, and where the flight
sits in its aircraft's day (leg number, scheduled turn time). Anything measured on the day (actual times, taxi,
delays, delay causes) is left out, because a model that sees it is predicting the past.
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "florida-flight-delays" / "data" / "raw"
COLS = ["FlightDate", "DayOfWeek", "Reporting_Airline", "Tail_Number", "Origin", "OriginState", "Dest",
        "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance", "ArrDel15", "Cancelled", "Diverted"]


def minutes(hhmm):
    return hhmm // 100 % 24 * 60 + hhmm % 100


def read_month(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        df = pd.read_csv(z.open(name), usecols=COLS, dtype={"Tail_Number": "string"})
    # The aircraft's day, from ALL US flights (its earlier legs may start outside Florida).
    # ponytail: legs ordered by local scheduled time; a short westbound hop could misorder (rare), UTC times would fix it
    df["dep_min"] = minutes(df["CRSDepTime"])
    df = df.sort_values(["FlightDate", "Tail_Number", "dep_min"])
    plane = df.groupby(["FlightDate", "Tail_Number"], sort=False)
    df["leg"] = plane.cumcount() + 1
    prev_dest, prev_arr = plane["Dest"].shift(), minutes(plane["CRSArrTime"].shift())
    turn = df["dep_min"] - prev_arr                          # same airport, so both times are the same local clock
    df["turn_min"] = turn.where((prev_dest == df["Origin"]) & (turn >= 0))
    df.loc[df["Tail_Number"].isna(), ["leg", "turn_min"]] = np.nan
    return df[df["OriginState"] == "FL"]


def build():
    files = sorted(RAW.glob("ontime_*.zip"))
    assert len(files) == 12, f"expected 12 monthly zips in {RAW}, found {len(files)}"
    raw = pd.concat([read_month(f) for f in files], ignore_index=True)
    out = pd.DataFrame({
        "date": raw["FlightDate"],
        "dow": raw["DayOfWeek"],                                   # 1 = Monday
        "carrier": raw["Reporting_Airline"],
        "origin": raw["Origin"],
        "dest": raw["Dest"],
        "dep_hour": raw["CRSDepTime"] // 100 % 24,
        "arr_hour": raw["CRSArrTime"] // 100 % 24,
        "sched_min": raw["CRSElapsedTime"],
        "distance": raw["Distance"],
        "leg": raw["leg"].astype("Int64"),
        "turn_min": raw["turn_min"].astype("Int64"),
        # the outcome a traveller cares about: 15+ minutes late, cancelled, or diverted
        "disrupted": ((raw["ArrDel15"] == 1) | (raw["Cancelled"] == 1) | (raw["Diverted"] == 1)).astype(int),
    }).sort_values(["date", "dep_hour", "carrier", "origin", "dest"], kind="stable")
    assert len(out) == 635_459, f"{len(out):,} rows; the Florida Flight Delays project has 635,459"
    (HERE / "data").mkdir(exist_ok=True)
    out.to_csv(HERE / "data" / "flights.csv.gz", index=False)
    print(f"{len(out):,} Florida departures, {out['disrupted'].mean():.1%} disrupted; "
          f"leg known for {out['leg'].notna().mean():.1%}, turn time for {out['turn_min'].notna().mean():.1%}")


if __name__ == "__main__":
    build()
