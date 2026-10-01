# Which Florida flights will be disrupted? (machine learning)

The question: the night before, using only what's on the schedule, can we tell which flights are most likely to
be disrupted (arrive 15+ minutes late, be cancelled or diverted)? 633,721 departures from 20 Florida airports,
Aug 2025 – Jul 2026, scikit-learn, with a test on two months the models never saw.

Build log: https://mrrishit909.github.io/projects/flight-delay-model/

## Steps

| Step | File | What it does |
|---|---|---|
| 1 | `prepare.py` → `data/flights.csv.gz` | one row per Florida departure, with night-before features only: airline, airports, scheduled times, distance, and the aircraft's day (which flight of the day it is for the plane, scheduled minutes on the ground before it), worked out from all 7 million US flights |
| 2 | `explore.py` → `charts/01–03` | disruption rate by departure hour, the plane's flight of the day, and month |
| 3 | `train.py` → `results/scores.csv` | time split; four methods from "guess the average" to gradient boosting; settings chosen on validation; test touched once |
| 4 | `evaluate.py` → `charts/04–06`, `results/evaluation.json` | risk groups, calibration by month, which features matter; a leak found and removed |
| 5 | `check.py` | fails if a day-of column sneaks in, the periods overlap, or any test score doesn't recompute (AUC from ranks, without sklearn) |

## Run it

```
python3 -m venv venv && ./venv/bin/pip install pandas scikit-learn matplotlib
./venv/bin/python prepare.py      # needs the 12 BTS zips (download.py in the florida-flight-delays repo); ~20 s
./venv/bin/python explore.py
./venv/bin/python train.py        # ~30 s on an M3 Pro
./venv/bin/python evaluate.py
./venv/bin/python check.py        # prints OK or fails
```
`data/flights.csv.gz` is committed, so steps 2–5 run without the raw files.

## How it's tested

| Period | Flights | Disrupted | Used for |
|---|---:|---:|---|
| Aug 2025 – Mar 2026 | 422,136 | 25.7% | fitting |
| Apr – May 2026 | 110,469 | 23.0% | choosing gradient-boosting settings (3 tried) |
| Jun – Jul 2026 | 101,116 | 30.1% | the test: every model refit on Aug–May, then scored once |

The split is by time, not random: a random split would let the model learn from flights later the same day.

## Results (test months, Jun – Jul 2026)

| Method | AUC | Avg precision | Riskiest 10%: disrupted | Riskiest 10%: share of all disruptions |
|---|---:|---:|---:|---:|
| Base rate (everyone gets the average) | 0.500 | 0.301 | 30.2% | 10.0% |
| Pivot table: airline × departure hour | 0.707 | 0.456 | 50.1% | 16.7% |
| Logistic regression | 0.724 | 0.519 | 60.9% | 20.3% |
| **Gradient boosting** | **0.736** | **0.548** | **64.3%** | **21.4%** |
| Gradient boosting without the aircraft features | 0.715 | 0.486 | 56.1% | 18.7% |

AUC is the chance a random disrupted flight gets a higher risk score than a random on-time one (0.5 = coin toss).

- **Delays snowball through the plane's day.** 13.2% of a plane's first flights of the day are disrupted vs 42.4% of
  its sixth or later. 8.6% of 5 am departures vs 38.9% at 8 pm. Turns of 30 minutes or less: 39.1% vs 28.7%
  for 31–180 minutes. The plane's flight of the day and departure hour are the features the model leans on most.
- **It ranks well but under-predicts a bad month.** Across the test months it predicted 24.9% on average; 30.1%
  happened. June was close (24.6% predicted, 26.1% actual), July was not (25.2% vs 33.8%). The model knows which
  flights are riskier, not how stormy the month will be; it has no weather.
- **Honest ceiling.** AUC 0.74 means it's a useful ranking, not a crystal ball: 36% of its riskiest 10% were
  not disrupted.

## Leaks found and how they're handled

- **No aircraft = cancelled.** 1,738 flights have no tail number, and every one was cancelled, so the model could have
  learnt "no plane listed → disrupted". The night-before schedule would have named a plane, so these rows are dropped.
  `check.py` fails if they come back.
- **Aircraft swaps.** BTS records the plane that actually flew. Airlines plan it the day before but swap some on the day,
  often because of trouble, so the plane's flight of the day and turn time may carry a little day-of information.
  At most, that is the gap to the model without those two features: AUC 0.736 vs 0.715. The real planned assignment
  isn't public, so this can't be measured exactly.

## Data

US DOT Bureau of Transportation Statistics, Reporting Carrier On-Time Performance, Aug 2025 – Jul 2026,
https://www.transtats.bts.gov/ (public domain). Same raw files as [florida-flight-delays](https://github.com/mrrishit909/florida-flight-delays):
its 635,459 Florida departures minus the 1,738 above.

## Not done

- No weather or forecast data. That is the obvious next feature, and the reason July is under-predicted.
- No recalibration for the season (e.g. scaling to the last few weeks' disruption rate).
- Leg order uses local scheduled times; a short westbound hop could be misordered (rare). UTC times would fix it.
- Not deployed: no daily scoring job or app. It's a model with an honest test, not a product.
