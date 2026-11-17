"""Reproducible synthetic benchmark, not observed Nairobi operations."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

FEATURES = ["stage_code", "service_code", "direction_code", "hour", "weekday",
            "elapsed_minutes", "last_headway_minutes"]
COLUMNS = ["session_id", "day", "split", "stage_code", "service_code", "direction_code",
           "query_at", "hour", "weekday", "elapsed_minutes", "last_headway_minutes",
           "wait_minutes", "arrival_within_5_min", "data_kind"]


def generate(output, days=365, seed=20261117):
    if days < 10:
        raise ValueError("Need at least ten complete dates")
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    counts = {s: {"queries": 0, "positive": 0, "sessions": 0, "arrivals": 0}
              for s in ("train", "validation", "test")}
    start = datetime.fromisoformat("2025-11-01T00:00:00+03:00")
    train_end, val_end = int(days * .70), int(days * .85)
    with gzip.open(output / "queries.csv.gz", "wt", newline="", encoding="utf-8") as qf, \
            gzip.open(output / "arrivals.csv.gz", "wt", newline="", encoding="utf-8") as af:
        qw = csv.writer(qf)
        aw = csv.writer(af)
        qw.writerow(COLUMNS)
        aw.writerow(["session_id", "day", "split", "stage_code", "service_code",
                     "direction_code", "observed_at", "session_start", "session_end", "data_kind"])
        for d in range(days):
            date = start + timedelta(days=d)
            split = "train" if d < train_end else "validation" if d < val_end else "test"
            for stage, base in enumerate([5.0, 7.0, 9.0]):
                for service in range(2):
                    for direction in range(2):
                        for hour in [7, 14]:
                            sid = f"sim-{d}-{stage}-{service}-{direction}-{hour}"
                            begin = date.replace(hour=hour)
                            end = begin + timedelta(minutes=180)
                            regime = base * (1 + .25 * service + .12 * direction)
                            regime *= (1.25 if hour == 14 else 1) * (1.2 if date.weekday() > 4 else 1)
                            # Unobserved day/session variation; never exported as a predictor.
                            regime *= math.exp(rng.gauss(0, .12))
                            times = []
                            at, previous = begin + timedelta(minutes=1), regime
                            while at < end:
                                times.append(at)
                                aw.writerow([sid, date.date(), split, stage, service, direction,
                                             at.isoformat(), begin.isoformat(), end.isoformat(), "synthetic"])
                                shock = rng.uniform(5, 15) if rng.random() < .025 else 0
                                gap = min(45, max(1.5, .35 * previous + .65 * regime
                                                  + rng.gauss(0, 2.0) + shock))
                                previous = gap
                                at += timedelta(seconds=round(gap * 60))
                            counts[split]["sessions"] += 1
                            counts[split]["arrivals"] += len(times)
                            for prev, current, following in zip(times, times[1:], times[2:]):
                                headway = (current - prev).total_seconds() / 60
                                at = current + timedelta(minutes=1)
                                while at < following:
                                    elapsed = (at - current).total_seconds() / 60
                                    wait = (following - at).total_seconds() / 60
                                    positive = int(wait <= 5)
                                    qw.writerow([sid, date.date(), split, stage, service, direction,
                                                 at.isoformat(), at.hour + at.minute / 60,
                                                 at.weekday(), elapsed, headway, wait, positive, "synthetic"])
                                    counts[split]["queries"] += 1
                                    counts[split]["positive"] += positive
                                    at += timedelta(minutes=1)
    manifest = {"data_kind": "synthetic", "seed": seed, "days": days,
                "start_date": str(start.date()),
                "split_day_offsets": [0, train_end, val_end, days], "counts": counts,
                "positive_class": "next vehicle arrival in <= 5 minutes",
                "probability_threshold": .5, "features": FEATURES,
                "place_names": ["Kencom", "Odeon", "Railways area"],
                "services": "Two fictional services at every stage; no real route assertion",
                "limitations": "All timestamps, headways, services and outcomes are simulated. "
                "Minute queries are correlated, not independent field observations. "
                "More synthetic rows do not establish real-world accuracy.",
                "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in output.glob("*.csv.gz")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.days, args.seed), indent=2))
