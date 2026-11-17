# SemFlow: waiting-time AI for NganyaFlow

The primary objective is **minutes until the next vehicle arrives** at a selected stage, route and direction, matching the mobile presentation. SemFlow is the AI workstream, not a separate demand predictor.

## Start here
1. Read [the shared target and evidence contract](WAITING_TIME_STANDARD.md).
2. Open `notebooks/00_waiting_time_pipeline.ipynb` for preparation, chronological evaluation and QRF median/interval predictions.
3. `waiting_time/` contains a snapshot of the existing NganyaFlow backend source, tests, catalog and explicitly synthetic arrival bundle. No credentials, live database or active model is copied. Keep source synchronized with the NganyaFlow backend rather than evolving a second target.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e "./waiting_time[dev,notebooks]"
.\.venv\Scripts\python.exe -m nganyaflow.prepare_waiting --bundle waiting_time/data/examples/nairobi-cbd-synthetic-v1.json --catalog waiting_time/data/reference/catalog.json --output data/processed/waiting-time
.\.venv\Scripts\python.exe -m pytest waiting_time/tests
```
Python 3.12+ is required for the waiting-time package. Preparation refuses to overwrite an existing output directory. It does not activate or publish a model.

## Evidence boundaries
The runnable arrival data is synthetic, not observed Nairobi service. Digital Matatus supplies real historical network geography only. No verified Nairobi arrival dataset has been added. Real field sessions remain necessary before claiming local accuracy. The QRF produces median and 10th/90th percentiles, not passenger counts, seat guarantees or journey duration.

## Preserved historical work
The MTA ridership notebooks and `scripts/prepare_data.py` are **legacy demand/background exercises**, not the current model or waiting-time labels. They remain unchanged computationally to preserve prior coursework. See [legacy preparation instructions](README-demand-preparation-legacy.md). Nairobi weather must not be joined to New York ridership. Static GTFS schedules must not be labelled as observed arrivals.

The shared standard's `server/` example paths refer to the original NganyaFlow backend; in this workspace use the `waiting_time/` paths shown above.

## Large synthetic comparison benchmark
See [the benchmark guide](WAITING_BENCHMARK.md) and `data/processed/waiting-benchmark-v1/` for the extended dataset and Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC and MCC comparisons. Scores are simulation-only.
