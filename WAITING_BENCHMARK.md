# Extended synthetic waiting-time benchmark

This benchmark is generated, not measured Nairobi service. Real place names are labels only. Two services per stage, both directions, arrival times, shocks and headways are fictional. Increasing simulated records does not increase evidence of field accuracy.

## Targets
- Primary regression target: `wait_minutes`, residual time until the next arrival.
- Comparison-only binary target: `arrival_within_5_min = 1` if wait is at most 5 minutes, else 0. Five minutes is an illustrative fixed research question, not a validated operational policy.
- Classification uses positive-class probabilities and a fixed 0.5 cutoff. ROC-AUC uses probabilities, never hard labels. Accuracy, precision, recall, F1, MCC and a labelled 2x2 confusion matrix use hard labels. A prior baseline exposes class imbalance. MAE/RMSE retain the original waiting-time objective.

## Scale and independence
Default: 365 days beginning 1 November 2025, three named CBD stages, two fictional services, two directions, morning/afternoon 180-minute continuous sessions: 8,760 sessions and roughly 1.4 million minute queries. The generator is configurable via `--days`, but unlimited synthetic data is neither a meaningful goal nor independent evidence. Gzip keeps disk use bounded. Queries within sessions are correlated.

## Leakage controls
Assign complete days to 70% train / 15% validation / 15% test before generating sessions and labels. Queries never cross sessions. Only intervals with an observed next arrival are labelled; unobserved final tails and initial intervals without prior headway are omitted. Features: stage/service/direction codes, query hour, weekday, elapsed time since last arrival and previous completed headway. Future arrival, target wait, binary target, session identifiers and hidden generator regime are never model inputs. Encoders/scalers fit on training only. No rebalancing or random-row split. Fixed model settings and 0.5 threshold are specified before test evaluation. Validation is reported, not used to tune on test.

## Run
Use the existing NganyaFlow Python 3.12 environment with numpy/pandas/scikit-learn installed, or install `waiting_time[dev,notebooks]` in this workspace.

```powershell
python scripts/build_waiting_benchmark.py --output data/processed/waiting-benchmark-v1 --days 365
python scripts/evaluate_waiting_benchmark.py --dataset data/processed/waiting-benchmark-v1
```

Generator refuses an existing output directory. `manifest.json` contains seed, split boundaries, counts and SHA256 hashes. `arrivals.csv.gz` retains session boundaries. `queries.csv.gz` retains provenance and both targets. `evaluation.json` records all measured comparisons and prevalence. Models: prior baseline, logistic regression, histogram gradient boosting classifier, plus median regression and last-headway baseline. This is a comparison experiment, not a replacement or activation of the mobile QRF model. No model is published.

## Independent review notes
The sampled population is eligible minute queries starting one minute after a vehicle arrival and ending before the next observed arrival. It is not all possible passenger arrival times; initial and right-censored terminal intervals are omitted. Scores apply to this sampling scheme. CSV content is reproducible for the same seed and day count; gzip container timestamps can change compressed-byte hashes on rebuild. The manifest checksums identify the delivered compressed files exactly.

