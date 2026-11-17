"""Leakage-checked model comparison on synthetic chronological holdouts."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, matthews_corrcoef,
                             mean_absolute_error, root_mean_squared_error)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["stage_code", "service_code", "direction_code", "hour", "weekday",
            "elapsed_minutes", "last_headway_minutes"]


def classification(y, probability):
    predicted = (probability >= .5).astype(int)
    return {"accuracy": accuracy_score(y, predicted),
            "precision": precision_score(y, predicted, zero_division=0),
            "recall": recall_score(y, predicted, zero_division=0),
            "f1_score": f1_score(y, predicted, zero_division=0),
            "roc_auc": roc_auc_score(y, probability),
            "mcc": matthews_corrcoef(y, predicted),
            "confusion_matrix_actual_rows_predicted_columns_0_1":
                confusion_matrix(y, predicted, labels=[0, 1]).tolist()}


def evaluate(dataset):
    data = pd.read_csv(dataset / "queries.csv.gz")
    assert set(data.data_kind) == {"synthetic"}
    assert (data.wait_minutes > 0).all()
    assert ((data.wait_minutes <= 5).astype(int) == data.arrival_within_5_min).all()
    assert not data.duplicated(["session_id", "query_at"]).any()
    assert data.groupby("session_id").split.nunique().max() == 1
    parts = {s: data.loc[data.split == s] for s in ["train", "validation", "test"]}
    assert parts["train"].day.max() < parts["validation"].day.min()
    assert parts["validation"].day.max() < parts["test"].day.min()
    for part in parts.values():
        assert set(part.arrival_within_5_min) == {0, 1}
    train = parts["train"]
    x, y = train[FEATURES], train.arrival_within_5_min
    categorical = ["stage_code", "service_code", "direction_code", "weekday"]
    numeric = [c for c in FEATURES if c not in categorical]
    preprocessing = ColumnTransformer([
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numeric", StandardScaler(), numeric),
    ])
    models = {
        "prior_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(preprocessing, LogisticRegression(max_iter=300)),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, max_bins=64, learning_rate=.1,
            categorical_features=[c in categorical for c in FEATURES],
            early_stopping=False, random_state=42),
    }
    report = {"data_kind": "synthetic", "positive_class": "wait_minutes <= 5",
              "probability_threshold": .5, "features": FEATURES,
              "split_summary": {s: {"queries": len(p), "sessions": p.session_id.nunique(),
                                    "days": p.day.nunique(),
                                    "positive_prevalence": float(p.arrival_within_5_min.mean())}
                                for s, p in parts.items()}, "classification": {}, "regression": {},
              "warning": "Synthetic-only scores; correlated queries; not Nairobi field accuracy. "
              "Fixed settings and threshold; no test-based tuning. Existing QRF app model unchanged."}
    for name, model in models.items():
        model.fit(x, y)
        report["classification"][name] = {}
        for split in ["validation", "test"]:
            p = parts[split]
            probability = model.predict_proba(p[FEATURES])[:, list(model.classes_).index(1)]
            report["classification"][name][split] = classification(p.arrival_within_5_min, probability)
        print(name, report["classification"][name]["test"], flush=True)
    regression = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=100, max_leaf_nodes=15, max_bins=64,
        categorical_features=[c in categorical for c in FEATURES],
        early_stopping=False, random_state=42)
    regression.fit(x, train.wait_minutes)
    for split in ["validation", "test"]:
        p = parts[split]
        predictions = {"last_headway_baseline": np.maximum(0, p.last_headway_minutes - p.elapsed_minutes),
                       "hist_gradient_boosting_median": regression.predict(p[FEATURES])}
        report["regression"][split] = {name: {
            "mae_minutes": mean_absolute_error(p.wait_minutes, pred),
            "rmse_minutes": root_mean_squared_error(p.wait_minutes, pred)}
            for name, pred in predictions.items()}
    report["software_versions"] = {p: importlib.metadata.version(p)
                                   for p in ["numpy", "pandas", "scikit-learn"]}
    report["source_hash_format"] = "sha256 of UTF-8 source with LF line endings"
    report["source_sha256"] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_text(encoding="utf-8")
                             .encode("utf-8")).hexdigest()
        for name in ["build_waiting_benchmark.py", "evaluate_waiting_benchmark.py"]
    }
    (dataset / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    print(json.dumps(evaluate(parser.parse_args().dataset), indent=2))
