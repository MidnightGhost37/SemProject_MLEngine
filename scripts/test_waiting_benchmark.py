import gzip
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from build_waiting_benchmark import FEATURES, generate
from evaluate_waiting_benchmark import classification


class BenchmarkTests(unittest.TestCase):
    def test_targets_splits_boundaries_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, b = root / "a", root / "b"
            generate(a, days=10, seed=7)
            generate(b, days=10, seed=7)
            with gzip.open(a / "queries.csv.gz", "rb") as x, gzip.open(b / "queries.csv.gz", "rb") as y:
                self.assertEqual(x.read(), y.read())
            q = pd.read_csv(a / "queries.csv.gz")
            arrivals = pd.read_csv(a / "arrivals.csv.gz")
            self.assertTrue(((q.wait_minutes <= 5).astype(int) == q.arrival_within_5_min).all())
            self.assertEqual(q.groupby("session_id").split.nunique().max(), 1)
            self.assertTrue(set(FEATURES).isdisjoint({"wait_minutes", "arrival_within_5_min", "query_at", "session_id", "split"}))
            self.assertLess(q.loc[q.split == "train", "day"].max(), q.loc[q.split == "validation", "day"].min())
            self.assertLess(q.loc[q.split == "validation", "day"].max(), q.loc[q.split == "test", "day"].min())
            for sid in q.session_id.unique()[:5]:
                events = pd.to_datetime(arrivals.loc[arrivals.session_id == sid, "observed_at"])
                rows = q.loc[q.session_id == sid]
                for row in rows.itertuples():
                    at = pd.Timestamp(row.query_at)
                    previous = events[events <= at]
                    following = events[events > at]
                    self.assertGreaterEqual(len(previous), 2)
                    self.assertGreaterEqual(len(following), 1)
                    self.assertAlmostEqual((following.iloc[0] - at).total_seconds() / 60, row.wait_minutes)
                    self.assertAlmostEqual((previous.iloc[-1] - previous.iloc[-2]).total_seconds() / 60, row.last_headway_minutes)
            with self.assertRaises(FileExistsError):
                generate(a, days=10)
            self.assertEqual(json.loads((a / "manifest.json").read_text())["data_kind"], "synthetic")

    def test_metric_semantics(self):
        import numpy as np
        result = classification(np.array([0, 0, 1, 1]), np.array([.1, .6, .4, .9]))
        self.assertEqual(result["confusion_matrix_actual_rows_predicted_columns_0_1"], [[1, 1], [1, 1]])
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["roc_auc"], .75)
        self.assertEqual(result["mcc"], 0)


if __name__ == "__main__":
    unittest.main()
