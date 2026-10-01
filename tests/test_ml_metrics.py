from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from src.ml.logistic import fit_logistic_regression, predict_proba
from src.ml.metrics import roc_auc_score, summarize_classifier


class MlMetricsTest(unittest.TestCase):
    def test_roc_auc_score_handles_perfect_and_single_class_cases(self) -> None:
        self.assertAlmostEqual(
            roc_auc_score(pd.Series([0, 0, 1, 1]), pd.Series([0.1, 0.2, 0.8, 0.9])),
            1.0,
        )
        self.assertTrue(math.isnan(roc_auc_score(pd.Series([1, 1]), pd.Series([0.8, 0.9]))))

    def test_summarize_classifier_reports_expected_metrics(self) -> None:
        summary = summarize_classifier(
            y_true=pd.Series([0, 1, 1, 0]),
            y_prob=pd.Series([0.1, 0.7, 0.4, 0.2]),
            threshold=0.5,
        )

        self.assertEqual(summary["samples"], 4.0)
        self.assertAlmostEqual(summary["accuracy"], 0.75)
        self.assertAlmostEqual(summary["precision"], 1.0)
        self.assertAlmostEqual(summary["recall"], 0.5)
        self.assertAlmostEqual(summary["roc_auc"], 1.0)

    def test_logistic_regression_single_class_returns_prior_model(self) -> None:
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 1.0, 1.0])

        model = fit_logistic_regression(x, y)
        proba = predict_proba(model, np.array([[10.0], [20.0]]))

        self.assertEqual(model.prior, 1.0)
        self.assertTrue(np.allclose(proba, np.array([1.0, 1.0])))


if __name__ == "__main__":
    unittest.main()

