from __future__ import annotations

import math
import unittest

from src.reports.build_watchlist import classify_asset, research_priority_score, score_from_auc, score_from_sharpe


class WatchlistTest(unittest.TestCase):
    def test_classify_asset_research_now_for_strong_manageable_setup(self) -> None:
        action, flags = classify_asset(
            final_score=72,
            timing_score=74,
            value_score=65,
            risk_score=30,
            best_sharpe=0.8,
            ml_auc=0.62,
        )

        self.assertEqual(action, "Research now")
        self.assertIn("strong blended score with manageable measured risk", flags)
        self.assertIn("ML benchmark shows directional separation", flags)

    def test_classify_asset_waits_on_high_risk(self) -> None:
        action, flags = classify_asset(
            final_score=62,
            timing_score=65,
            value_score=55,
            risk_score=82,
            best_sharpe=0.5,
            ml_auc=0.55,
        )

        self.assertEqual(action, "Wait")
        self.assertIn("risk score is high enough to demand extra caution", flags)

    def test_priority_score_uses_neutral_defaults_for_missing_model_context(self) -> None:
        score = research_priority_score(final_score=60, risk_score=40, best_sharpe=math.nan, ml_auc=math.nan)

        self.assertAlmostEqual(score, 57.0)

    def test_component_scores_are_clipped(self) -> None:
        self.assertEqual(score_from_sharpe(-5), 0)
        self.assertEqual(score_from_sharpe(10), 100)
        self.assertEqual(score_from_auc(0.1), 0)
        self.assertEqual(score_from_auc(0.9), 100)


if __name__ == "__main__":
    unittest.main()

