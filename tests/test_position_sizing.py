from __future__ import annotations

import unittest

from src.scoring.position_sizing import suggest_position_size


class PositionSizingTest(unittest.TestCase):
    def test_low_risk_position_is_capped_by_max_position(self) -> None:
        result = suggest_position_size(account_value=10_000, entry_price=100, risk_score=0)

        self.assertTrue(result.capped_by_max_position)
        self.assertAlmostEqual(result.stop_distance_pct, 0.03)
        self.assertAlmostEqual(result.suggested_dollar_amount, 2_000)
        self.assertAlmostEqual(result.suggested_pct_of_account, 0.20)

    def test_high_risk_position_gets_wider_stop_and_smaller_size(self) -> None:
        result = suggest_position_size(account_value=10_000, entry_price=100, risk_score=100)

        self.assertFalse(result.capped_by_max_position)
        self.assertAlmostEqual(result.stop_distance_pct, 0.25)
        self.assertAlmostEqual(result.suggested_dollar_amount, 400)
        self.assertAlmostEqual(result.suggested_pct_of_account, 0.04)

    def test_invalid_account_or_price_raises(self) -> None:
        with self.assertRaises(ValueError):
            suggest_position_size(account_value=0, entry_price=100, risk_score=50)
        with self.assertRaises(ValueError):
            suggest_position_size(account_value=10_000, entry_price=0, risk_score=50)


if __name__ == "__main__":
    unittest.main()

