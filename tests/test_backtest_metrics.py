from __future__ import annotations

import unittest

import pandas as pd

from src.backtest.metrics import max_drawdown, summarize_performance


class BacktestMetricsTest(unittest.TestCase):
    def test_max_drawdown_uses_running_peak(self) -> None:
        equity = pd.Series([100.0, 120.0, 90.0, 150.0])

        self.assertAlmostEqual(max_drawdown(equity), -0.25)

    def test_summarize_performance_reports_core_fields(self) -> None:
        returns = pd.Series([0.0, 0.10, 0.10])
        equity = pd.Series([100.0, 110.0, 121.0])
        exposure = pd.Series([0.0, 1.0, 1.0])
        turnover = pd.Series([0.0, 1.0, 0.0])

        summary = summarize_performance(
            daily_returns=returns,
            equity_curve=equity,
            exposure=exposure,
            turnover=turnover,
            periods_per_year=252,
        )

        self.assertAlmostEqual(summary["total_return"], 0.21)
        self.assertAlmostEqual(summary["max_drawdown"], 0.0)
        self.assertAlmostEqual(summary["average_exposure"], 2 / 3)
        self.assertAlmostEqual(summary["turnover"], 1.0)
        self.assertEqual(summary["days"], 3.0)

    def test_summarize_performance_rejects_empty_equity_curve(self) -> None:
        with self.assertRaises(ValueError):
            summarize_performance(
                daily_returns=pd.Series(dtype=float),
                equity_curve=pd.Series(dtype=float),
                exposure=pd.Series(dtype=float),
                turnover=pd.Series(dtype=float),
            )


if __name__ == "__main__":
    unittest.main()

