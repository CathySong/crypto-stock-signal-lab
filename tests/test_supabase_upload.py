from __future__ import annotations

import math
import unittest

import pandas as pd

from src.data.upload_to_supabase import clean_records, dedupe_on_conflict, infer_price_col


class SupabaseUploadTest(unittest.TestCase):
    def test_dedupe_on_conflict_keeps_last_record(self) -> None:
        records = [
            {"symbol": "BTC", "date": "2026-01-01", "value": 1},
            {"symbol": "BTC", "date": "2026-01-01", "value": 2},
            {"symbol": "ETH", "date": "2026-01-01", "value": 3},
        ]

        deduped = dedupe_on_conflict(records, "symbol,date")

        self.assertEqual(len(deduped), 2)
        self.assertEqual(
            sorted(deduped, key=lambda row: row["symbol"]),
            [
                {"symbol": "BTC", "date": "2026-01-01", "value": 2},
                {"symbol": "ETH", "date": "2026-01-01", "value": 3},
            ],
        )

    def test_clean_records_replaces_non_json_numbers_with_none(self) -> None:
        cleaned = clean_records(
            [
                {"ok": 1.0, "nan": float("nan"), "inf": float("inf"), "missing": pd.NA},
            ]
        )

        self.assertEqual(cleaned[0]["ok"], 1.0)
        self.assertIsNone(cleaned[0]["nan"])
        self.assertIsNone(cleaned[0]["inf"])
        self.assertIsNone(cleaned[0]["missing"])
        self.assertFalse(math.isnan(cleaned[0]["ok"]))

    def test_infer_price_col_prefers_adjusted_close(self) -> None:
        self.assertEqual(infer_price_col(pd.DataFrame(columns=["close", "adj_close"])), "adj_close")
        self.assertEqual(infer_price_col(pd.DataFrame(columns=["price_usd"])), "price_usd")
        self.assertIsNone(infer_price_col(pd.DataFrame(columns=["volume"])))


if __name__ == "__main__":
    unittest.main()

