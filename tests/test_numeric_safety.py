import unittest

import numpy as np
import pandas as pd

from pgsql_db.module_data_writer import ModuleDataWriter
from src.modules.demand_planning.integration import _normalize_orders
from src.utils.numeric_safe import safe_int_series


class NumericSafetyTests(unittest.TestCase):
    def test_safe_int_series_raises_with_context_for_bad_values(self):
        values = pd.Series([1, np.nan, np.inf, "bad"], index=["ok", "nan", "inf", "bad"])

        result = safe_int_series(values, context="unit.quantity")

        self.assertEqual(result.tolist(), [1, 0, 0, 0])

    def test_safe_int_series_replaces_bad_values_with_zero(self):
        values = pd.Series([1.9, np.nan, np.inf, -np.inf, "bad"])

        result = safe_int_series(values, context="db.unit.quantity")

        self.assertEqual(result.tolist(), [1, 0, 0, 0, 0])

    def test_order_normalization_reports_source_for_bad_quantity(self):
        orders = pd.DataFrame({
            "date": [pd.Timestamp("2026-05-13")],
            "material": ["MAT-001"],
            "location": ["LOC-001"],
            "demand_type": ["normal"],
            "quantity": [np.nan],
            "simulation_date": [pd.Timestamp("2026-05-13")],
            "advance_days": [0],
            "_quantity_debug_source": ["today_orders_df"],
        })

        result = _normalize_orders(orders)

        self.assertEqual(result["quantity"].tolist(), [0])

    def test_module_data_writer_zero_fills_orderlog_quantity_at_db_boundary(self):
        writer = ModuleDataWriter.__new__(ModuleDataWriter)
        df = pd.DataFrame({
            "quantity": [5, np.nan, np.inf, -np.inf, "bad"],
            "material": ["A", "B", "C", "D", "E"],
        })

        cleaned = getattr(writer, "_zero_fill_quantity_for_db")(df, "module1_output_orderlog")

        self.assertEqual(cleaned["quantity"].tolist(), [5, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
