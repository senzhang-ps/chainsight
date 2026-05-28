import tempfile
import unittest
from pathlib import Path

import pandas as pd

import src.services.summary_report_generator as summary_module
from src.services.summary_report_generator import SummaryReportGenerator


class LocalSummaryLargeOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        (self.base / "module1").mkdir()
        (self.base / "module4").mkdir()
        (self.base / "module5").mkdir()
        (self.base / "module6").mkdir()
        (self.base / "orchestrator").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _generator(self, limit=1_000_000):
        generator = SummaryReportGenerator(str(self.base), config_dict={})
        generator.end_date = pd.to_datetime("2025-01-31")
        generator.LOCAL_SUMMARY_EXCEL_ROW_LIMIT = limit
        return generator

    def _write_module1_file(self, date_key: str, orders, shipments=None, cuts=None):
        path = self.base / "module1" / f"module1_output_{date_key}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame(orders).to_excel(writer, sheet_name="OrderLog", index=False)
            pd.DataFrame(shipments or []).to_excel(writer, sheet_name="ShipmentLog", index=False)
            pd.DataFrame(cuts or []).to_excel(writer, sheet_name="CutLog", index=False)
        return str(path)

    def test_order_summary_small_data_stays_xlsx_and_avoids_concat(self):
        file_path = self._write_module1_file(
            "20250105",
            orders=[
                {"date": "2025-01-05", "material": "M1", "location": "L1", "quantity": 10, "demand_type": "A"},
                {"date": "2025-01-05", "material": "M1", "location": "L1", "quantity": 10, "demand_type": "A"},
                {"date": "2025-02-01", "material": "M1", "location": "L1", "quantity": 99, "demand_type": "A"},
            ],
            shipments=[
                {"date": "2025-01-05", "material": "M1", "location": "L1", "quantity": 4},
            ],
            cuts=[
                {"date": "2025-01-06", "material": "M2", "location": "L2", "quantity": 8},
            ],
        )

        generator = self._generator()
        original_concat = summary_module.pd.concat

        def fail_concat(*args, **kwargs):
            raise AssertionError("order summary should not concatenate raw daily logs")

        summary_module.pd.concat = fail_concat
        try:
            output_path = generator._generate_order_shipment_report({"module1": [file_path]})
        finally:
            summary_module.pd.concat = original_concat

        self.assertTrue(output_path.endswith(".xlsx"))
        out = pd.read_excel(output_path)
        out = out.sort_values(["date", "material", "location"]).reset_index(drop=True)

        self.assertEqual(len(out), 2)
        self.assertEqual(int(out.loc[0, "order_qty"]), 10)
        self.assertEqual(int(out.loc[0, "shipment_qty"]), 4)
        self.assertEqual(int(out.loc[0, "cut_qty"]), 6)
        self.assertEqual(out.loc[1, "material"], "M2")
        self.assertEqual(int(out.loc[1, "cut_qty"]), 0)

    def test_order_summary_uses_csv_when_input_rows_exceed_limit(self):
        file_path = self._write_module1_file(
            "20250105",
            orders=[
                {"date": "2025-01-05", "material": "M1", "location": "L1", "quantity": 1},
                {"date": "2025-01-05", "material": "M1", "location": "L1", "quantity": 2},
                {"date": "2025-01-05", "material": "M1", "location": "L1", "quantity": 3},
            ],
        )

        generator = self._generator(limit=2)
        output_path = generator._generate_order_shipment_report({"module1": [file_path]})

        self.assertTrue(output_path.endswith(".csv"))
        self.assertFalse((self.base / "summary" / "full_order_shipment_cut_report.xlsx").exists())
        out = pd.read_csv(output_path)
        self.assertEqual(len(out), 1)
        self.assertEqual(int(out.loc[0, "order_qty"]), 6)

    def test_tabular_summary_uses_streaming_csv_when_input_rows_exceed_limit(self):
        file_path = self.base / "module6" / "Module6Output_20250105.xlsx"
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            pd.DataFrame([
                {"date": "2025-01-05", "sending": "S1", "receiving": "R1", "truck_type": "T", "used_trucks": 1},
                {"date": "2025-01-06", "sending": "S1", "receiving": "R1", "truck_type": "T", "used_trucks": 2},
                {"date": "2025-01-07", "sending": "S1", "receiving": "R1", "truck_type": "T", "used_trucks": 3},
            ]).to_excel(writer, sheet_name="TruckUsageLog", index=False)

        generator = self._generator(limit=2)
        output_path = generator._generate_truck_usage_report({"module6": [str(file_path)]})

        self.assertTrue(output_path.endswith(".csv"))
        self.assertFalse((self.base / "summary" / "full_truck_usage_report.xlsx").exists())
        out = pd.read_csv(output_path)
        self.assertEqual(len(out), 3)
        self.assertEqual(int(out["used_trucks"].sum()), 6)


if __name__ == "__main__":
    unittest.main()
