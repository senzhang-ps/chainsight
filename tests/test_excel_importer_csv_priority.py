"""ExcelImporter CSV priority tests."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pgsql_db.excel_importer import ExcelImporter


class FakeDatabase:
    def __init__(self) -> None:
        self.writes: dict[str, pd.DataFrame] = {}

    def create_table_from_df(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
        config_name: str | None = None,
        config_type: str | None = None,
    ) -> bool:
        captured = df.copy()
        if config_name:
            captured["config_name"] = config_name
        if config_type:
            captured["config_type"] = config_type
        self.writes[table_name] = captured
        return True


class ExcelImporterCsvPriorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.cfg_dir = Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_excel(self, sheets: dict[str, pd.DataFrame]) -> Path:
        path = self.cfg_dir / "config.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
        return path

    def test_import_excel_file_uses_csv_priority_for_non_empty_sheet(self):
        excel_path = self._write_excel({
            "M1_SupplyChoiceConfig": pd.DataFrame({
                "material": ["excel"],
                "qty": [1],
            })
        })
        (self.cfg_dir / "M1_SupplyChoiceConfig.CSV").write_text(
            "material,qty\ncsv_a,10\ncsv_b,20\n",
            encoding="utf-8",
        )
        db = FakeDatabase()

        results = ExcelImporter(db).import_excel_file(
            str(excel_path),
            config_name="config",
        )

        self.assertEqual(results["M1_SupplyChoiceConfig"], 2)
        written = db.writes["cfg_m1_supplychoiceconfig"]
        self.assertEqual(written["material"].tolist(), ["csv_a", "csv_b"])


if __name__ == "__main__":
    unittest.main()
