"""Database mode config path resolution tests."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.core.run import db_runner


def _write_xlsx(path: Path, sheets: dict[str, pd.DataFrame] | None = None) -> None:
    sheets = sheets or {"Sheet1": pd.DataFrame({"a": [1]})}
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


class DbRunnerConfigPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.cfg_dir = Path(self._tmp.name).resolve() / "config" / "sdc"
        self.cfg_dir.mkdir(parents=True)
        _write_xlsx(self.cfg_dir / "SDC.xlsx")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_directory_config_arg_resolves_unique_excel(self):
        self.assertEqual(
            db_runner._find_local_config_file(str(self.cfg_dir)),
            self.cfg_dir / "SDC.xlsx",
        )

    def test_directory_config_arg_uses_excel_stem_as_config_name(self):
        config_name, config_file = db_runner._resolve_db_config_input(str(self.cfg_dir))

        self.assertEqual(config_name, "SDC")
        self.assertEqual(config_file, self.cfg_dir / "SDC.xlsx")

    def test_expected_local_config_uses_csv_priority_for_non_empty_sheet(self):
        target_sheet = "M1_SupplyChoiceConfig"
        _write_xlsx(
            self.cfg_dir / "SDC.xlsx",
            {target_sheet: pd.DataFrame({"material": ["excel"], "qty": [1]})},
        )
        (self.cfg_dir / "M1_SupplyChoiceConfig.CSV").write_text(
            "material,qty\ncsv_a,10\ncsv_b,20\n",
            encoding="utf-8",
        )

        expected = db_runner._build_expected_local_config(str(self.cfg_dir))

        df = expected["m1_supplychoiceconfig"]
        self.assertEqual(len(df), 2)
        self.assertEqual(df["material"].tolist(), ["csv_a", "csv_b"])


if __name__ == "__main__":
    unittest.main()
