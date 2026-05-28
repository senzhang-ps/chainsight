"""Backup main_integration CLI config-dir tests."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.main_integration import cli


class MainIntegrationCliConfigDirTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name).resolve()
        # 3 段拓扑：project="SDC", scenario="baseline"（config 父目录名）
        self.cfg_dir = self.root / "workspace" / "SDC" / "baseline" / "config"
        self.cfg_dir.mkdir(parents=True)
        (self.cfg_dir / "baseline.xlsx").write_bytes(b"placeholder")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_config_dir_resolves_excel_and_default_output_subpath(self):
        with patch.object(
            sys,
            "argv",
            [
                "cli",
                "--config-dir", str(self.cfg_dir),
                "--start-date", "2026-01-01",
                "--end-date", "2026-01-02",
            ],
        ), patch.object(
            cli,
            "run_integrated_simulation",
            return_value={"simulation_completed": True},
        ) as run_mock:
            cli.main()

        _, kwargs = run_mock.call_args
        self.assertEqual(kwargs["config_path"], str(self.cfg_dir / "baseline.xlsx"))
        # cli.py 新入口走二级 outputs/<project>/<scenario>/ 路径
        self.assertEqual(kwargs["output_base_dir"], str(Path("outputs") / "SDC" / "baseline"))


if __name__ == "__main__":
    unittest.main()
