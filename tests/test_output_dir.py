"""Output directory naming tests."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.run.config_dir import ConfigDir
from src.core.run.output_dir import _ensure_output_dir


class EnsureOutputDirTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name).resolve()
        # patch _PROJECT_ROOT → temp dir，使输出隔离，不写入真实仓库
        self._patcher = patch("src.core.run.output_dir._PROJECT_ROOT", self.root)
        self._patcher.start()
        # 3 段拓扑：project = "SDC"（scenario 父目录），scenario = "baseline"（config 父目录）
        self.cfg_dir = self.root / "workspace" / "SDC" / "baseline" / "config"
        self.cfg_dir.mkdir(parents=True)
        (self.cfg_dir / "baseline.xlsx").write_bytes(b"placeholder")
        self.cfg = ConfigDir.from_path(self.cfg_dir)

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()

    def test_config_dir_uses_project_scenario_subpath(self):
        run_dir = _ensure_output_dir(
            self.cfg,
            start_date="2026-01-01",
            end_date="2026-01-02",
            interactive=False,
        )

        # outputs/<project>/<scenario>/run_<ts>/
        self.assertTrue(run_dir.name.startswith("run_"))
        self.assertEqual(run_dir.parent.name, "baseline")       # scenario
        self.assertEqual(run_dir.parent.parent.name, "SDC")     # project
        self.assertEqual(run_dir.parent.parent.parent.name, "outputs")

    def test_resume_from_uses_project_scenario_subpath(self):
        existing = self.root / "outputs" / "SDC" / "baseline" / "run_existing"
        existing.mkdir(parents=True)

        selected = _ensure_output_dir(
            self.cfg,
            resume_from="run_existing",
            start_date="2026-01-01",
            end_date="2026-01-02",
            interactive=False,
        )

        self.assertEqual(selected, existing)

    def test_path_input_keeps_backward_compatible_stem(self):
        config_path = self.root / "legacy_config.xlsx"
        config_path.write_bytes(b"placeholder")

        run_dir = _ensure_output_dir(
            config_path,
            start_date="2026-01-01",
            end_date="2026-01-02",
            interactive=False,
        )

        # 旧式 Path 输入仍走单层 outputs/<stem>/run_*/
        self.assertEqual(run_dir.parent.name, "legacy_config")
        self.assertEqual(run_dir.parent.parent.name, "outputs")

    def test_run_suffix_produces_run_ts_db(self):
        run_dir = _ensure_output_dir(
            self.cfg,
            start_date="2026-01-01",
            end_date="2026-01-02",
            interactive=False,
            run_suffix="db",
        )

        self.assertTrue(run_dir.name.endswith("_db"))
        self.assertEqual(run_dir.parent.name, "baseline")
        self.assertEqual(run_dir.parent.parent.name, "SDC")


if __name__ == "__main__":
    unittest.main()
