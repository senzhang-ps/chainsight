"""CLI config-dir parsing and workspace_root tests."""
from __future__ import annotations

import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.run import run_main
from src.core.run.config_dir import ConfigDir
from src.core.run import utils as run_utils


class RunMainArgumentTests(unittest.TestCase):
    def test_config_dir_and_config_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            run_main._parse_args([
                "--config-dir", "SDC/baseline",
                "--config", "SDC_V1",
                "--end-date", "2026-01-31",
            ])

    def test_config_source_is_required(self):
        with self.assertRaises(SystemExit):
            run_main._parse_args(["--end-date", "2026-01-31"])

    def test_config_dir_short_form_is_accepted_by_parser(self):
        ns = run_main._parse_args([
            "--config-dir", "SDC/baseline",
            "--start-date", "2026-01-01",
            "--end-date", "2026-01-31",
        ])

        self.assertEqual(ns.config_dir, "SDC/baseline")
        self.assertIsNone(ns.config)


class WorkspaceRootResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_env_has_highest_priority(self):
        env_root = self.root / "env_ws"
        with patch.object(run_utils, "_PROJECT_ROOT", self.root), patch.dict(
            os.environ, {"CHAINSIGHT_WORKSPACE": str(env_root)}, clear=True
        ):
            self.assertEqual(run_utils.resolve_workspace_root(), env_root.resolve())

    def test_dotenv_overrides_defaults_yaml(self):
        config_dir = self.root / "config"
        config_dir.mkdir()
        (config_dir / "defaults.yaml").write_text(
            "workspace_root: ./yaml_ws\n", encoding="utf-8"
        )
        (self.root / ".env").write_text(
            "CHAINSIGHT_WORKSPACE=./dotenv_ws\n", encoding="utf-8"
        )

        with patch.object(run_utils, "_PROJECT_ROOT", self.root), patch.dict(
            os.environ, {}, clear=True
        ):
            self.assertEqual(
                run_utils.resolve_workspace_root(),
                (self.root / "dotenv_ws").resolve(),
            )

    def test_defaults_yaml_used_before_fallback(self):
        config_dir = self.root / "config"
        config_dir.mkdir()
        (config_dir / "defaults.yaml").write_text(
            "workspace_root: ./yaml_ws\n", encoding="utf-8"
        )

        with patch.object(run_utils, "_PROJECT_ROOT", self.root), patch.dict(
            os.environ, {}, clear=True
        ):
            self.assertEqual(
                run_utils.resolve_workspace_root(),
                (self.root / "yaml_ws").resolve(),
            )

    def test_fallback_is_project_workspace(self):
        with patch.object(run_utils, "_PROJECT_ROOT", self.root), patch.dict(
            os.environ, {}, clear=True
        ):
            self.assertEqual(
                run_utils.resolve_workspace_root(),
                (self.root / "workspace").resolve(),
            )

    def test_expand_short_config_dir_with_workspace_root(self):
        ws = self.root / "workspace"
        with patch.object(run_utils, "_PROJECT_ROOT", self.root), patch.dict(
            os.environ, {"CHAINSIGHT_WORKSPACE": str(ws)}, clear=True
        ):
            self.assertEqual(
                run_utils.expand_config_dir_arg("SDC/baseline"),
                (ws / "SDC" / "baseline" / "config").resolve(),
            )

    def test_expand_rejects_invalid_short_form(self):
        with patch.object(run_utils, "_PROJECT_ROOT", self.root), patch.dict(
            os.environ, {}, clear=True
        ):
            with self.assertRaises(ValueError):
                run_utils.expand_config_dir_arg("SDC/baseline/config")


class RunMainConfigDirIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name).resolve()
        self.old_cwd = Path.cwd()
        os.chdir(self.root)
        # 3 段拓扑：project="SDC", scenario="baseline"（config 父目录名）
        self.cfg_dir = self.root / "workspace" / "SDC" / "baseline" / "config"
        self.cfg_dir.mkdir(parents=True)
        (self.cfg_dir / "baseline.xlsx").write_bytes(b"placeholder")

    def tearDown(self) -> None:
        os.chdir(self.old_cwd)
        self._tmp.cleanup()

    def test_main_passes_config_dir_to_output_dir_selection(self):
        captured = {}
        run_dir = self.root / "outputs" / "SDC" / "baseline" / "run_mock"
        run_dir.mkdir(parents=True)

        def fake_ensure_output_dir(config_source, **kwargs):
            captured["config_source"] = config_source
            return run_dir

        logger = logging.getLogger("test_run_main_config_dir")
        with patch.object(run_main, "load_configuration", return_value={}), \
             patch.object(run_main, "_ensure_output_dir", side_effect=fake_ensure_output_dir), \
             patch.object(run_main, "setup_logging", return_value=(logger, None)), \
             patch.object(
                 run_main,
                 "run_integrated_simulation",
                 return_value={"simulation_completed": True},
             ):
            code = run_main.main([
                "--config-dir", str(self.cfg_dir),
                "--start-date", "2026-01-01",
                "--end-date", "2026-01-02",
                "--non-interactive",
            ])

        self.assertEqual(code, 0)
        self.assertIsInstance(captured["config_source"], ConfigDir)
        self.assertEqual(
            captured["config_source"].output_subpath,
            Path("SDC") / "baseline",
        )

    def test_deprecated_config_accepts_directory_with_unique_excel(self):
        captured = {}
        run_dir = self.root / "outputs" / "SDC" / "baseline" / "run_mock"
        run_dir.mkdir(parents=True)

        def fake_ensure_output_dir(config_source, **kwargs):
            captured["config_source"] = config_source
            return run_dir

        logger = logging.getLogger("test_run_main_config_dir")
        with patch.object(run_main, "load_configuration", return_value={}), \
             patch.object(run_main, "_ensure_output_dir", side_effect=fake_ensure_output_dir), \
             patch.object(run_main, "setup_logging", return_value=(logger, None)), \
             patch.object(
                 run_main,
                 "run_integrated_simulation",
                 return_value={"simulation_completed": True},
             ):
            code = run_main.main([
                "--config", str(self.cfg_dir),
                "--start-date", "2026-01-01",
                "--end-date", "2026-01-02",
                "--non-interactive",
            ])

        self.assertEqual(code, 0)
        # --config 旧入口现在也传 ConfigDir，产生与 --config-dir 相同的二级输出结构
        self.assertIsInstance(captured["config_source"], ConfigDir)
        self.assertEqual(
            captured["config_source"].output_subpath,
            Path("SDC") / "baseline",
        )


if __name__ == "__main__":
    unittest.main()
