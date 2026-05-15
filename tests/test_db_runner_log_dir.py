"""Database mode log directory naming tests.

锁定 DB 模式日志目录命名规范：``outputs/<project>/<scenario>/db_run_<ts>/``，
确保目录前缀 ``db_run_`` 与 DB 中的 ``run_id``（``db_<config>_<ts>``）共享
``db_`` 前缀与同一时间戳，便于人工对账。
"""
from __future__ import annotations

import unittest
from pathlib import Path

from src.core.run.db_runner import _build_db_log_dir


class BuildDbLogDirTests(unittest.TestCase):
    def test_uses_db_run_prefix_with_two_level_subpath(self):
        log_dir = _build_db_log_dir(
            config_name="BC_S5",
            timestamp="20260514_221942",
            project_root=Path("/repo"),
            output_subpath=Path("BC") / "BC_S5",
        )

        self.assertEqual(
            log_dir,
            Path("/repo") / "outputs" / "BC" / "BC_S5" / "db_run_20260514_221942",
        )
        self.assertTrue(log_dir.name.startswith("db_run_"))
        self.assertFalse(log_dir.name.endswith("_db"))

    def test_falls_back_to_config_name_when_no_output_subpath(self):
        log_dir = _build_db_log_dir(
            config_name="BC_S5",
            timestamp="20260514_221942",
            project_root=Path("/repo"),
        )

        # 纯 DB 配置名（无本地 Excel）：fallback 到 <config>/<config>
        self.assertEqual(
            log_dir,
            Path("/repo") / "outputs" / "BC_S5" / "BC_S5" / "db_run_20260514_221942",
        )

    def test_dir_name_aligns_with_run_id_timestamp(self):
        ts = "20260514_221942"
        config_name = "BC_S5"

        log_dir = _build_db_log_dir(
            config_name=config_name,
            timestamp=ts,
            project_root=Path("/repo"),
            output_subpath=Path("BC") / "BC_S5",
        )
        # db_runner 内 effective_run_id 模板：f"db_{config_name}_{ts}"
        run_id = f"db_{config_name}_{ts}"

        # 共享 db_ 前缀
        self.assertTrue(log_dir.name.startswith("db_"))
        self.assertTrue(run_id.startswith("db_"))
        # 共享时间戳
        self.assertIn(ts, log_dir.name)
        self.assertIn(ts, run_id)


if __name__ == "__main__":
    unittest.main()
