"""``_write_run_id_file`` 落盘行为测试。

锁定以下不变量：
- 内容与传入 ``run_id`` 完全一致；无尾换行、无 BOM
- 覆盖写入幂等（续跑命中同一目录时不变）
- DB 模式 vs 本地模式只是文件名不同（``db_run_id.txt`` / ``run_id.txt``）
- 写入失败仅 ``logger.warning``，不抛 OSError
"""
from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.run.output_dir import _write_run_id_file


class WriteRunIdFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.log_dir = Path(self._tmp.name).resolve()
        self.logger = logging.getLogger("test_run_id_file")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # --- 内容/编码不变量 ---

    def test_writes_exact_content_no_trailing_newline(self):
        _write_run_id_file(
            self.log_dir, "db_BC_S5_20260514_221942", "db_run_id.txt", self.logger
        )

        raw = (self.log_dir / "db_run_id.txt").read_bytes()
        self.assertEqual(raw, b"db_BC_S5_20260514_221942")
        # 显式断言无尾换行 / 无回车
        self.assertFalse(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\r"))

    def test_no_utf8_bom(self):
        _write_run_id_file(
            self.log_dir, "run_20260514_221942", "run_id.txt", self.logger
        )

        raw = (self.log_dir / "run_id.txt").read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))

    # --- 文件名区分两种模式 ---

    def test_db_mode_filename(self):
        _write_run_id_file(
            self.log_dir, "db_BC_S5_20260514_221942", "db_run_id.txt", self.logger
        )

        self.assertTrue((self.log_dir / "db_run_id.txt").is_file())
        self.assertFalse((self.log_dir / "run_id.txt").exists())

    def test_local_mode_filename(self):
        _write_run_id_file(
            self.log_dir, "run_20260514_221942_test", "run_id.txt", self.logger
        )

        self.assertTrue((self.log_dir / "run_id.txt").is_file())
        self.assertFalse((self.log_dir / "db_run_id.txt").exists())

    # --- 覆盖写幂等 ---

    def test_overwrite_is_idempotent_for_same_run_id(self):
        target = self.log_dir / "run_id.txt"
        _write_run_id_file(self.log_dir, "run_20260514_221942", "run_id.txt", self.logger)
        first = target.read_bytes()

        _write_run_id_file(self.log_dir, "run_20260514_221942", "run_id.txt", self.logger)
        second = target.read_bytes()

        self.assertEqual(first, second)

    # --- 写入失败仅 warning ---

    def test_oserror_is_swallowed_and_logged(self):
        mock_logger = MagicMock()

        with patch(
            "pathlib.Path.write_text",
            side_effect=OSError("simulated disk failure"),
        ):
            # 不应抛
            _write_run_id_file(
                self.log_dir, "run_20260514_221942", "run_id.txt", mock_logger
            )

        mock_logger.warning.assert_called_once()
        (msg,) = mock_logger.warning.call_args.args
        self.assertIn("run_id.txt", msg)
        self.assertIn("simulated disk failure", msg)


if __name__ == "__main__":
    unittest.main()
