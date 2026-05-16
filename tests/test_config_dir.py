"""ConfigDir 单元测试。

覆盖矩阵详见 ``.claude/IO配置绝对路径读取_实施计划.md`` §2。
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.core.run.config_dir import ConfigDir


def _write_xlsx(path: Path, sheets: dict[str, pd.DataFrame] | None = None) -> Path:
    """造一个最小可读的 xlsx。默认建一个 ``Sheet1`` 空表。"""
    sheets = sheets or {"Sheet1": pd.DataFrame({"a": [1]})}
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
    return path


def _write_csv(path: Path, df: pd.DataFrame | None = None) -> Path:
    df = df if df is not None else pd.DataFrame({"a": [1, 2]})
    df.to_csv(path, index=False)
    return path


class ConfigDirFromPathTests(unittest.TestCase):
    """正常路径与基本边界。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ---- 正常 ----
    def test_normal_one_xlsx_two_csv(self):
        _write_xlsx(self.tmp / "config.xlsx")
        _write_csv(self.tmp / "Sheet_A.csv")
        _write_csv(self.tmp / "Sheet_B.csv")

        cfg = ConfigDir.from_path(self.tmp)

        self.assertEqual(cfg.excel_path.name, "config.xlsx")
        self.assertEqual(set(cfg.csv_map.keys()), {"sheet_a", "sheet_b"})
        # key 全小写
        for k in cfg.csv_map:
            self.assertEqual(k, k.lower())

    # ---- 目录不存在 ----
    def test_directory_not_found(self):
        missing = self.tmp / "does_not_exist"
        with self.assertRaises(FileNotFoundError) as ctx:
            ConfigDir.from_path(missing)
        self.assertIn(str(missing), str(ctx.exception))

    # ---- 无 Excel ----
    def test_no_excel(self):
        _write_csv(self.tmp / "Sheet_A.csv")
        with self.assertRaises(FileNotFoundError) as ctx:
            ConfigDir.from_path(self.tmp)
        self.assertIn(str(self.tmp), str(ctx.exception))
        self.assertIn("未检测到 Excel", str(ctx.exception))

    # ---- 多个 Excel ----
    def test_multiple_excel(self):
        _write_xlsx(self.tmp / "a.xlsx")
        _write_xlsx(self.tmp / "b.xlsx")
        with self.assertRaises(ValueError) as ctx:
            ConfigDir.from_path(self.tmp)
        msg = str(ctx.exception)
        self.assertIn("a.xlsx", msg)
        self.assertIn("b.xlsx", msg)
        self.assertIn("唯一性", msg)

    # ---- 相对路径 ----
    def test_relative_path_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ConfigDir.from_path("../config")
        self.assertIn("绝对路径", str(ctx.exception))

    # ---- 绝对路径（Windows / POSIX 都用 tmp 路径覆盖） ----
    def test_absolute_path_accepted(self):
        _write_xlsx(self.tmp / "config.xlsx")
        abs_path = self.tmp  # tempfile 给的是绝对路径
        self.assertTrue(abs_path.is_absolute())
        cfg = ConfigDir.from_path(abs_path)
        self.assertEqual(cfg.dir_path, abs_path)

    @unittest.skipUnless(sys.platform == "win32", "Windows 盘符场景")
    def test_absolute_path_windows_with_backslash(self):
        _write_xlsx(self.tmp / "config.xlsx")
        # 传一个反斜杠形式的绝对路径
        abs_str = str(self.tmp).replace("/", "\\")
        cfg = ConfigDir.from_path(abs_str)
        self.assertTrue(cfg.dir_path.is_absolute())

    # ---- 大小写匹配 ----
    def test_case_insensitive_csv_match(self):
        _write_xlsx(self.tmp / "config.xlsx", {"Sheet_A": pd.DataFrame({"a": [1]})})
        _write_csv(self.tmp / "sheet_a.csv")
        cfg = ConfigDir.from_path(self.tmp)
        # 用 Excel 的大小写 sheet 名查 CSV，应命中
        self.assertIsNotNone(cfg.csv_for_sheet("Sheet_A"))
        self.assertEqual(cfg.csv_for_sheet("Sheet_A").name, "sheet_a.csv")
        # 不存在的查不到
        self.assertIsNone(cfg.csv_for_sheet("NotThere"))

    # ---- CSV 多于 Excel sheet ----
    def test_extra_csv_not_in_excel(self):
        _write_xlsx(self.tmp / "config.xlsx", {"Sheet1": pd.DataFrame({"a": [1]})})
        _write_csv(self.tmp / "Extra.csv")
        cfg = ConfigDir.from_path(self.tmp)
        self.assertIn("extra", cfg.csv_map)

    # ---- 忽略锁文件 ----
    def test_ignores_excel_lock_file(self):
        _write_xlsx(self.tmp / "config.xlsx")
        (self.tmp / "~$config.xlsx").write_bytes(b"\x00\x01")  # 模拟 Excel 锁
        # 不应触发"多个 Excel"
        cfg = ConfigDir.from_path(self.tmp)
        self.assertEqual(cfg.excel_path.name, "config.xlsx")


class ConfigDirCsvUniquenessTests(unittest.TestCase):
    """CSV 同级唯一性（大小写不敏感）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = Path(self._tmp.name).resolve()
        _write_xlsx(self.tmp / "config.xlsx")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @unittest.skipIf(sys.platform == "win32",
                     "Windows 文件系统大小写不敏感，无法在同一目录同时创建 Sheet_A.csv 与 sheet_a.csv")
    def test_case_insensitive_duplicate_raises(self):
        _write_csv(self.tmp / "Sheet_A.csv")
        _write_csv(self.tmp / "sheet_a.csv")
        with self.assertRaises(ValueError) as ctx:
            ConfigDir.from_path(self.tmp)
        msg = str(ctx.exception)
        self.assertIn("Sheet_A.csv", msg)
        self.assertIn("sheet_a.csv", msg)
        self.assertIn("sheet_a", msg)  # 冲突 key
        self.assertIn("唯一", msg)

    @unittest.skipIf(sys.platform == "win32",
                     "Windows 文件系统大小写不敏感，无法在同一目录创建多个大小写变体")
    def test_multi_variant_duplicate_raises(self):
        _write_csv(self.tmp / "A.csv")
        _write_csv(self.tmp / "a.csv")
        # 注意：.CSV 与 .csv 在大小写敏感 FS 上是两个不同的扩展名实例，
        # 但 _collect_csvs 用 suffix.lower() 匹配，会一并算成 .csv 然后撞 stem。
        _write_csv(self.tmp / "A.CSV")
        with self.assertRaises(ValueError) as ctx:
            ConfigDir.from_path(self.tmp)
        msg = str(ctx.exception)
        self.assertIn("A.csv", msg)
        self.assertIn("a.csv", msg)
        self.assertIn("A.CSV", msg)


class ConfigDirPathIsolationTests(unittest.TestCase):
    """路径独立性：B 目录构造的 ConfigDir 绝不读到 A 目录的 CSV。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self._tmp.name).resolve()
        self.dir_a = root / "A"
        self.dir_b = root / "B"
        self.dir_a.mkdir()
        self.dir_b.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_does_not_read_sibling_csv(self):
        # A 目录有 Sheet_X.csv，B 目录只有 Excel
        _write_xlsx(self.dir_a / "config.xlsx")
        _write_csv(self.dir_a / "Sheet_X.csv")
        _write_xlsx(self.dir_b / "config.xlsx")

        cfg_b = ConfigDir.from_path(self.dir_b)
        # B 构造出的 ConfigDir 不应包含来自 A 的 CSV
        self.assertEqual(cfg_b.csv_map, {})
        self.assertIsNone(cfg_b.csv_for_sheet("Sheet_X"))

    def test_does_not_recurse_subdirectories(self):
        # 在 B 下放一个子目录、子目录内放 CSV——绝不能被收
        _write_xlsx(self.dir_b / "config.xlsx")
        sub = self.dir_b / "extras"
        sub.mkdir()
        _write_csv(sub / "Sheet_Y.csv")

        cfg_b = ConfigDir.from_path(self.dir_b)
        self.assertEqual(cfg_b.csv_map, {})


class ConfigDirFromExcelPathTests(unittest.TestCase):
    """from_excel_path：宽松 Excel、严格 CSV。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_loose_excel_uniqueness(self):
        # 目录里 2 个 xlsx，指定其中一个——不报错
        _write_xlsx(self.tmp / "a.xlsx")
        _write_xlsx(self.tmp / "b.xlsx")
        cfg = ConfigDir.from_excel_path(self.tmp / "a.xlsx")
        self.assertEqual(cfg.excel_path.name, "a.xlsx")

    @unittest.skipIf(sys.platform == "win32",
                     "Windows 文件系统大小写不敏感")
    def test_strict_csv_uniqueness(self):
        _write_xlsx(self.tmp / "a.xlsx")
        _write_csv(self.tmp / "Foo.csv")
        _write_csv(self.tmp / "foo.csv")
        with self.assertRaises(ValueError):
            ConfigDir.from_excel_path(self.tmp / "a.xlsx")

    def test_missing_excel(self):
        with self.assertRaises(FileNotFoundError):
            ConfigDir.from_excel_path(self.tmp / "missing.xlsx")


class ConfigDirDerivedPropertiesTests(unittest.TestCase):
    """project / scenario / output_subpath 派生属性。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        # 磁盘约定：3 段 ``<root>/<project>/<scenario>/config/***.xlsx``。
        # project = scenario 的父目录名；scenario = config 的父目录名；Excel 名任意。
        self.workspace = Path(self._tmp.name).resolve() / "workspace"
        self.cfg_dir = self.workspace / "SDC" / "baseline" / "config"
        self.cfg_dir.mkdir(parents=True)
        _write_xlsx(self.cfg_dir / "baseline.xlsx")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_project_scenario_output_subpath(self):
        cfg = ConfigDir.from_path(self.cfg_dir)
        # scenario 来自 config 父目录名
        self.assertEqual(cfg.scenario, "baseline")
        # project 来自 scenario 父目录名
        self.assertEqual(cfg.project, "SDC")
        # 二级输出子路径
        self.assertEqual(cfg.output_subpath, Path("SDC") / "baseline")


if __name__ == "__main__":
    unittest.main()
