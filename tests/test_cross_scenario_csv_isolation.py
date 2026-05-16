"""跨场景同名 CSV 误读防护测试。

锁定以下不变量（针对 ``input/BC/BC_S5/config/`` 与 ``input/sdc/sdc/config/`` 这种
同时存在同名 ``M3_SafetyStock`` CSV 的真实拓扑）：

- ``ConfigDir.from_path`` 只扫描传入目录一层；不会因为另一 scenario 也有同名 CSV
  而读到对方文件。
- ``ConfigDir.csv_for_sheet`` 在两个独立 ConfigDir 间互不影响；同名查找各自返回
  自己目录下的 CSV。
- 不同后缀大小写（``.CSV`` vs ``.csv``）不影响匹配，但两者**只能取本目录的那一份**。
- ``output_subpath`` 仍能正确派生为 ``<project>/<scenario>``，避免输出目录互相覆盖。

测试使用临时目录构造与生产相同的目录拓扑，不依赖真实 input/ 数据。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.core.run.config_dir import ConfigDir


def _write_xlsx(path: Path, sheets: dict[str, pd.DataFrame] | None = None) -> None:
    """写入一份最小 Excel；至少含一个 sheet（默认 ``Sheet1``）。"""
    sheets = sheets or {"Sheet1": pd.DataFrame({"a": [1]})}
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


class CrossScenarioCsvIsolationTests(unittest.TestCase):
    """场景 A 与场景 B 的同名 CSV 互不污染。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self._tmp.name).resolve()

        # 还原磁盘约定：<workspace>/<project>/<scenario>/config/{xlsx,csv}
        self.bc_cfg = root / "BC" / "BC_S5" / "config"
        self.sdc_cfg = root / "sdc" / "sdc" / "config"
        self.bc_cfg.mkdir(parents=True)
        self.sdc_cfg.mkdir(parents=True)

        # Excel：内容随意，仅为构造合法 ConfigDir。
        _write_xlsx(self.bc_cfg / "BC_S5.xlsx", {"M3_SafetyStock": pd.DataFrame()})
        _write_xlsx(self.sdc_cfg / "sdc.xlsx", {"M3_SafetyStock": pd.DataFrame()})

        # 关键：两份同名 CSV，schema 不同 + 后缀大小写不同（贴近真实生产数据）。
        self.bc_df = pd.DataFrame({
            "material": ["80784365"],
            "location": ["A668"],
            "date": ["2025-10-06"],
            "key": ["80784365A668"],            # BC 独有列
            "safety_stock_qty": [2372],
        })
        self.sdc_df = pd.DataFrame({
            "material": ["13250625"],
            "location": ["C816"],
            "date": ["2026-06-29"],
            "safety_stock_qty": [819.223647],
        })
        _write_csv(self.bc_cfg / "M3_SafetyStock.CSV", self.bc_df)   # 大写后缀
        _write_csv(self.sdc_cfg / "M3_SafetyStock.csv", self.sdc_df)  # 小写后缀

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ---- 1. ConfigDir 仅扫描自身目录 ----

    def test_each_configdir_collects_only_its_own_csv(self):
        bc = ConfigDir.from_path(self.bc_cfg)
        sdc = ConfigDir.from_path(self.sdc_cfg)

        self.assertEqual(set(bc.csv_map.keys()), {"m3_safetystock"})
        self.assertEqual(set(sdc.csv_map.keys()), {"m3_safetystock"})

        self.assertEqual(bc.csv_map["m3_safetystock"], self.bc_cfg / "M3_SafetyStock.CSV")
        self.assertEqual(sdc.csv_map["m3_safetystock"], self.sdc_cfg / "M3_SafetyStock.csv")

    # ---- 2. csv_for_sheet 返回各自目录下的 CSV ----

    def test_csv_for_sheet_returns_own_directory_file(self):
        bc = ConfigDir.from_path(self.bc_cfg)
        sdc = ConfigDir.from_path(self.sdc_cfg)

        bc_csv = bc.csv_for_sheet("M3_SafetyStock")
        sdc_csv = sdc.csv_for_sheet("M3_SafetyStock")

        self.assertIsNotNone(bc_csv)
        self.assertIsNotNone(sdc_csv)
        self.assertNotEqual(bc_csv, sdc_csv)
        # 父目录必须分别落在各自 scenario 下
        self.assertEqual(bc_csv.parent, self.bc_cfg)
        self.assertEqual(sdc_csv.parent, self.sdc_cfg)

    # ---- 3. 读出的内容与本目录文件一致，未发生交叉污染 ----

    def test_csv_content_matches_own_directory(self):
        bc = ConfigDir.from_path(self.bc_cfg)
        sdc = ConfigDir.from_path(self.sdc_cfg)

        bc_loaded = pd.read_csv(bc.csv_for_sheet("M3_SafetyStock"))
        sdc_loaded = pd.read_csv(sdc.csv_for_sheet("M3_SafetyStock"))

        # BC 多出 'key' 列；SDC 没有
        self.assertIn("key", bc_loaded.columns)
        self.assertNotIn("key", sdc_loaded.columns)

        # 列数严格匹配各自 schema
        self.assertEqual(list(bc_loaded.columns), list(self.bc_df.columns))
        self.assertEqual(list(sdc_loaded.columns), list(self.sdc_df.columns))

    # ---- 4. 后缀大小写不影响匹配 ----

    def test_uppercase_csv_extension_is_matched(self):
        # BC 用 .CSV（大写）也应被识别为 CSV 并参与匹配。
        bc = ConfigDir.from_path(self.bc_cfg)
        self.assertIsNotNone(bc.csv_for_sheet("m3_safetystock"))
        self.assertIsNotNone(bc.csv_for_sheet("M3_SAFETYSTOCK"))

    # ---- 5. output_subpath 隔离两个场景的输出 ----

    def test_output_subpath_is_per_scenario(self):
        bc = ConfigDir.from_path(self.bc_cfg)
        sdc = ConfigDir.from_path(self.sdc_cfg)

        self.assertEqual(bc.output_subpath, Path("BC") / "BC_S5")
        self.assertEqual(sdc.output_subpath, Path("sdc") / "sdc")
        # 关键：两条二级路径不重叠，确保 outputs/ 下不会互相覆盖
        self.assertNotEqual(bc.output_subpath, sdc.output_subpath)

    # ---- 6. ConfigDir 不递归子目录（防御性） ----

    def test_configdir_does_not_recurse_into_subdirs(self):
        # 在 BC config/ 下放一个看起来"诱人"的子目录，里面塞另一份同名 CSV。
        # ConfigDir 仍应只识别本目录那一份，不被子目录干扰。
        nested = self.bc_cfg / "archive"
        nested.mkdir()
        _write_csv(nested / "M3_SafetyStock.csv", pd.DataFrame({"poison": [1]}))

        bc = ConfigDir.from_path(self.bc_cfg)
        self.assertEqual(len(bc.csv_map), 1)
        self.assertEqual(bc.csv_map["m3_safetystock"].parent, self.bc_cfg)


class CrossScenarioConfigLoaderIsolationTests(unittest.TestCase):
    """``load_configuration`` 端到端：两份 ConfigDir 串行加载不互相串味。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self._tmp.name).resolve()
        self.bc_cfg = root / "BC" / "BC_S5" / "config"
        self.sdc_cfg = root / "sdc" / "sdc" / "config"
        self.bc_cfg.mkdir(parents=True)
        self.sdc_cfg.mkdir(parents=True)

        # Excel 含一个 M3_SafetyStock 空 sheet（CSV 无条件优先策略下 sheet 内容由 CSV 决定）。
        _write_xlsx(self.bc_cfg / "BC_S5.xlsx", {"M3_SafetyStock": pd.DataFrame()})
        _write_xlsx(self.sdc_cfg / "sdc.xlsx", {"M3_SafetyStock": pd.DataFrame()})

        self.bc_df = pd.DataFrame({
            "material": ["80784365"], "location": ["A668"],
            "date": ["2025-10-06"], "key": ["80784365A668"],
            "safety_stock_qty": [2372],
        })
        self.sdc_df = pd.DataFrame({
            "material": ["13250625"], "location": ["C816"],
            "date": ["2026-06-29"], "safety_stock_qty": [819.223647],
        })
        _write_csv(self.bc_cfg / "M3_SafetyStock.CSV", self.bc_df)
        _write_csv(self.sdc_cfg / "M3_SafetyStock.csv", self.sdc_df)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_load_configuration_picks_csv_from_each_scenario_independently(self):
        from src.core.main_integration.config_loader import load_configuration

        bc_cfg = ConfigDir.from_path(self.bc_cfg)
        sdc_cfg = ConfigDir.from_path(self.sdc_cfg)

        bc_dict = load_configuration(bc_cfg)
        sdc_dict = load_configuration(sdc_cfg)

        # BC 应拿到自己那份（含 key 列）
        self.assertIn("M3_SafetyStock", bc_dict)
        self.assertIn("key", bc_dict["M3_SafetyStock"].columns)
        # SDC 应拿到自己那份（不含 key 列）
        self.assertIn("M3_SafetyStock", sdc_dict)
        self.assertNotIn("key", sdc_dict["M3_SafetyStock"].columns)

    def test_reverse_load_order_does_not_leak_state(self):
        # 反向顺序加载，验证 load_configuration 无残留状态污染下一次调用。
        from src.core.main_integration.config_loader import load_configuration

        sdc_cfg = ConfigDir.from_path(self.sdc_cfg)
        bc_cfg = ConfigDir.from_path(self.bc_cfg)

        sdc_dict = load_configuration(sdc_cfg)
        bc_dict = load_configuration(bc_cfg)

        self.assertNotIn("key", sdc_dict["M3_SafetyStock"].columns)
        self.assertIn("key", bc_dict["M3_SafetyStock"].columns)


if __name__ == "__main__":
    unittest.main()
