"""load_configuration 新语义测试（接受 ConfigDir + CSV 无条件优先）。

测试矩阵详见 ``.claude/IO配置绝对路径读取_实施计划.md`` §3.5。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.core.main_integration.config_loader import load_configuration
from src.core.run.config_dir import ConfigDir


# 这些必需 sheet 在 _validate_config_dict 里检查，缺失会 warning 但不 raise；
# 在 fixture 里都补齐，避免日志噪音。
_REQUIRED = [
    "M1_InitialInventory", "Global_SpaceCapacity", "Global_Network",
    "Global_LeadTime", "Global_DemandPriority",
]


def _empty_required_sheets() -> dict[str, pd.DataFrame]:
    return {name: pd.DataFrame() for name in _REQUIRED}


class LoadConfigurationCsvPriorityTests(unittest.TestCase):
    def setUp(self) -> None:
        # ignore_cleanup_errors: Windows 上若仍有文件句柄滞留也不让 tearDown 失败
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_xlsx(self, sheets: dict[str, pd.DataFrame]) -> Path:
        path = self.tmp / "config.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            for name, df in sheets.items():
                df.to_excel(xw, sheet_name=name, index=False)
        return path

    # ---- 1. CSV 无条件优先：Excel 有 10 行，CSV 有 5 行 → 取 CSV ----
    def test_csv_overrides_non_empty_excel_sheet(self):
        excel_sheets = _empty_required_sheets() | {
            "Sheet_A": pd.DataFrame({"v": list(range(10))}),
        }
        self._write_xlsx(excel_sheets)
        # CSV 同名（大小写不敏感）
        (self.tmp / "Sheet_A.csv").write_text("v\n100\n200\n300\n400\n500\n",
                                              encoding="utf-8")

        cfg = ConfigDir.from_path(self.tmp)
        result = load_configuration(cfg)

        # 加载结果取自 CSV（5 行），不是 Excel（10 行）
        self.assertEqual(len(result["Sheet_A"]), 5)
        self.assertEqual(result["Sheet_A"]["v"].tolist(), [100, 200, 300, 400, 500])

    # ---- 2. 大小写不敏感匹配 ----
    def test_case_insensitive_match(self):
        excel_sheets = _empty_required_sheets() | {
            "M3_SafetyStock": pd.DataFrame({"qty": [1, 2, 3, 4, 5]}),
        }
        self._write_xlsx(excel_sheets)
        (self.tmp / "m3_safetystock.csv").write_text("qty\n10\n20\n30\n",
                                                    encoding="utf-8")

        cfg = ConfigDir.from_path(self.tmp)
        result = load_configuration(cfg)

        # CSV 命中（3 行），key 用 Excel 的大小写
        self.assertIn("M3_SafetyStock", result)
        self.assertEqual(len(result["M3_SafetyStock"]), 3)

    # ---- 3. CSV 无对应 Excel sheet → 作为新 sheet ----
    def test_csv_extension_no_matching_sheet(self):
        excel_sheets = _empty_required_sheets() | {
            "Sheet1": pd.DataFrame({"a": [1]}),
        }
        self._write_xlsx(excel_sheets)
        (self.tmp / "Extra_Sheet.csv").write_text("x\n1\n2\n", encoding="utf-8")

        cfg = ConfigDir.from_path(self.tmp)
        result = load_configuration(cfg)

        # CSV 文件原大小写 stem 作为 key
        self.assertIn("Extra_Sheet", result)
        self.assertEqual(len(result["Extra_Sheet"]), 2)

    # ---- 4. 无 CSV 的 sheet 来自 Excel ----
    def test_sheet_without_csv_comes_from_excel(self):
        excel_sheets = _empty_required_sheets() | {
            "OnlyInExcel": pd.DataFrame({"k": ["a", "b", "c"]}),
        }
        self._write_xlsx(excel_sheets)

        cfg = ConfigDir.from_path(self.tmp)
        result = load_configuration(cfg)

        self.assertEqual(result["OnlyInExcel"]["k"].tolist(), ["a", "b", "c"])

    # ---- 5. 向后兼容：传 str 路径仍能工作 ----
    def test_backward_compat_string_path(self):
        excel_sheets = _empty_required_sheets()
        excel_path = self._write_xlsx(excel_sheets)

        # 传字符串 → 内部走 ConfigDir.from_excel_path
        result = load_configuration(str(excel_path))

        # 至少必需 sheet 在结果里
        for s in _REQUIRED:
            self.assertIn(s, result)

    # ---- 6. sheet_names 快照不可变性（迭代中即使 xl 对象变化也不影响） ----
    def test_sheet_names_snapshot_is_tuple(self):
        # 这条用例只是一种行为契约：load_configuration 内部使用
        # tuple(xl.sheet_names) 快照——即便 ExcelFile 是 mock 出来的、
        # sheet_names 是一个可变 list，循环结果也不应受影响。
        # 这里通过"加载成功 + 必需 sheet 都在"间接验证不会出现迭代崩溃。
        excel_sheets = _empty_required_sheets() | {
            f"Sheet_{i}": pd.DataFrame({"v": [i]}) for i in range(5)
        }
        self._write_xlsx(excel_sheets)

        cfg = ConfigDir.from_path(self.tmp)
        result = load_configuration(cfg)

        for i in range(5):
            self.assertIn(f"Sheet_{i}", result)


if __name__ == "__main__":
    unittest.main()
