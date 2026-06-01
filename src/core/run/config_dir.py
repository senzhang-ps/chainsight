"""config/ 目录解析：目录验证 + Excel 自动发现 + CSV 收集。

设计与约束详见 ``.claude/IO配置绝对路径读取_实施计划.md``。要点：

- ``--config-dir`` 给到的路径只到 ``config/`` 一层；目录内 Excel 必须**唯一**，
  同级 CSV 文件名（不区分大小写）必须**唯一**；任一不满足都先 warning 再 raise。
- 只扫一层（``Path.iterdir()``）——**绝不**回退到其他目录、**绝不**递归子目录。
- 本类**不预设 sheet 名清单**；CSV ↔ sheet 名的匹配权威由调用方持有的
  ``xl.sheet_names`` 快照决定（见 ``config_loader.load_configuration``）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

# 与 main_integration / utils 复用同名 logger，方便在 simulation_log_*.txt 里看到这些消息。
logger = logging.getLogger("SupplyChainSimulation")

_EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")


@dataclass(frozen=True)
class ConfigDir:
    """一个 config/ 目录的解析结果（不可变）。

    Attributes:
        dir_path:   config/ 目录绝对路径。
        excel_path: 目录内唯一的 Excel 文件绝对路径。
        csv_map:    ``{stem.lower(): Path}``——key 是 CSV 文件 stem 的小写形式，
                    用于 ``csv_for_sheet`` 做大小写不敏感查找。
    """

    dir_path: Path
    excel_path: Path
    csv_map: dict[str, Path] = field(default_factory=dict)

    # ---------- 工厂方法 ----------
    @classmethod
    def from_path(cls, abs_dir: str | Path) -> "ConfigDir":
        """对 config/ 目录做验证、发现唯一 Excel、收集 CSV；失败抛明确异常。

        严格约束（违反任一即先 ``logger.warning`` 再 ``raise``）：

        - 必须是绝对路径（``Path.is_absolute()``）。
        - 目录存在。
        - 目录内 ``.xlsx``/``.xlsm``/``.xls`` 文件**有且仅一个**（排除 ``~$`` 锁文件）。
        - 同级 ``.csv`` 文件**大小写不敏感唯一**（不允许 ``Sheet_A.csv`` 与
          ``sheet_a.csv`` 并存）。
        - 仅扫描 ``d.iterdir()`` 一层；不递归子目录、不回退到其他目录。
        """
        d = Path(abs_dir).expanduser()
        if not d.is_absolute():
            raise ValueError(f"--config-dir 须为绝对路径，实际收到：{abs_dir}")
        d = d.resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"config 目录不存在：{d}")
        _validate_project_scenario_layout(d)

        # ---- Excel 唯一性 ----
        xls_files = sorted(
            f for f in d.iterdir()
            if f.is_file() and f.suffix.lower() in _EXCEL_SUFFIXES
            and not f.name.startswith("~$")  # 跳过 Excel 打开时生成的锁文件
        )
        if len(xls_files) == 0:
            raise FileNotFoundError(f"config 目录中未检测到 Excel 文件：{d}")
        if len(xls_files) > 1:
            names = ", ".join(f.name for f in xls_files)
            msg = f"config 目录中存在多个 Excel 文件（{names}），须保证唯一性：{d}"
            logger.warning(msg)
            raise ValueError(msg)

        # ---- CSV 同级唯一性（大小写不敏感）----
        csv_map = _collect_csvs_or_raise(d, scope_label="config 目录")

        return cls(dir_path=d, excel_path=xls_files[0], csv_map=csv_map)

    @classmethod
    def from_excel_path(cls, excel_path: str | Path) -> "ConfigDir":
        """兼容旧路径：给一个 Excel 文件，按它所在目录构造 ConfigDir。

        与 ``from_path`` 的差异：仅放宽"Excel 唯一性"约束（旧 ``--config`` 流程
        已显式选定了某个 xlsx，同目录有其他 xlsx 也不报错）；CSV 同级唯一性约束
        与 ``from_path`` **保持一致**，避免行为模糊。
        """
        p = Path(excel_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"配置文件不存在：{p}")
        d = p.parent
        _validate_project_scenario_layout(d)
        csv_map = _collect_csvs_or_raise(d, scope_label="Excel 同目录")
        return cls(dir_path=d, excel_path=p, csv_map=csv_map)

    # ---------- 查询 ----------
    def csv_for_sheet(self, sheet_name: str) -> Path | None:
        """按 sheet 名（大小写不敏感）查同名 CSV；没有返回 ``None``。"""
        return self.csv_map.get(sheet_name.lower())

    # ---------- 派生属性 ----------
    @property
    def scenario(self) -> str:
        """场景名 = ``config/`` 的父目录名（即 scenario 目录）。

        支持两种磁盘拓扑，scenario 派生规则一致：

        - 4 级（旧）：``<input_root>/<project>/<scenario>/config/***.xlsx``
        - 5 级（生产）：``<input_root>/<project>/scenarios/<scenario>/config/***.xlsx``

        scenario 由目录名决定，与 Excel 文件名无关。
        """
        return self.dir_path.parent.name

    @property
    def project(self) -> str:
        """项目名：自动识别 ``scenarios/`` 中间层。

        - 4 级（旧）：``<input_root>/<project>/<scenario>/config/`` →
          ``project = dir_path.parent.parent.name``
        - 5 级（生产）：``<input_root>/<project>/scenarios/<scenario>/config/`` →
          当 ``config/`` 的祖父目录名恰好为字面 ``scenarios`` 时，上溯一层取
          ``dir_path.parent.parent.parent.name``，跳过 ``scenarios/`` 这层。
        """
        grandparent = self.dir_path.parent.parent
        if grandparent.name == "scenarios":
            return grandparent.parent.name
        return grandparent.name

    @property
    def output_subpath(self) -> Path:
        """输出二级子路径：``Path(<project>) / <scenario>``。

        用于在 ``outputs/`` 根下定位 ``outputs/<project>/<scenario>/`` 目录，
        所有重跑（文件模式 ``run_<ts>``、DB 模式 ``db_run_<ts>``）共享此目录。
        """
        return Path(self.project) / self.scenario

# ---------- 内部辅助 ----------
def _validate_project_scenario_layout(d: Path) -> None:
    """Reject ``<project>/scenarios/config`` because it is missing scenario."""
    if d.name != "config" or d.parent.name != "scenarios":
        return

    msg = (
        "Invalid config directory layout: expected "
        "<project>/scenarios/<scenario>/config, got "
        f"{d}. Missing <scenario> between 'scenarios' and 'config'."
    )
    logger.warning(msg)
    raise ValueError(msg)


def _collect_csvs_or_raise(d: Path, *, scope_label: str) -> dict[str, Path]:
    """收集 ``d`` 下的 ``.csv``，校验大小写不敏感唯一性。

    冲突时先 ``logger.warning`` 再 ``raise ValueError``，满足"先警告再终止"的
    双重语义。CLI 层捕获后会翻译为 ``[ConfigError]`` 并以非零退出码退出。
    """
    csv_by_key: dict[str, list[Path]] = {}
    for f in sorted(d.iterdir()):
        if f.is_file() and f.suffix.lower() == ".csv":
            csv_by_key.setdefault(f.stem.lower(), []).append(f)

    duplicates = {k: v for k, v in csv_by_key.items() if len(v) > 1}
    if duplicates:
        detail = "; ".join(
            f"{k} -> [{', '.join(p.name for p in paths)}]"
            for k, paths in duplicates.items()
        )
        msg = (
            f"{scope_label}中存在大小写不敏感重名的 CSV 文件（{detail}），"
            f"须保证每个 sheet 对应的 CSV 唯一：{d}"
        )
        logger.warning(msg)
        raise ValueError(msg)

    return {k: v[0] for k, v in csv_by_key.items()}
