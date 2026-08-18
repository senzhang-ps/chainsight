"""统一配置读取器 — 以 model 层为权威，model 驱动读取 + 列标准化。

设计原则：
- **model 权威**：每个 sheet 经 ``CONFIG_TABLE_REGISTRY`` 映射到 SA 模型类，
  按 model 列投影、rename（``Column.info["local_name"]`` → db 列名）、丢弃非
  model 列。返回的列名一律为小写 db 列名。
- **CSV 覆盖 Excel**：``config/`` 下同名 CSV（大小写不敏感）覆盖 Excel sheet；
  registry 中有同名 CSV 但 Excel 没有的表也作为「扩展表」纳入。
- **不做业务清洗**：不在读取层做 normalize_identifiers / M4 去重 / M4 别名映射 /
  校验——读取层只负责「按 model 取列 + 小写化」，保持纯净。业务清洗由下游模块
  自行调用 ``normalize_identifiers``。

用法：
    reader = ConfigReader(config_dir)
    all_config = reader.load_all()
    m1_config = reader.load_module("M1")
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..utils.normalization import normalize_identifiers

if TYPE_CHECKING:
    from ..core.run.config_dir import ConfigDir

logger = logging.getLogger(__name__)


class ConfigReader:
    """model 驱动的统一配置读取器。

    Args:
        config: ConfigDir 实例（首选）；或 Excel 路径字符串 / Path（向后兼容，
            内部经 ``ConfigDir.from_excel_path`` 构造）。
    """

    def __init__(self, config: Any):
        self._config = config
        self._cache: dict[str, pd.DataFrame] | None = None

    def load_all(self) -> dict[str, pd.DataFrame]:
        """按 model 注册表加载所有映射 sheet，列标准化为小写 db 列名。

        流程：
        1. 构造 ``ConfigDir``（若传入的是路径）。
        2. 打开 Excel，锁定 sheet 名快照。
        3. 候选 sheet = Excel sheet ∪ registry 中有同名 CSV 的 sheet。
        4. 逐 sheet 取源（CSV 优先，回退 Excel），经 model 投影后写入结果。
        5. 未映射到 model 的 sheet 跳过（``ignore_unmapped_sheets``）。

        Returns:
            ``{registry_sheet_name: DataFrame}``，列名为小写 model 列。
        """
        if self._cache is not None:
            return dict(self._cache)

        from ..core.run.config_dir import ConfigDir
        from ..models.cfg import CONFIG_TABLE_REGISTRY

        config = self._config
        if isinstance(config, ConfigDir) or hasattr(config, "excel_path"):
            cfg_dir = config
        else:
            cfg_dir = ConfigDir.from_excel_path(config)

        # 大小写不敏感的 sheet → (规范名, 模型类) 查找表
        registry_lower: dict[str, tuple[str, type]] = {
            s.lower(): (s, m) for s, m in CONFIG_TABLE_REGISTRY.items()
        }

        xl = pd.ExcelFile(str(cfg_dir.excel_path))
        excel_lower: dict[str, str] = {s.lower(): s for s in xl.sheet_names}

        # 候选 sheet（lower）：Excel 中的 + registry 中有同名 CSV 的
        candidate_lower: set[str] = set(excel_lower.keys())
        for sheet_name in CONFIG_TABLE_REGISTRY:
            if cfg_dir.csv_for_sheet(sheet_name) is not None:
                candidate_lower.add(sheet_name.lower())

        config_dict: dict[str, pd.DataFrame] = {}
        for low in sorted(candidate_lower):
            reg = registry_lower.get(low)
            sheet_label = reg[0] if reg else excel_lower.get(low, low)

            # 取源：CSV 优先，回退 Excel
            csv_path = cfg_dir.csv_for_sheet(sheet_label)
            if csv_path is not None:
                df = pd.read_csv(csv_path, float_precision="round_trip")
            elif low in excel_lower:
                df = xl.parse(excel_lower[low])
            else:
                continue

            # 仅保留 model 映射的 sheet（未映射 → 跳过）
            if reg is None:
                logger.debug(f"ConfigReader: sheet '{sheet_label}' 无 model 映射，跳过")
                continue

            df = self._project_to_model(df, reg[1])
            # ``M4_MaterialLocationLineCfg`` 同时由 M4 和 M5 消费。Excel 中
            # 工厂地点常以数值单元格保存（例如 ``386``），而 Global_Network
            # 使用四位业务键 ``0386``。M5 的 PTF/LSK 查找按该地点键关联；若
            # 此处不统一，文件配置路径会静默漏掉 PTF，缩短 route horizon。
            # 这是配置输入的标识符契约，不属于模块业务清洗。
            if reg[0] == "M4_MaterialLocationLineCfg":
                df = normalize_identifiers(df)
            config_dict[reg[0]] = df  # 用 registry 规范名作 key

        self._cache = config_dict
        logger.info(f"ConfigReader: 已加载 {len(config_dict)} 张配置表")
        return dict(self._cache)

    @staticmethod
    def _project_to_model(df: pd.DataFrame, model_cls: type) -> pd.DataFrame:
        """按 model 列投影 + rename + 按列类型定型。

        - 表头映射：``Column.info["local_name"]``（无则等于列名）→ db 列名。
        - df 列大小写不敏感匹配 local_name，rename 成 db 列。
        - 仅保留源里实际存在的 model 列（不补缺列、不填默认值）。
        - 按 model 列的 SA 类型给 pandas 列定型（Text→string、Integer→Int64、
          Float/Numeric→float、DateTime→datetime），使下游 join/计算类型一致。
          转换不了的值用 ``errors='coerce'`` 归为缺失值，不抛异常——具体脏值由
          DQ 反馈，不在读取层阻断。
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

        header_map: dict[str, str] = {}
        model_cols: set[str] = set()
        # db 列名 → SA Column（用于按类型定型）
        col_objs: dict[str, Any] = {}
        for col in model_cls.__table__.columns:
            info = col.info or {}
            local = str(info.get("local_name", col.name)).strip().lower()
            header_map[local] = col.name
            model_cols.add(col.name)
            col_objs[col.name] = col

        rename: dict[Any, str] = {}
        for c in df.columns:
            key = str(c).strip().lower()
            if key in header_map:
                rename[c] = header_map[key]
        df = df.rename(columns=rename)

        keep = [c for c in df.columns if c in model_cols]
        dropped = [c for c in df.columns if c not in model_cols]
        if dropped:
            logger.debug(
                f"ConfigReader: {model_cls.__name__} 丢弃非 model 列 {dropped}"
            )
        result = df[keep].copy() if keep else df.iloc[:, :0].copy()

        # 按 model 列类型定型
        for c in list(result.columns):
            result[c] = ConfigReader._coerce_to_model_dtype(result[c], col_objs[c])

        # 丢弃全空行：Excel 稀疏布局会产生垃圾空行（散落文本在非 model 列已被投影
        # 丢弃，行变为全 NaN）。这类行写入有 NOT NULL/主键约束的表会失败。
        if not result.empty:
            before = len(result)
            result = result.dropna(how="all")
            if len(result) < before:
                logger.debug(
                    f"ConfigReader: {model_cls.__name__} 丢弃 {before - len(result)} 个全空行"
                )
        return result

    @staticmethod
    def _coerce_to_model_dtype(series: pd.Series, col: Any) -> pd.Series:
        """按 SA Column 类型把 series 转为对应 pandas dtype。

        - Text/字符串 → 可空 string（保留 <NA>，数字型地点 99 → "99"）
        - BigInteger/Integer → 可空 Int64
        - Float/Numeric → float64
        - DateTime → datetime64[ns]
        - 其余类型原样返回（不强行转换）
        """
        from sqlalchemy import (
            BigInteger, DateTime, Float, Integer, Numeric, Text,
        )
        sa_type = type(col.type)
        try:
            if sa_type is Text or isinstance(col.type, Text):
                return series.astype("string")
            if sa_type in (BigInteger, Integer) or isinstance(col.type, (BigInteger, Integer)):
                return pd.to_numeric(series, errors="coerce").astype("Int64")
            if sa_type in (Float, Numeric) or isinstance(col.type, (Float, Numeric)):
                return pd.to_numeric(series, errors="coerce")
            if sa_type is DateTime or isinstance(col.type, DateTime):
                return pd.to_datetime(series, errors="coerce")
        except Exception as e:  # 转换失败原样返回，交给 DQ 反馈
            logger.debug(f"ConfigReader: 列 {col.name} 类型转换失败({e})，保留原值")
        return series

    def load_module(self, module: str) -> dict[str, pd.DataFrame]:
        """只加载某个模块需要的 sheet（按 cfg 注册表的 module 过滤）。

        模块标签来自 sheet 名前缀（如 "M1_DemandForecast" → module="M1"）。

        Args:
            module: 模块标识前缀，如 "M1"、"M4"、"M6"。

        Returns:
            该模块关联的配置表 dict {sheet_name: DataFrame}。
        """
        all_config = self.load_all()

        from ..models.cfg import CONFIG_TABLE_REGISTRY
        module_prefix = module.upper() + "_"
        module_sheets = {
            sheet for sheet in CONFIG_TABLE_REGISTRY
            if sheet.startswith(module_prefix)
        }

        return {
            sheet: df for sheet, df in all_config.items() if sheet in module_sheets
        }

    def get_sheet(self, sheet_name: str) -> pd.DataFrame | None:
        """从缓存中获取单个 sheet 的 DataFrame。"""
        all_config = self.load_all()
        return all_config.get(sheet_name)


__all__ = ["ConfigReader"]
