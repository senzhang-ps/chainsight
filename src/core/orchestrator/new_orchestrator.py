import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from ..main_integration.config_loader import load_configuration
from ..main_integration.seed import set_module_seeds
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, start_date, end_date, config_path, output_path,
                 config_dict=None, engine='pandas'):
        self.start_date = start_date if isinstance(start_date, date) else pd.Timestamp(start_date).date()
        self.end_date = end_date if isinstance(end_date, date) else pd.Timestamp(end_date).date()
        self.config_path = config_path
        self.output_path = output_path
        self.module_idx = [1, 3, 4, 5, 6]
        self.all_config = load_configuration(self.config_path) if config_dict is None else config_dict
        set_module_seeds(self.all_config)
        self.build_output_folder()
        self.all_results = {}
        self.sim_dates = []
        self.engine = engine

    def load_params(self, module_config):
        return None

    def load_datas(self, module) -> None:
        """按 module.schema 加载数据，直接写入 module.datas。"""
        from ..modules.module import Module
        if not isinstance(module, Module):
            raise TypeError(f"load_datas 期望 Module 实例，收到 {type(module).__name__}")
        schema = getattr(module, 'schema', {})
        if not schema:
            return
        datas = {}
        for sheet_name, col_schema in schema.items():
            df = self.all_config.get(sheet_name, pd.DataFrame())
            if not df.empty:
                df = self._normalize_datas(df, sheet_name, col_schema)
            datas[sheet_name] = df
        module.datas = datas

    def get_module_config(self, module_config: str) -> dict:
        """从 all_config 中提取模块相关配置子集。默认返回空。"""
        return {}

    @staticmethod
    def _normalize_datas(df: pd.DataFrame, sheet_name: str, col_schema: dict) -> pd.DataFrame:
        """按 schema 校验并转换配置表的列名与类型。"""
        if df.empty:
            return df

        if col_schema is None:
            logger.warning("_normalize_datas: 未知 sheet '%s'，跳过校验", sheet_name)
            return df

        missing = [c for c in col_schema if c not in df.columns]
        if missing:
            raise ValueError(f"[{sheet_name}] 缺少必需列: {missing}")

        result = df[list(col_schema.keys())].copy()

        conversion_failures = {}
        for col, dtype in col_schema.items():
            series = result[col]

            if dtype == 'str':
                result[col] = series.astype(str).str.strip()

            elif dtype == 'int':
                numeric = pd.to_numeric(series, errors='coerce')
                bad = numeric.isna()
                if bad.any():
                    conversion_failures[col] = int(bad.sum())
                result[col] = numeric.fillna(0).astype(np.int64)

            elif dtype == 'float':
                numeric = pd.to_numeric(series, errors='coerce')
                bad = numeric.isna()
                if bad.any():
                    conversion_failures[col] = int(bad.sum())
                result[col] = numeric.fillna(0.0)

            elif dtype == 'datetime':
                converted = pd.to_datetime(series, errors='coerce')
                bad = converted.isna() & series.notna() & (series.astype(str).str.strip() != '')
                if bad.any():
                    conversion_failures[col] = int(bad.sum())
                result[col] = converted

        if conversion_failures:
            raise ValueError(
                f"[{sheet_name}] 以下列存在无法转换的值（已用默认值填充）: "
                + ", ".join(f"{col}({cnt}条)" for col, cnt in conversion_failures.items())
            )

        return result

    def iter_dates(self, actual_start_date=None):
        sim_start = actual_start_date or self.start_date
        sim_dates = pd.date_range(sim_start, self.end_date, freq='D')
        self.sim_dates = sim_dates
        logger.info(f"仿真日期范围: {len(sim_dates)} 天")

        pbar = tqdm(enumerate(sim_dates, 1), total=len(sim_dates), desc='仿真进度', unit='天', ncols=80, leave=True)
        for i, current_date in pbar:
            progress_info = f"第 {i}/{len(sim_dates)} 天"
            logger.info(f"{'=' * 20} {progress_info}: {current_date.strftime('%Y-%m-%d')} {'=' * 20}")
            pbar.set_postfix(date=current_date.strftime('%Y-%m-%d'), day=progress_info)
            yield i, current_date

    def build_output_folder(self):
        self._output_dirs = {}
        for i in self.module_idx:
            path = Path(self.output_path) / f'module{i}'
            path.mkdir(parents=True, exist_ok=True)
            self._output_dirs[f'module{i}'] = path

    def get_output(self, module_name: str) -> Path:
        if module_name not in self._output_dirs:
            raise KeyError(f"未找到模块输出路径: {module_name}，可选: {list(self._output_dirs.keys())}")
        return self._output_dirs[module_name]

