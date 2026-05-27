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

# M1 各 sheet 的列名与期望类型定义
_M1_SCHEMA = {
    'M1_DemandForecast': {
        'material': 'str',
        'location': 'str',
        'week': 'int',
        'quantity': 'float',
    },
    'M1_ForecastError': {
        'material': 'str',
        'location': 'str',
        'order_type': 'str',
        'error_std_percent': 'float',
    },
    'M1_OrderCalendar': {
        'date': 'datetime',
        'order_day_flag': 'int',
    },
    'M1_AOConfig': {
        'material': 'str',
        'location': 'str',
        'advance_days': 'int',
        'ao_percent': 'float',
    },
    'M1_DPSConfig': {
        'material': 'str',
        'location': 'str',
        'dps_location': 'str',
        'dps_percent': 'float',
    },
    'M1_SupplyChoiceConfig': {
        'material': 'str',
        'location': 'str',
        'week': 'int',
        'adjust_quantity': 'float',
    },
}

class Orchestrator:
    def __init__(self, start_date, end_date, config_path, output_path, config_dict=None):
        self.start_date = start_date if isinstance(start_date, date) else pd.Timestamp(start_date).date()
        self.end_date = end_date if isinstance(end_date, date) else pd.Timestamp(end_date).date()
        self.datas = None
        self.config_path = config_path
        self.output_path = output_path
        self.module_idx = [1, 3, 4, 5, 6]
        self.all_config = load_configuration(self.config_path) if config_dict is None else config_dict
        set_module_seeds(self.all_config)
        self.build_output_folder()
        self.all_results = {}
        self.sim_dates = []
        # data
        self.m1_demandforecast = None 
        self.m1_forecasterror = None
        self.m1_ordercalendar = None 
        self.m1_aoconfig = None 
        self.m1_dpsconfig = None
        self.m1_supplychoiceconfig = None 
        

    def load_params(self, module_config):
        if module_config == 'M1':
            return self._load_m1_params()

    def _load_m1_params(self):
        return None

    def load_datas(self, module_config):
        if module_config == 'M1':
            datas = self._load_m1_datas()
            self.m1_demandforecast = datas[0]
            self.m1_forecasterror = datas[1]
            self.m1_ordercalendar = datas[2]
            self.m1_aoconfig = datas[3]
            self.m1_dpsconfig = datas[4]
            self.m1_supplychoiceconfig = datas[5]

    def _load_m1_datas(self):
        dfs = []
        required_sheet = ['M1_DemandForecast',
                          'M1_ForecastError',
                          'M1_OrderCalendar',
                          'M1_AOConfig',
                          'M1_DPSConfig',
                          'M1_SupplyChoiceConfig']
        for sheet in required_sheet:
            df = self.all_config.get(sheet, pd.DataFrame())
            # if sheet!='M1_SupplyChoiceConfig' and df.empty:
            #     raise ValueError(f"缺少必需的配置数据：{sheet}")
            df = self._normalize_m1_datas(df, sheet)
            dfs.append(df)
        return dfs

    @staticmethod
    def _normalize_m1_datas(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """按 schema 校验并转换 M1 配置表的列名与类型。

        1. 检查必需列是否存在
        2. 按声明类型尝试转换
        3. 转换后检查是否产生新的 NaN（转换失败的值）
        4. 失败则报错，成功则只返回 schema 定义的列
        """
        if df.empty:
            return df

        schema = _M1_SCHEMA.get(sheet_name)
        if schema is None:
            logger.warning("_normalize_m1_datas: 未知 sheet '%s'，跳过校验", sheet_name)
            return df

        # 1) 检查必需列
        missing = [c for c in schema if c not in df.columns]
        if missing:
            raise ValueError(f"[{sheet_name}] 缺少必需列: {missing}")

        # 只保留 schema 定义的列
        result = df[list(schema.keys())].copy()

        # 2) 按类型逐列转换
        conversion_failures = {}
        for col, dtype in schema.items():
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

        # 3) 转换失败则报错
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

