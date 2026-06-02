"""Module4 集成模式主入口 — ModuleFour 类。

本模块提供 M4 生产计划模块的面向对象封装：
- 无约束日计划构建（按 material/location/line 汇总，按 min_batch / rv 向上取整）

当前实现先使用 pandas 直接处理，后续再考虑抽 backends。
"""

import logging
from typing import Any, Optional

import pandas as pd
import numpy as np

from ..module import Module
from .constants import UNCONSTRAINED_PLAN_COLUMNS

logger = logging.getLogger("SupplyChainSimulation")


class ModuleFour(Module):
    """M4 生产计划模块 — 无约束日计划构建。

    Parameters
    ----------
    net_demand : pd.DataFrame, optional
        当日净需求（M3 输出），列含 ``material / location / requirement_date / quantity``。
    net_demand_path : str, optional
        净需求文件路径；当 ``net_demand`` 未提供时使用。
    """

    schema = {
        'M4_MaterialLocationLineCfg': {
            'material': 'str',
            'location': 'str',
            'delegate_line': 'str',
            'prd_rate': 'int',
            'min_batch': 'float',
            'rv': 'float',
            'ptf': 'int',
            'lsk': 'int',
            'day': 'int',
            'MCT': 'int',
        },
        'M4_LineCapacity': {
            'location': 'str',
            'line': 'str',
            'date': 'datetime',
            'capacity': 'int',
        },
        'M4_ChangeoverMatrix': {
            'from_material': 'str',
            'to_material': 'str',
            'changeover_id': 'str',
            'from line': 'str',
            'to line': 'str',
        },
        'M4_ChangeoverDefinition': {
            'changeover_id': 'str',
            'line': 'str',
            'time': 'float',
            'cost': 'float',
            'mu_loss': 'float',
        },
        'M4_ProductionReliability': {
            'location': 'str',
            'line': 'str',
            'pr': 'float',
        },
    }

    def __init__(self, simulation_date, simulation_start_date, output_dir='',
                 orchestrator=None, orch=None,
                 skip_file_output=False,
                 verbose=False, config=None,
                 net_demand: Optional[pd.DataFrame] = None,
                 net_demand_path: Optional[str] = None):
        super().__init__(simulation_date, orch, 'M4', verbose, config=config)
        self.legacy_orchestrator = orchestrator
        self.output_dir = output_dir
        self.skip_file_output = skip_file_output
        self.net_demand = net_demand
        self.net_demand_path = net_demand_path
        self.simulation_start_date = simulation_start_date
        self.unconstrained_plan: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # 步骤方法（TimedMeta 自动计时）
    # ------------------------------------------------------------------

    def load_net_demand(self) -> pd.DataFrame:
        """加载净需求（M3 输出）并对齐 production_runner 预处理：

        - 仅保留 ``layer == 0``（下游需求）；
        - ``quantity`` 取绝对值（M3 约定需求侧为负，M4 需要正值）；
        - ``material`` / ``location`` 转为字符串；
        - ``requirement_date`` 规范为 datetime。
        """
        if self.net_demand is not None:
            df = self.net_demand.copy()
        elif self.net_demand_path is not None:
            df = pd.read_excel(self.net_demand_path)
        else:
            return pd.DataFrame()

        if df.empty:
            return df

        if 'layer' in df.columns:
            df = df[df['layer'] == 0].copy()
        if 'quantity' in df.columns:
            df['quantity'] = df['quantity'].abs()
        if 'material' in df.columns:
            df['material'] = df['material'].astype(str)
        if 'location' in df.columns:
            df['location'] = df['location'].astype(str)
        if 'requirement_date' in df.columns:
            df['requirement_date'] = pd.to_datetime(df['requirement_date'])

        return df

    def build_unconstrained_plan(self) -> pd.DataFrame:
        """构建无约束日计划：

        1. 从 ``self.datas['M4_MaterialLocationLineCfg']`` 取 mlcfg；
        2. 按 ``(simulation_date - simulation_start_date)`` 计算 ``days_since_start``，
           用 :meth:`is_offset_review_day` 标记并筛选 review day；
        3. 与 net_demand（M3 输出）按 material/location 关联，按 ``requirement_date``
           过滤到 simulation_date；
        4. 按 material/location/line 汇总，并按 ``min_batch`` / ``rv`` 向上取整。
        """
        if 'M4_MaterialLocationLineCfg' not in self.datas:
            return pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS)

        mat_loc_line_config = self.datas['M4_MaterialLocationLineCfg'].copy()

        # 1) review day 标记 + 筛选
        days_since_start = (
            pd.Timestamp(self.simulation_date) - pd.Timestamp(self.simulation_start_date)
        ).days
        mat_loc_line_config['is_offset_review_day'] = mat_loc_line_config.apply(
            lambda r: self.is_offset_review_day(r, days_since_start), axis=1,
        )
        mat_loc_line_config = mat_loc_line_config[
            mat_loc_line_config['is_offset_review_day']
        ].drop(columns=['is_offset_review_day'])

        if mat_loc_line_config.empty:
            return pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS)

        # 2) 关联 net_demand
        net_demand = self.load_net_demand()
        if net_demand is None or net_demand.empty:
            return pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS)
        net_demand['material'] = net_demand['material'].astype('str')

        df = pd.merge(
            mat_loc_line_config, net_demand,
            how='inner', on=['material', 'location'],
        )
        if df.empty or 'requirement_date' not in df.columns:
            return pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS)

        # 3) 按 simulation_date 过滤
        df['requirement_date'] = pd.to_datetime(df['requirement_date']).dt.normalize()
        sim_date = pd.Timestamp(self.simulation_date).normalize()
        df = df[df['requirement_date'] == sim_date]
        if df.empty:
            return pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS)

        # 4) 按 material/location/line 汇总，向上取整
        grouped = df.groupby(['material', 'location', 'delegate_line'], as_index=False).agg(
            original_quantity=('quantity', 'sum'),
            min_batch=('min_batch', 'first'),
            rv=('rv', 'first'),
        )

        grouped['uncon_planned_qty'] = grouped.apply(
            lambda r: np.ceil(max(r['original_quantity'], r['min_batch']) / r['rv'].round()) * r['rv'].round()
                      if max(r['original_quantity'], r['min_batch']) % r['rv'].round() != 0
                      else int(max(r['original_quantity'], r['min_batch'])),
            axis=1,
        ).round().astype(int)

        grouped = grouped.rename(columns={'delegate_line': 'line'})
        grouped['planned_date'] = sim_date
        grouped['simulation_date'] = sim_date

        return grouped[UNCONSTRAINED_PLAN_COLUMNS]

    def is_offset_review_day(self, row, days_since_start: int) -> bool:
        """对齐 :func:`src.utils.date_helpers.is_offset_review_day` 的语义：
        判断当前 simulation_date 是否落在该 mlcfg 行的 review cycle 上。
        """
        first_review_day = int(row['day']) - 1
        is_on_cycle = (days_since_start - first_review_day) % int(row['lsk']) == 0
        is_after_first = days_since_start >= first_review_day
        return is_on_cycle and is_after_first

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def prepare(self):
        # 0) 数据加载（委托给 orchestrator）
        if self.orchestrator is not None:
            self.orchestrator.load_datas(self)

        # 1) 构建无约束日计划
        self.unconstrained_plan = self.build_unconstrained_plan()

    def run(self):
        try:
            unconstrained_plan = self.validate_data(
                self.unconstrained_plan,
                name='unconstrained_plan',
                numeric_columns=['uncon_planned_qty', 'original_quantity'],
                required_columns=['material', 'location', 'line', 'planned_date'],
            )

            self._result = {
                'unconstrained_plan': unconstrained_plan,
            }
        except Exception:
            import traceback
            traceback.print_exc()
            self._empty_result()

    # ------------------------------------------------------------------
    # 私有工具方法
    # ------------------------------------------------------------------

    def _empty_result(self):
        self._result = {
            'unconstrained_plan': pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS),
        }
