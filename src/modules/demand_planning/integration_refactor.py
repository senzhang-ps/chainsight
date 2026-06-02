"""Module1 集成模式主入口 — ModuleOne 类。

本模块提供 M1 需求规划模块的面向对象封装：
- 订单生成（含 AO/normal 拆分、误差应用）
- 发货与缺货计算
- 供需日志生成

通过 ``engine`` 参数支持 pandas / polars 两种计算后端（默认 pandas）。
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from ...utils.defaults import M1_FUTURE_CUTOFF_DAYS, DEFAULT_MAX_ADVANCE_DAYS
from ...utils.normalization import normalize_identifiers
from ..module import Module
from .backends import _PandasBackend, _PolarsBackend

logger = logging.getLogger("SupplyChainSimulation")


class ModuleOne(Module):
    """M1 需求规划模块 — 订单生成、发货与供需日志。

    Parameters
    ----------
    engine : str
        计算后端，从 orchestrator.engine 获取，``'pandas'`` 或 ``'polars'``。
    """

    schema = {
        'M1_DemandForecast': {
            'material': 'str', 'location': 'str', 'week': 'int', 'quantity': 'float',
        },
        'M1_ForecastError': {
            'material': 'str', 'location': 'str', 'order_type': 'str',
            'error_std_percent': 'float',
        },
        'M1_OrderCalendar': {
            'date': 'datetime', 'order_day_flag': 'int',
        },
        'M1_AOConfig': {
            'material': 'str', 'location': 'str', 'advance_days': 'int',
            'ao_percent': 'float',
        },
        'M1_DPSConfig': {
            'material': 'str', 'location': 'str', 'dps_location': 'str',
            'dps_percent': 'float',
        },
        'M1_SupplyChoiceConfig': {
            'material': 'str', 'location': 'str', 'week': 'int',
            'adjust_quantity': 'float',
        },
    }

    def __init__(self, simulation_date, output_dir='',
                 orchestrator=None, orch=None,
                 skip_file_output=False, previous_orders_df=None,
                 verbose=False, config=None):
        super().__init__(simulation_date, orch, 'M1', verbose, config=config)
        self.legacy_orchestrator = orchestrator
        self.output_dir = output_dir
        self.skip_file_output = skip_file_output
        self.previous_orders_df = previous_orders_df
        self.order_df = None

        if self._engine == 'polars':
            self._backend = _PolarsBackend(self)
        else:
            self._backend = _PandasBackend(self)

    # ------------------------------------------------------------------
    # 属性 — 保持向后兼容
    # ------------------------------------------------------------------

    @property
    def demand_forecast(self):
        return self._backend.demand_forecast

    @property
    def forecast_error(self):
        return self._backend.forecast_error

    @property
    def order_calendar(self):
        return self._backend.order_calendar

    @property
    def ao_config(self):
        return self._backend.ao_config

    @property
    def dps_config(self):
        return self._backend.dps_config

    @property
    def dps_sc_config(self):
        return self._backend.dps_sc_config

    # ------------------------------------------------------------------
    # 步骤方法（委托到 backend，TimedMeta 自动计时）
    # ------------------------------------------------------------------

    # 1
    def prepare_ao_summary(self):
        return self._backend.prepare_ao_summary()

    # 2
    def apply_dps_and_supply_choice(self):
        return self._backend.apply_dps_and_supply_choice()

    # 3
    def distribute_to_daily(self, demand_forecast_total):
        return self._backend.distribute_to_daily(demand_forecast_total)

    # 4
    def split_by_ao_and_apply_error(self, demand_forecast_total, ao_config_summary):
        return self._backend.split_by_ao_and_apply_error(demand_forecast_total, ao_config_summary)

    # 5
    def split_ao_by_advance_days(self, df_with_error, order_calendar):
        return self._backend.split_ao_by_advance_days(df_with_error, order_calendar)

    # 6
    def build_order_df(self, ao_detail):
        return self._backend.build_order_df(ao_detail)

    # ---- 新流程方法 ----

    def build_dps(self):
        return self._backend.build_dps()

    def build_cov(self, demand_total, ao_config_summary):
        return self._backend.build_cov(demand_total, ao_config_summary)

    def build_daily_order(self, demand_forecast, qty_col="quantity_total"):
        return self._backend.build_daily_order(demand_forecast, qty_col=qty_col)

    def adjust_daily_order_ao(self, daily_order):
        return self._backend.adjust_daily_order_ao(daily_order)

    def merge_with_history(self, today_orders_df):
        return self._backend.merge_with_history(today_orders_df)

    def apply_orders_consumption(self, orders_df):
        consumed = self._backend.apply_orders_consumption(self.daily_detail_sc, orders_df)
        self.daily_detail_sc = consumed
        return consumed

    def generate_supply_demand_log(self, consumed_forecast):
        return self._backend.generate_supply_demand_log(self.daily_detail_sc, consumed_forecast)

    def build_summary(self, orders_df, shipment_df, cut_df, supply_demand_df):
        return self._backend.build_summary(orders_df, shipment_df, cut_df, supply_demand_df)

    def generate_shipments(self, orders_df):
        return self._backend.generate_shipments(orders_df, self.daily_detail)

    def save_output(self, orders_df, shipment_df, cut_df, supply_demand_df, summary_df):
        return self._backend.save_output(orders_df, shipment_df, cut_df, supply_demand_df, summary_df)

    def get_order_day_flag(self, order_cal):
        return self._backend.get_order_day_flag(order_cal)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def prepare(self):
        # 0) 数据加载
        if self.orchestrator is not None:
            self.orchestrator.load_datas(self)

        # 1) AO 汇总
        ao_config, ao_config_summary = self.prepare_ao_summary()

        # 2) DPS 拆分（周级）
        demand_total, demand_total_sc = self.build_dps()

        # 3) AO 拆分 + 误差抽样（周级，产出 cov_quantity）
        df_with_cov = self.build_cov(demand_total, ao_config_summary)

        # 4) 日度拆分 — 订单流（关联 order_calendar，基于 cov_quantity）
        daily_order = self.build_daily_order(df_with_cov, qty_col="cov_quantity")

        # 5) AO detail 拆分 + 日期调整（关联 ao_detail，调整 advance_days）
        order_df = self.adjust_daily_order_ao(daily_order)

        # 6) 日度拆分 — consumption（直接拆 quantity_total，跳过 cov）
        daily_detail_sc = self.build_daily_order(demand_total_sc, qty_col="quantity_total")

        self.order_df = order_df
        self.daily_detail = daily_order
        self.daily_detail_sc = daily_detail_sc
        self.order_cal = self.order_calendar

    def run(self):
        try:
            order_cal = self.order_cal

            # 6) 合并历史订单（backend 内部按 simulation_date 筛选）
            all_orders, today_orders = self.merge_with_history(self.order_df)
            all_orders = self.validate_data(
                all_orders, name='orders',
                numeric_columns=['quantity', 'advance_days'],
                required_columns=['date', 'material', 'location'],
            )

            self.legacy_orchestrator.shipment_valid = int(self.get_order_day_flag(order_cal))

            # 7) 生成发货
            shipment_df, cut_df = self.generate_shipments(all_orders)
            shipment_df = self.validate_data(
                shipment_df, name='shipment',
                numeric_columns=['quantity'],
                required_columns=['date', 'material', 'location'],
            )

            # 8) 供需日志
            consumed = self.apply_orders_consumption(today_orders)
            supply_demand_df = self.generate_supply_demand_log(consumed)
            supply_demand_df = self.validate_data(
                supply_demand_df, name='supply_demand',
                numeric_columns=['quantity'],
                required_columns=['date', 'material', 'location'],
            )

            # 10) Summary
            summary_df = self.build_summary(all_orders, shipment_df, cut_df, supply_demand_df)

            # 9) 保存输出
            output_file = self.save_output(all_orders, shipment_df, cut_df, supply_demand_df, summary_df)


            # 统一输出为 pandas（外部消费者均为 pandas）
            self._result = {
                'orders_df': self._to_pandas(all_orders),
                'shipment_df': self._to_pandas(shipment_df),
                'cut_df': self._to_pandas(cut_df),
                'supply_demand_df': self._to_pandas(supply_demand_df),
                'summary_df': self._to_pandas(summary_df),
                'output_file': output_file,
                'all_orders_for_next_day': self._to_pandas(all_orders),
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._empty_result()

    # ------------------------------------------------------------------
    # 转换辅助
    # ------------------------------------------------------------------

    def _to_pandas(self, df):
        if df is None:
            return pd.DataFrame()
        if isinstance(df, pd.DataFrame):
            return df
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
        return df

    # ------------------------------------------------------------------
    # validate_data — 委托到对应 backend
    # ------------------------------------------------------------------

    def validate_data(self, df, name='data', numeric_columns=None,
                      required_columns=None, strict=False):
        if self._engine == 'polars':
            return self._backend.validate_data(
                df, name=name, numeric_columns=numeric_columns,
                required_columns=required_columns, strict=strict,
            )
        return super().validate_data(
            df, name=name, numeric_columns=numeric_columns,
            required_columns=required_columns, strict=strict,
        )

    # ------------------------------------------------------------------
    # 私有工具方法
    # ------------------------------------------------------------------

    def _empty_result(self):
        self._result = {
            'orders_df': pd.DataFrame(),
            'shipment_df': pd.DataFrame(),
            'cut_df': pd.DataFrame(),
            'supply_demand_df': pd.DataFrame(),
            'summary_df': pd.DataFrame(),
            'output_file': None,
        }

    @staticmethod
    def _apply_fast_consumption(orders, quantities, idx_map, offsets):
        for r in orders.itertuples():
            if pd.isna(r.quantity) or r.quantity <= 0:
                continue
            mat = r.material
            loc = r.location
            order_date = pd.to_datetime(r.date)
            if pd.isna(order_date):
                continue
            remaining = int(r.quantity)

            for offset in offsets:
                if remaining <= 0:
                    break
                target_date = order_date + pd.Timedelta(days=offset)
                key = (mat, loc, target_date.date())
                if key in idx_map:
                    for idx in idx_map[key]:
                        if remaining <= 0:
                            break
                        avail = int(quantities[idx])
                        if avail <= 0:
                            continue
                        take = min(avail, remaining)
                        quantities[idx] = avail - take
                        remaining -= take


# ------------------------------------------------------------------
# 兼容入口函数
# ------------------------------------------------------------------

def run_daily_order_generation(
    simulation_date, output_dir='', orchestrator=None, orch=None,
    skip_file_output=False, previous_orders_df=None,
    **kwargs
):
    """兼容入口：创建 ModuleOne 实例并执行。"""
    m1 = ModuleOne(
        simulation_date=simulation_date,
        output_dir=output_dir,
        orchestrator=orchestrator,
        orch=orch,
        skip_file_output=skip_file_output,
        previous_orders_df=previous_orders_df,
        **kwargs
    )
    if m1.order_df is None:
        m1.prepare()
    m1.run()
    return m1.output()


def generate_supply_demand_log_for_integration(
    demand_forecast: pd.DataFrame,
    consumed_forecast: pd.DataFrame,
    simulation_date: pd.Timestamp,
) -> pd.DataFrame:
    """为集成模式生成供需日志（兼容旧接口）。"""
    empty_cols = ['date', 'material', 'location', 'quantity', 'demand_element']

    if consumed_forecast.empty or 'date' not in consumed_forecast.columns:
        return pd.DataFrame(columns=empty_cols)

    sim_date = pd.Timestamp(simulation_date).normalize()
    future_cutoff = sim_date + pd.Timedelta(days=M1_FUTURE_CUTOFF_DAYS)

    future_demand = consumed_forecast[
        (pd.to_datetime(consumed_forecast['date']) > sim_date)
        & (pd.to_datetime(consumed_forecast['date']) <= future_cutoff)
    ].copy()

    if future_demand.empty:
        return pd.DataFrame(columns=empty_cols)

    future_demand['demand_element'] = 'forecast'
    supply_demand_log = future_demand[empty_cols].copy()
    return normalize_identifiers(supply_demand_log)
