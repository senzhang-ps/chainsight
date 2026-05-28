"""ModuleOne 的 pandas / polars 后端实现。

每个 backend 类实现相同的方法签名，供 ModuleOne 通过委托调用。
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import polars as pl

from ...utils.defaults import M1_FUTURE_CUTOFF_DAYS, DEFAULT_MAX_ADVANCE_DAYS
from .io_utils import load_previous_orders

logger = logging.getLogger("SupplyChainSimulation")


# ======================================================================
# Polars 标识符规范化（镜像 normalization.py 的逻辑）
# ======================================================================

_MISSING_TOKENS_LIST = ["nan", "None", "<NA>", "NaN", ""]

_LOCATION_COLUMNS = [
    "location", "dps_location", "sending", "receiving", "sourcing",
]
_MATERIAL_COLUMNS = [
    "material", "from_material", "to_material",
]
_STRING_ONLY_COLUMNS = [
    "line", "delegate_line", "changeover_id",
]


def _normalize_identifiers_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Polars 版标识符规范化 — 纯原生表达式，无 map_elements。"""
    if df.is_empty():
        return df

    exprs = []
    for col_name in df.columns:
        if col_name in _MATERIAL_COLUMNS:
            s = pl.col(col_name).cast(pl.Utf8).str.strip_chars().str.replace(r"\.0$", "")
            exprs.append(
                pl.when(s.is_in(_MISSING_TOKENS_LIST))
                .then(pl.lit(""))
                .otherwise(s)
                .alias(col_name)
            )
        elif col_name in _LOCATION_COLUMNS:
            s = pl.col(col_name).cast(pl.Utf8).str.strip_chars()
            s_clean = pl.when(s.is_in(_MISSING_TOKENS_LIST)).then(pl.lit("")).otherwise(s)
            exprs.append(
                pl.when(s_clean.str.contains(r"^\d+$"))
                .then(s_clean.str.zfill(4))
                .otherwise(s_clean)
                .alias(col_name)
            )
        elif col_name in _STRING_ONLY_COLUMNS:
            exprs.append(pl.col(col_name).cast(pl.Utf8))
        else:
            exprs.append(pl.col(col_name))

    return df.select(exprs)


# ======================================================================
# Polars 版数值安全转换
# ======================================================================

def _safe_int_series_polars(series: pl.Series, context: str, default: int = 0) -> pl.Series:
    """Polars 版 safe_int_series：NaN/inf 替换为 default 并转 int。"""
    arr = series.to_numpy()
    numeric = pd.to_numeric(arr, errors="coerce").astype("float64")
    bad_mask = ~np.isfinite(numeric)
    if bad_mask.any():
        logger.warning(
            "[安全整数转换] %s：发现 %d 个异常值，已替换为 %d",
            context, int(bad_mask.sum()), default,
        )
        numeric = np.where(bad_mask, float(default), numeric)
    return pl.Series(numeric.astype(np.int64), name=series.name)


# ======================================================================
# Pandas 后端 — 直接复用现有逻辑
# ======================================================================

class _PandasBackend:
    """将 ModuleOne 中原有 pandas 操作原样保留。"""

    def __init__(self, owner):
        self._o = owner
        self._ao_config = None  # prepare_ao_summary 后缓存

    # ---- 数据访问（直接返回 pandas DataFrame） ----

    @property
    def demand_forecast(self):
        return self._o.orchestrator.m1_demandforecast

    @property
    def forecast_error(self):
        return self._o.orchestrator.m1_forecasterror

    @property
    def order_calendar(self):
        return self._o.orchestrator.m1_ordercalendar

    @property
    def ao_config(self):
        if self._ao_config is not None:
            return self._ao_config
        return self._o.orchestrator.m1_aoconfig

    @property
    def dps_config(self):
        return self._o.orchestrator.m1_dpsconfig

    @property
    def dps_sc_config(self):
        return self._o.orchestrator.m1_supplychoiceconfig

    # ---- 步骤方法 ----

    def prepare_ao_summary(self):
        ao_config = self.ao_config.copy()
        ao_config.drop_duplicates(
            ["material", "location", "advance_days", "ao_percent"], inplace=True,
        )
        ao_config_summary = ao_config.groupby(
            ["material", "location"]
        ).agg(ao_percent=("ao_percent", "sum")).reset_index()

        ao_config["order_type"] = "AO"
        ao_config.rename(columns={"ao_percent": "percent"}, inplace=True)

        ao_config_summary["order_type"] = "AO"

        ao_config_summary_normal = ao_config_summary.copy()
        ao_config_summary_normal["ao_percent"] = 1 - ao_config_summary_normal["ao_percent"].clip(0, 1)
        ao_config_summary_normal = ao_config_summary_normal[ao_config_summary_normal['ao_percent']>0]
        ao_config_summary_normal["order_type"] = "normal"

        ao_config_summary = pd.concat([
            ao_config_summary[["material", "location", "order_type", "ao_percent"]],
            ao_config_summary_normal[["material", "location", "order_type", "ao_percent"]]
        ])
        self._ao_config = ao_config  # 缓存修改后的版本，供下游 split_ao_by_advance_days 使用
        return ao_config, ao_config_summary

    def build_dps(self):
        demand_forecast = self.demand_forecast
        dps_config = self.dps_config
        dps_sc_config = self.dps_sc_config 

        dps_config = dps_config.copy()
        dps_config["reverse_dps_percent"] = 1 - dps_config["dps_percent"]
        dps_config = pd.concat([
            dps_config[["material", "location", "reverse_dps_percent"]].rename(
                columns={"reverse_dps_percent": "dps_percent"}),
            dps_config[["material", "dps_location", "dps_percent"]].rename(
                columns={"dps_location": "location"})
        ])

        demand_forecast = demand_forecast.groupby(
            ["week", "material", "location"]
        )["quantity"].sum().reset_index()
        demand_forecast = demand_forecast.sort_values(["material", "location", "week"])
        demand_forecast["week_start"] = demand_forecast["week"].map(
            lambda x: self._o.orchestrator.start_date + pd.Timedelta(days=(x - 1) * 7)
        )
        demand_forecast["week_start"] = pd.to_datetime(demand_forecast["week_start"].astype(object))
        demand_forecast['month'] = ((demand_forecast['week']-1)//4)+1

        demand_forecast_split_by_dps = pd.merge(
            demand_forecast, dps_config, on=["material", "location"], how="left",
        )
        demand_forecast_split_by_dps = demand_forecast_split_by_dps.fillna(1)
        demand_forecast_split_by_dps["quantity_percentage"] = (
            demand_forecast_split_by_dps["quantity"] * demand_forecast_split_by_dps["dps_percent"]
        )
        # demand_forecast_split_by_dps.drop('')

        demand_forecast_total = demand_forecast_split_by_dps.copy()
        demand_forecast_total["quantity_total"] = demand_forecast_total["quantity_percentage"]

        if not dps_sc_config.empty:
            demand_forecast_total_sc = pd.merge(
                demand_forecast_split_by_dps, dps_sc_config,
                on=["week", "material", "quantity"], how="left",
            )
            demand_forecast_total_sc["quantity_total"] = (
                demand_forecast_total_sc["quantity_percentage"]
                + demand_forecast_total_sc["adjust_quantity"]
            )
        else:
            demand_forecast_total_sc = demand_forecast_total.copy()

        # order_calendar["week_start"] = pd.to_datetime(order_calendar["week_start"].astype(object))

        return demand_forecast_total, demand_forecast_total_sc

    def build_cov(self, demand_forecast_total, ao_config_summary):
        df = pd.merge(demand_forecast_total, ao_config_summary, on=["material", "location"], how="left")
        # df = df.dropna(subset='order_type')
        df["order_type"] = df["order_type"].fillna("normal")
        df["ao_percent"] = df["ao_percent"].fillna(1)
        df = df[df['quantity_total']>0]
        
        df["split_quantity"] = df["ao_percent"] * df["quantity_total"]

        df = pd.merge(df, self.forecast_error, on=["material", "location", "order_type"], how="left")
        df["error_std_percent"] = df["error_std_percent"].fillna(0)
        df["abs_std"] = df["split_quantity"] * df["error_std_percent"]
        df["cov_quantity_raw"] = np.maximum(
            0, np.round(np.random.normal(df["split_quantity"], df["abs_std"]))
        ).astype(int)
        df['cov_quantity_raw'] = df['cov_quantity_raw'].replace(0,1)
        
        

        df['rescue_rate'] = (
            df.groupby(['material', 'location', 'month', 'order_type'])
            .apply(lambda g: g['split_quantity'].sum() / g['cov_quantity_raw'].sum())
            .reindex(df.set_index(['material', 'location', 'month', 'order_type']).index)
            .values
        )

        df['cov_quantity'] = df['rescue_rate'] * df['cov_quantity_raw'] 
        return df

    def build_daily_order(self, demand_forecast, qty_col="quantity_total"):
        order_cal = self.order_calendar.copy()
        # 生成simulation_date，再根据simulation_date移动窗口统计flag_count
        week_starts = demand_forecast["week_start"].sort_values().drop_duplicates().reset_index(drop=True)
        max_date, min_date = demand_forecast['week_start'].max(),demand_forecast['week_start'].min()
        # 1. 生成所有 week_start（每周一）
        all_week_starts = pd.date_range(start=min_date, end=max_date, freq='7D')

        # 2. 对每个 week_start，生成7天的 simulation_date
        date_records = []
        for ws in all_week_starts:
            sim_dates = pd.date_range(start=ws, periods=7, freq='D')
            for sd in sim_dates:
                date_records.append({'week_start': ws, 'simulation_date': sd})

        # 3. 构造 date_df
        date_df = pd.DataFrame(date_records)

        demand_forecast = pd.merge(demand_forecast,date_df,how='left',on='week_start')
        demand_forecast = pd.merge(demand_forecast, order_cal, how='left', left_on='simulation_date', right_on='date')
        demand_forecast['order_day_flag'] = demand_forecast['order_day_flag'].fillna(1)
        demand_forecast.sort_values(by=['material','location','simulation_date'],inplace=True, ascending=True)

        bins = week_starts.tolist() + [week_starts.iloc[-1] + pd.Timedelta(days=7)]
        order_cal["week_start"] = pd.cut(
            order_cal["date"], bins=bins, labels=week_starts, right=False,
        )

        flag_summary = order_cal.groupby("week_start")["order_day_flag"].sum().reset_index(name="flag_count")
        flag_summary["week_start"] = pd.to_datetime(flag_summary["week_start"])

        demand_forecast = demand_forecast.merge(flag_summary, on="week_start", how="left")

        demand_forecast["_qty_int"] = demand_forecast[qty_col].round().astype(int)
        demand_forecast["base_qty"] = np.where(
            (demand_forecast["order_day_flag"] == 1) & (demand_forecast["flag_count"] > 0),
            demand_forecast["_qty_int"] // demand_forecast["flag_count"],
            0,
        )
        demand_forecast["remainder"] = demand_forecast["_qty_int"] % demand_forecast["flag_count"]
        rank_group = ["material", "location", "week_start"]
        if "order_type" in demand_forecast.columns:
            rank_group.append("order_type")
        ordering_rank = (
            demand_forecast["order_day_flag"]
            .groupby([demand_forecast[c] for c in rank_group])
            .cumsum() - 1
        )
        demand_forecast["base_qty"] += (
            (demand_forecast["order_day_flag"] == 1) & (ordering_rank < demand_forecast["remainder"])
        ).astype(int)
        drop_cols = [c for c in ["quantity", "_qty_int"] if c in demand_forecast.columns]
        demand_forecast.drop(drop_cols, axis=1, inplace=True)
        demand_forecast = demand_forecast.rename(columns={"base_qty": "quantity"})
        return demand_forecast

    def adjust_daily_order_ao(self, daily_order):
        result = daily_order.copy()
        result = pd.merge(result, self.ao_config, on=["material", "location", "order_type"], how="left")
        result["advance_days"] = result["advance_days"].fillna(0)
        result["percent"] = result["percent"].fillna(result["ao_percent"])
        result["quantity"] = result["quantity"] * result["percent"] / result["ao_percent"]
        result["quantity"] = result["quantity"].fillna(0)
        result["quantity"] = result["quantity"].round()
        result["date"] += pd.to_timedelta(result["advance_days"], unit="D")
        result = result.rename(columns={"order_type": "demand_type"})
        return result



    def apply_dps_and_supply_choice(self):
        demand_forecast = self.demand_forecast
        dps_config = self.dps_config
        dps_sc_config = self.dps_sc_config
        order_calendar = self.order_calendar

        dps_config = dps_config.copy()
        dps_config["reverse_dps_percent"] = 1 - dps_config["dps_percent"]
        dps_config = pd.concat([
            dps_config[["material", "location", "reverse_dps_percent"]].rename(
                columns={"reverse_dps_percent": "dps_percent"}),
            dps_config[["material", "dps_location", "dps_percent"]].rename(
                columns={"dps_location": "location"})
        ])

        demand_forecast = demand_forecast.groupby(
            ["week", "material", "location"]
        )["quantity"].sum().reset_index()
        demand_forecast = demand_forecast.sort_values(["material", "location", "week"])
        demand_forecast["week_start"] = demand_forecast["week"].map(
            lambda x: self._o.orchestrator.start_date + pd.Timedelta(days=(x - 1) * 7)
        )
        demand_forecast["week_start"] = pd.to_datetime(demand_forecast["week_start"].astype(object))

        # 生成simulation_date，再根据simulation_date移动窗口统计flag_count
        week_starts = demand_forecast["week_start"].sort_values().drop_duplicates().reset_index(drop=True)
        max_date, min_date = demand_forecast['week_start'].max(),demand_forecast['week_start'].min()
        # 1. 生成所有 week_start（每周一）
        all_week_starts = pd.date_range(start=min_date, end=max_date, freq='7D')

        # 2. 对每个 week_start，生成7天的 simulation_date
        date_records = []
        for ws in all_week_starts:
            sim_dates = pd.date_range(start=ws, periods=7, freq='D')
            for sd in sim_dates:
                date_records.append({'week_start': ws, 'simulation_date': sd})

        # 3. 构造 date_df
        date_df = pd.DataFrame(date_records)

        demand_forecast = pd.merge(demand_forecast,date_df,how='left',on='week_start')
        demand_forecast = pd.merge(demand_forecast, order_calendar, how='left', left_on='simulation_date', right_on='date')
        demand_forecast['order_day_flag'] = demand_forecast['order_day_flag'].fillna(1)
        demand_forecast.sort_values(by=['material','location','simulation_date'],inplace=True, ascending=True)
        # demand_forecast['flag_count'] = (
        #     demand_forecast.groupby(['material', 'location'])['order_day_flag']
        #     .apply(lambda x: x[::-1].rolling(window=7, min_periods=1).sum()[::-1])
        #     .values  # .values 避免索引对齐问题
        # )
        demand_forecast['month'] = ((demand_forecast['week']-1)//4)+1


        bins = week_starts.tolist() + [week_starts.iloc[-1] + pd.Timedelta(days=7)]
        order_calendar["week_start"] = pd.cut(
            order_calendar["date"], bins=bins, labels=week_starts, right=False,
        )

        flag_summary = order_calendar.groupby("week_start")["order_day_flag"].sum().reset_index(name="flag_count")
        flag_summary["week_start"] = pd.to_datetime(flag_summary["week_start"])

        demand_forecast = demand_forecast.merge(flag_summary, on="week_start", how="left")
        # demand_forecast.dropna(inplace=True)

        demand_forecast_split_by_dps = pd.merge(
            demand_forecast, dps_config, on=["material", "location"], how="left",
        )
        demand_forecast_split_by_dps = demand_forecast_split_by_dps.fillna(1)
        demand_forecast_split_by_dps["quantity_percentage"] = (
            demand_forecast_split_by_dps["quantity"] * demand_forecast_split_by_dps["dps_percent"]
        )
        # demand_forecast_split_by_dps.drop('')

        demand_forecast_total = demand_forecast_split_by_dps.copy()
        demand_forecast_total["quantity_total"] = demand_forecast_total["quantity_percentage"]

        if not dps_sc_config.empty:
            demand_forecast_total_sc = pd.merge(
                demand_forecast_split_by_dps, dps_sc_config,
                on=["week", "material", "quantity"], how="left",
            )
            demand_forecast_total_sc["quantity_total"] = (
                demand_forecast_total_sc["quantity_percentage"]
                + demand_forecast_total_sc["adjust_quantity"]
            )
        else:
            demand_forecast_total_sc = demand_forecast_total.copy()

        # order_calendar["week_start"] = pd.to_datetime(order_calendar["week_start"].astype(object))

        return demand_forecast_total, demand_forecast_total_sc, order_calendar

    def distribute_to_daily(self, demand_forecast_total):
        # detail = pd.merge(
        #     demand_forecast_total[["week", "location", "material", "quantity_total", "week_start", "flag_count"]],
        #     order_calendar, on="week_start", how="left",
        # )
        detail = demand_forecast_total.copy()
        detail["base_qty"] = np.where(
            (detail["order_day_flag"] == 1) & (detail["flag_count"] > 0),
            (detail["quantity_total"] // detail["flag_count"]).astype(int),
            0,
        )
        detail["remainder"] = (detail["quantity_total"] % detail["flag_count"]).astype(int)
        day_offset = (detail["date"] - detail["week_start"]).dt.days
        detail["base_qty"] += (day_offset < detail["remainder"]).astype(int)
        drop_cols = [c for c in ["quantity_total", "quantity"] if c in detail.columns]
        detail.drop(drop_cols, axis=1, inplace=True)
        detail = detail.rename(columns={"base_qty": "quantity"})
        return detail

    def split_by_ao_and_apply_error(self, demand_forecast_total, ao_config_summary):
        df = pd.merge(demand_forecast_total, ao_config_summary, on=["material", "location"], how="left")
        # df = df.dropna(subset='order_type')
        df["order_type"] = df["order_type"].fillna("normal")
        df["ao_percent"] = df["ao_percent"].fillna(1)
        df = df[df['quantity_total']>0]
        
        df["split_quantity"] = df["ao_percent"] * df["quantity_total"]

        df = pd.merge(df, self.forecast_error, on=["material", "location", "order_type"], how="left")
        df["error_std_percent"] = df["error_std_percent"].fillna(0)
        df["abs_std"] = df["split_quantity"] * df["error_std_percent"]
        df["cov_quantity_raw"] = np.maximum(
            0, np.round(np.random.normal(df["split_quantity"], df["abs_std"]))
        ).astype(int)
        df['cov_quantity_raw'] = df['cov_quantity_raw'].replace(0,1)
        

        df['rescue_rate'] = (
            df.groupby(['material', 'location', 'month', 'order_type'])
            .apply(lambda g: g['split_quantity'].sum() / g['cov_quantity_raw'].sum())
            .reindex(df.set_index(['material', 'location', 'month', 'order_type']).index)
            .values
        )

        df['cov_quantity'] = df['rescue_rate'] * df['cov_quantity_raw']
        # df.to_csv('sub_orders.csv',index=False)
        return df

    def split_ao_by_advance_days(self, df_with_error, order_calendar):
        # target_date = (
        #     order_calendar[order_calendar["date"] == self._o.simulation_date]["week_start"]
        #     .dt.strftime("%Y-%m-%d").item()
        # )
        # df_filtered = df_with_error[df_with_error["week_start"] == target_date]
        df_filtered = df_with_error.copy()
        result = pd.merge(df_filtered, self.ao_config, on=["material", "location", "order_type"], how="left")
        result["advance_days"] = result["advance_days"].fillna(0)
        result["percent"] = result["percent"].fillna(result["ao_percent"])
        result["cov_quantity_ao_detail"] = (
            result["cov_quantity"] * result["percent"] / result["ao_percent"]
        )
        result['cov_quantity_ao_detail'] = result['cov_quantity_ao_detail'].fillna(0)
        result['date']+=pd.to_timedelta(result["advance_days"], unit="D")
        result['daily_quantity'] = result["cov_quantity_ao_detail"] / result["flag_count"]
        result.drop('quantity', axis=1, inplace=True)
        result=result.rename(columns={"daily_quantity": "quantity", "order_type": "demand_type"})
        return result

    def build_order_df(self, ao_detail):
        ao_detail["daily_quantity"] = ao_detail["cov_quantity_ao_detail"] / ao_detail["flag_count"]
        ao_detail["date"] = (
            pd.to_datetime(self._o.simulation_date)
            + pd.to_timedelta(ao_detail["advance_days"], unit="D")
        )
        order_df = (
            ao_detail
            .groupby(["date", "material", "location", "order_type", "advance_days"])["daily_quantity"]
            .sum()
            .reset_index()
            .rename(columns={"daily_quantity": "quantity", "order_type": "demand_type"})
        )
        # order_df["simulation_date"] = self._o.simulation_date
        return order_df

    def merge_with_history(self, order_df):
        sim_date = self._o.simulation_date

        # 今日订单: simulation_date == today
        if order_df is not None and not order_df.empty and "simulation_date" in order_df.columns:
            today_orders = order_df[order_df["simulation_date"] == sim_date].copy()
        else:
            today_orders = order_df.copy() if order_df is not None else pd.DataFrame()

        # 所有有效订单: simulation_date <= today AND date >= today
        all_orders = order_df.copy() if order_df is not None else pd.DataFrame()
        if not all_orders.empty:
            if "date" in all_orders.columns:
                all_orders["date"] = pd.to_datetime(all_orders["date"])
            if "simulation_date" in all_orders.columns:
                all_orders = all_orders[
                    (all_orders["simulation_date"] <= sim_date)
                    & (all_orders["date"] >= sim_date)
                ]
            all_orders = self._deduplicate_orders(all_orders)
            all_orders = self._normalize_orders(all_orders)

        return all_orders, today_orders

    def apply_orders_consumption(self, forecast_df, orders_df):
        if forecast_df is None or forecast_df.empty:
            return pd.DataFrame(columns=["material", "location", "date", "quantity"])

        consumed = forecast_df.copy()

        if orders_df is None or orders_df.empty:
            return consumed

        from ...utils.normalization import normalize_identifiers
        consumed = normalize_identifiers(consumed)
        orders_df = normalize_identifiers(orders_df.copy())
        consumed["date"] = pd.to_datetime(consumed["date"])
        orders_df["date"] = pd.to_datetime(orders_df["date"])
        offsets = [0, -1, -2, 1, 2, 3]

        idx_map = {}
        for idx, row in enumerate(consumed.itertuples()):
            row_date = pd.to_datetime(row.date)
            if pd.isna(row_date):
                continue
            key = (row.material, row.location, row_date.date())
            idx_map.setdefault(key, []).append(idx)

        quantity_data = consumed.loc[:, "quantity"]
        if isinstance(quantity_data, pd.DataFrame):
            quantity_data = quantity_data.iloc[:, -1]
            consumed = consumed.loc[:, ~consumed.columns.duplicated(keep="last")]
        quantities = pd.to_numeric(
            quantity_data, errors="coerce"
        ).fillna(0).values.copy().astype(float)

        ao_orders = orders_df[orders_df["demand_type"] == "AO"].copy()
        if not ao_orders.empty:
            ao_orders = ao_orders.sort_values(
                by=["date", "advance_days", "quantity", "simulation_date"],
            )
            self._o._apply_fast_consumption(ao_orders, quantities, idx_map, offsets)

        normal_orders = orders_df[orders_df["demand_type"] == "normal"].copy()
        if not normal_orders.empty:
            normal_orders = normal_orders.sort_values(
                by=["date", "quantity", "simulation_date"],
            )
            self._o._apply_fast_consumption(normal_orders, quantities, idx_map, offsets)

        consumed["quantity"] = quantities.astype(int)
        return normalize_identifiers(consumed)

    def generate_supply_demand_log(self, demand_forecast, consumed_forecast):
        empty_cols = ["date", "material", "location", "quantity", "demand_element"]
        if consumed_forecast.empty or "date" not in consumed_forecast.columns:
            return pd.DataFrame(columns=empty_cols)

        sim_date = pd.Timestamp(self._o.simulation_date).normalize()
        future_cutoff = sim_date + pd.Timedelta(days=M1_FUTURE_CUTOFF_DAYS)
        future_demand = consumed_forecast[
            (pd.to_datetime(consumed_forecast["date"]) > sim_date)
            & (pd.to_datetime(consumed_forecast["date"]) <= future_cutoff)
        ].copy()

        if future_demand.empty:
            return pd.DataFrame(columns=empty_cols)

        future_demand = future_demand.groupby(
            ["date", "material", "location"], as_index=False
        )["quantity"].sum()
        future_demand["demand_element"] = "forecast"
        from ...utils.normalization import normalize_identifiers
        return normalize_identifiers(future_demand[empty_cols].copy())

    def build_summary(self, orders_df, shipment_df, cut_df, supply_demand_df):
        from ...utils.normalization import normalize_identifiers
        date_val = orders_df["date"].iloc[0] if not orders_df.empty else None
        return pd.DataFrame([{
            "Total_Orders": len(orders_df),
            "Total_Shipments": len(shipment_df),
            "Total_Cuts": len(cut_df),
            "Total_SupplyDemand": len(supply_demand_df),
            "Date": date_val,
        }])

    def validate_data(self, df, name="data", numeric_columns=None, required_columns=None, strict=False):
        from ..module import Module
        return Module.validate_data(self._o, df, name=name,
                                    numeric_columns=numeric_columns,
                                    required_columns=required_columns,
                                    strict=strict)

    # ---- 私有工具 ----

    def _get_max_advance_days(self):
        ao_cfg = self.ao_config
        if not ao_cfg.empty and "advance_days" in ao_cfg.columns:
            max_val = ao_cfg["advance_days"].max(skipna=True)
            return int(max_val) if pd.notna(max_val) else DEFAULT_MAX_ADVANCE_DAYS
        return DEFAULT_MAX_ADVANCE_DAYS

    def _filter_future_orders(self, orders):
        if not orders.empty and "date" in orders.columns:
            orders["date"] = pd.to_datetime(orders["date"])
            return orders[orders["date"] >= self._o.simulation_date].copy()
        return orders

    def _deduplicate_orders(self, orders):
        if orders.empty:
            return orders
        dedup_keys = [
            c for c in [
                "date", "material", "location", "demand_type",
                "simulation_date", "advance_days", "quantity",
            ]
            if c in orders.columns
        ]
        if dedup_keys:
            return orders.drop_duplicates(subset=dedup_keys)
        return orders

    def _normalize_orders(self, orders_df):
        if orders_df.empty:
            return orders_df
        from ...utils.normalization import normalize_identifiers
        if "quantity" in orders_df.columns:
            orders_df["quantity"] = orders_df["quantity"].astype(int)
        if "simulation_date" not in orders_df.columns:
            orders_df["simulation_date"] = orders_df["date"]
        return normalize_identifiers(orders_df)

    # ---- 发货 / 保存 / 辅助 ----

    def generate_shipments(self, orders_df, daily_detail):
        orch = self._o.legacy_orchestrator
        if orch is None or orch.shipment_valid == 0:
            return pd.DataFrame(), pd.DataFrame()
        from .shipment import generate_shipment_with_inventory_check
        return generate_shipment_with_inventory_check(
            orders_df, self._o.simulation_date, orch, daily_detail, None,
        )

    def save_output(self, orders_df, shipment_df, cut_df, supply_demand_df, summary_df):
        if self._o.skip_file_output:
            return None
        import os
        from .io_utils import save_module1_output_with_supply_demand
        date_str = pd.Timestamp(self._o.simulation_date).strftime("%Y%m%d")
        output_file = os.path.join(
            self._o.output_dir,
            f"module1_output_{date_str}.xlsx",
        )
        save_module1_output_with_supply_demand(
            orders_df, shipment_df, supply_demand_df, output_file, cut_df, summary_df
        )
        return output_file

    def get_order_day_flag(self, order_cal):
        filtered = order_cal.loc[
            order_cal["date"] == self._o.simulation_date, "order_day_flag"
        ]
        return filtered.item() if len(filtered) > 0 else 0


# ======================================================================
# Polars 后端
# ======================================================================

class _PolarsBackend:
    """ModuleOne 的 polars 实现 — 方法签名与 _PandasBackend 一致。"""

    def __init__(self, owner):
        self._o = owner
        # 缓存转换后的 polars 数据
        self._demand_forecast = None
        self._forecast_error = None
        self._order_calendar = None
        self._ao_config = None
        self._dps_config = None
        self._dps_sc_config = None

    def _to_pl(self, df):
        if df is None:
            return pl.DataFrame()
        if isinstance(df, pl.DataFrame):
            return df
        # 空 DataFrame 也走 numpy 转换，以保留 dtype
        # 将 nullable / extension dtypes 转为 numpy-backed dtypes，
        # 避免 pl.from_pandas 要求 pyarrow 依赖。
        converted = {}
        for col in df.columns:
            s = df[col]
            try:
                # 先尝试直接转 numpy
                arr = s.to_numpy()
                # 如果已经是纯 numpy dtype（非 object 中的混合类型），直接用
                converted[col] = arr
            except Exception:
                # 降级为 str
                converted[col] = s.astype(str).to_numpy()
        return pl.DataFrame(converted)

    @property
    def demand_forecast(self):
        if self._demand_forecast is None:
            self._demand_forecast = self._to_pl(self._o.orchestrator.m1_demandforecast)
        return self._demand_forecast

    @property
    def forecast_error(self):
        if self._forecast_error is None:
            self._forecast_error = self._to_pl(self._o.orchestrator.m1_forecasterror)
        return self._forecast_error

    @property
    def order_calendar(self):
        if self._order_calendar is None:
            oc = self._to_pl(self._o.orchestrator.m1_ordercalendar)
            if not oc.is_empty() and "date" in oc.columns:
                oc = oc.with_columns(pl.col("date").cast(pl.Date))
            self._order_calendar = oc
        return self._order_calendar

    @property
    def ao_config(self):
        if self._ao_config is None:
            self._ao_config = self._to_pl(self._o.orchestrator.m1_aoconfig)
        return self._ao_config

    @property
    def dps_config(self):
        if self._dps_config is None:
            self._dps_config = self._to_pl(self._o.orchestrator.m1_dpsconfig)
        return self._dps_config

    @property
    def dps_sc_config(self):
        if self._dps_sc_config is None:
            self._dps_sc_config = self._to_pl(self._o.orchestrator.m1_supplychoiceconfig)
        return self._dps_sc_config

    # ---- 步骤方法 ----

    def prepare_ao_summary(self):
        ao_config = self.ao_config.unique(
            subset=["material", "location", "advance_days", "ao_percent"],
        )

        ao_config_summary = (
            ao_config
            .group_by(["material", "location"])
            .agg(ao_percent=pl.col("ao_percent").sum())
        )

        ao_config = ao_config.with_columns(
            pl.lit("AO").alias("order_type"),
        ).rename({"ao_percent": "percent"})

        ao_config_summary = ao_config_summary.with_columns(
            pl.lit("AO").alias("order_type"),
        )

        ao_config_summary_normal = ao_config_summary.with_columns(
            (1 - pl.col("ao_percent").clip(0, 1)).alias("ao_percent"),
        ).filter(
            pl.col("ao_percent") > 0,
        ).with_columns(
            pl.lit("normal").alias("order_type"),
        )

        ao_config_summary = pl.concat([
            ao_config_summary.select("material", "location", "order_type", "ao_percent"),
            ao_config_summary_normal.select("material", "location", "order_type", "ao_percent"),
        ], how="vertical")

        self._ao_config = ao_config  # 缓存修改后的版本，供下游 split_ao_by_advance_days 使用
        return ao_config, ao_config_summary

    def apply_dps_and_supply_choice(self):
        demand_forecast = self.demand_forecast
        dps_config = self.dps_config
        dps_sc_config = self.dps_sc_config
        order_calendar = self.order_calendar

        # DPS 反向拆分
        dps_config = dps_config.with_columns(
            (1 - pl.col("dps_percent")).alias("reverse_dps_percent"),
        )
        dps_config = pl.concat([
            dps_config.select("material", "location", pl.col("reverse_dps_percent").alias("dps_percent")),
            dps_config.select("material", pl.col("dps_location").alias("location"), "dps_percent"),
        ], how="vertical")

        # 按周聚合 + week_start（合并为一次操作，无需 clone）
        start_date = self._o.orchestrator.start_date
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.to_pydatetime()

        demand_forecast = (
            demand_forecast
            .group_by(["week", "material", "location"])
            .agg(pl.col("quantity").sum())
            .sort(["material", "location", "week"])
            .with_columns(
                (start_date + (pl.col("week") - 1) * pl.duration(days=7)).cast(pl.Date).alias("week_start"),
            )
        )

        # 生成simulation_date，再根据simulation_date移动窗口统计flag_count
        from datetime import timedelta

        ws_list = demand_forecast.select("week_start").unique().sort("week_start")["week_start"].to_list()
        if not ws_list:
            return pl.DataFrame(), pl.DataFrame(), order_calendar

        max_ws, min_ws = ws_list[-1], ws_list[0]

        # 1. 生成所有 week_start（每周一）
        week_starts = ws_list

        # 2. 对每个 week_start，生成7天的 simulation_date
        date_records = []
        for ws in week_starts:
            for offset in range(7):
                date_records.append({"week_start": ws, "simulation_date": ws + timedelta(days=offset)})

        # 3. 构造 date_df
        date_df = pl.DataFrame(date_records)

        demand_forecast = demand_forecast.join(date_df, on="week_start", how="left")
        demand_forecast = demand_forecast.join(
            order_calendar.select(
                pl.col("date").cast(pl.Date),
                pl.col("date").cast(pl.Date).alias("_calendar_date"),
                "order_day_flag",
            ),
            left_on="simulation_date", right_on="date", how="left",
        ).with_columns(
            pl.col("_calendar_date").alias("date"),
        ).drop("_calendar_date")
        demand_forecast = demand_forecast.with_columns(
            pl.col("order_day_flag").fill_null(1),
        )
        demand_forecast = demand_forecast.sort(["material", "location", "simulation_date"])
        demand_forecast = demand_forecast.with_columns(
            ((pl.col("week") - 1) // 4 + 1).alias("month"),
        )

        n_weeks = len(ws_list)

        # flag_summary: 按 week_start 统计 order_day_flag
        order_calendar = order_calendar.with_columns(pl.col("date").cast(pl.Date))
        _days = (pl.col("date") - min_ws).dt.total_days()
        _idx = _days // 7
        order_calendar = order_calendar.with_columns(
            pl.when((_idx >= 0) & (_idx < n_weeks))
            .then((min_ws + _idx * pl.duration(days=7)).cast(pl.Date))
            .otherwise(None)
            .alias("week_start"),
        )
        flag_summary = (
            order_calendar
            .filter(pl.col("week_start").is_not_null())
            .group_by("week_start")
            .agg(pl.col("order_day_flag").sum().alias("flag_count"))
        )
        demand_forecast = demand_forecast.join(flag_summary, on="week_start", how="left")

        demand_forecast_split_by_dps = demand_forecast.join(
            dps_config, on=["material", "location"], how="left",
        )
        demand_forecast_split_by_dps = demand_forecast_split_by_dps.with_columns(
            pl.col("dps_percent").fill_null(1),
        )
        demand_forecast_split_by_dps = demand_forecast_split_by_dps.with_columns(
            (pl.col("quantity") * pl.col("dps_percent")).alias("quantity_percentage"),
        )

        demand_forecast_total = demand_forecast_split_by_dps.with_columns(
            pl.col("quantity_percentage").alias("quantity_total"),
        )

        # Supply choice 版本
        if not dps_sc_config.is_empty():
            demand_forecast_total_sc = demand_forecast_split_by_dps.join(
                dps_sc_config, on=["week", "material", "quantity"], how="left",
            ).with_columns(
                (pl.col("quantity_percentage") + pl.col("adjust_quantity").fill_null(0)).alias("quantity_total"),
            )
        else:
            demand_forecast_total_sc = demand_forecast_total.clone()

        return demand_forecast_total, demand_forecast_total_sc, order_calendar

    def distribute_to_daily(self, demand_forecast_total):
        detail = demand_forecast_total.clone()
        detail = detail.with_columns([
            pl.when((pl.col("order_day_flag") == 1) & (pl.col("flag_count") > 0))
            .then((pl.col("quantity_total") // pl.col("flag_count")).cast(pl.Int64))
            .otherwise(0)
            .alias("base_qty"),
            (pl.col("quantity_total") % pl.col("flag_count")).cast(pl.Int64).alias("remainder"),
        ])
        day_offset = (pl.col("date") - pl.col("week_start")).dt.total_days()
        detail = detail.with_columns(
            (pl.col("base_qty") + (day_offset < pl.col("remainder")).cast(pl.Int64)).alias("base_qty"),
        )
        drop_cols = [c for c in ["quantity_total", "quantity"] if c in detail.columns]
        detail = detail.drop(drop_cols).rename({"base_qty": "quantity"})

        return detail

    def split_by_ao_and_apply_error(self, demand_forecast_total, ao_config_summary):
        df = demand_forecast_total.join(ao_config_summary, on=["material", "location"], how="left")
        df = df.with_columns([
            pl.col("order_type").fill_null("normal"),
            pl.col("ao_percent").fill_null(1),
        ])
        df = df.filter(pl.col("quantity_total") > 0)
        df = df.with_columns(
            (pl.col("ao_percent") * pl.col("quantity_total")).alias("split_quantity"),
        )

        df = df.join(self.forecast_error, on=["material", "location", "order_type"], how="left")
        df = df.with_columns([
            pl.col("error_std_percent").fill_null(0),
        ])
        df = df.with_columns(
            (pl.col("split_quantity") * pl.col("error_std_percent")).alias("abs_std"),
        )

        # 随机数仍用 numpy
        split_qty = df["split_quantity"].to_numpy().astype(float)
        abs_std = df["abs_std"].to_numpy().astype(float)
        raw = np.random.normal(np.nan_to_num(split_qty), np.nan_to_num(abs_std))
        cov = np.maximum(0, np.round(np.nan_to_num(raw))).astype(int)
        cov = np.where(cov == 0, 1, cov)
        df = df.with_columns(pl.Series("cov_quantity_raw", cov))

        df = df.with_columns(
            (pl.col("split_quantity").sum() / pl.col("cov_quantity_raw").sum())
            .over(["material", "location", "month", "order_type"])
            .alias("rescue_rate"),
        )
        df = df.with_columns(
            (pl.col("rescue_rate") * pl.col("cov_quantity_raw")).alias("cov_quantity"),
        )
        return df

    def split_ao_by_advance_days(self, df_with_error, order_calendar):
        df_filtered = df_with_error.clone()

        result = df_filtered.join(self.ao_config, on=["material", "location", "order_type"], how="left")
        result = result.with_columns(
            pl.col("advance_days").fill_null(0),
        )
        result = result.with_columns(
            pl.col("percent").fill_null(pl.col("ao_percent")),
        )
        result = result.with_columns(
            (pl.col("cov_quantity") * pl.col("percent") / pl.col("ao_percent")).alias("cov_quantity_ao_detail"),
        )
        result = result.with_columns(
            pl.col("cov_quantity_ao_detail").fill_null(0),
        )
        result = result.with_columns(
            (pl.col("date") + pl.duration(days=pl.col("advance_days").cast(pl.Int64))).alias("date"),
        )
        result = result.with_columns(
            (pl.col("cov_quantity_ao_detail") / pl.col("flag_count")).alias("daily_quantity"),
        )
        result = result.drop("quantity").rename({"daily_quantity": "quantity", "order_type": "demand_type"})
        return result

    # ---- 新流程方法 ----

    def build_dps(self):
        demand_forecast = self.demand_forecast
        dps_config = self.dps_config
        dps_sc_config = self.dps_sc_config

        # DPS 反向拆分
        dps_config = dps_config.with_columns(
            (1 - pl.col("dps_percent")).alias("reverse_dps_percent"),
        )
        dps_config = pl.concat([
            dps_config.select("material", "location", pl.col("reverse_dps_percent").alias("dps_percent")),
            dps_config.select("material", pl.col("dps_location").alias("location"), "dps_percent"),
        ], how="vertical")

        # 按周聚合 + week_start + month
        start_date = self._o.orchestrator.start_date
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.to_pydatetime()

        demand_forecast = (
            demand_forecast
            .group_by(["week", "material", "location"])
            .agg(pl.col("quantity").sum())
            .sort(["material", "location", "week"])
            .with_columns([
                (start_date + (pl.col("week") - 1) * pl.duration(days=7)).cast(pl.Date).alias("week_start"),
                ((pl.col("week") - 1) // 4 + 1).alias("month"),
            ])
        )

        # Join DPS config → quantity_percentage
        demand_forecast_split_by_dps = demand_forecast.join(
            dps_config, on=["material", "location"], how="left",
        ).with_columns(
            pl.col("dps_percent").fill_null(1),
        ).with_columns(
            (pl.col("quantity") * pl.col("dps_percent")).alias("quantity_percentage"),
        )

        demand_forecast_total = demand_forecast_split_by_dps.with_columns(
            pl.col("quantity_percentage").alias("quantity_total"),
        )

        # Supply choice 版本
        if not dps_sc_config.is_empty():
            demand_forecast_total_sc = demand_forecast_split_by_dps.join(
                dps_sc_config, on=["week", "material", "quantity"], how="left",
            ).with_columns(
                (pl.col("quantity_percentage") + pl.col("adjust_quantity").fill_null(0)).alias("quantity_total"),
            )
        else:
            demand_forecast_total_sc = demand_forecast_total.clone()

        return demand_forecast_total, demand_forecast_total_sc

    def build_cov(self, demand_forecast_total, ao_config_summary):
        df = demand_forecast_total.join(ao_config_summary, on=["material", "location"], how="left")
        df = df.with_columns([
            pl.col("order_type").fill_null("normal"),
            pl.col("ao_percent").fill_null(1),
        ])
        df = df.filter(pl.col("quantity_total") > 0)
        df = df.with_columns(
            (pl.col("ao_percent") * pl.col("quantity_total")).alias("split_quantity"),
        )

        df = df.join(self.forecast_error, on=["material", "location", "order_type"], how="left")
        df = df.with_columns(
            pl.col("error_std_percent").fill_null(0),
        )
        df = df.with_columns(
            (pl.col("split_quantity") * pl.col("error_std_percent")).alias("abs_std"),
        )

        # 随机数仍用 numpy
        split_qty = df["split_quantity"].to_numpy().astype(float)
        abs_std = df["abs_std"].to_numpy().astype(float)
        raw = np.random.normal(np.nan_to_num(split_qty), np.nan_to_num(abs_std))
        cov = np.maximum(0, np.round(np.nan_to_num(raw))).astype(int)
        cov = np.where(cov == 0, 1, cov)
        df = df.with_columns(pl.Series("cov_quantity_raw", cov))

        df = df.with_columns(
            (pl.col("split_quantity").sum() / pl.col("cov_quantity_raw").sum())
            .over(["material", "location", "month", "order_type"])
            .alias("rescue_rate"),
        )
        df = df.with_columns(
            (pl.col("rescue_rate") * pl.col("cov_quantity_raw")).alias("cov_quantity"),
        )
        return df

    def build_daily_order(self, demand_forecast, qty_col="quantity_total"):
        from datetime import timedelta

        order_cal = self.order_calendar.clone()

        ws_list = demand_forecast.select("week_start").unique().sort("week_start")["week_start"].to_list()
        if not ws_list:
            return pl.DataFrame()

        min_ws, max_ws = ws_list[0], ws_list[-1]
        n_weeks = len(ws_list)

        # 生成 simulation_date
        date_records = []
        for ws in ws_list:
            for offset in range(7):
                date_records.append({"week_start": ws, "simulation_date": ws + timedelta(days=offset)})
        date_df = pl.DataFrame(date_records)

        # 展开到天级
        demand_forecast = demand_forecast.join(date_df, on="week_start", how="left")
        demand_forecast = demand_forecast.join(
            order_cal.select(
                pl.col("date").cast(pl.Date),
                pl.col("date").cast(pl.Date).alias("_calendar_date"),
                "order_day_flag",
            ),
            left_on="simulation_date", right_on="date", how="left",
        ).with_columns(
            pl.col("_calendar_date").alias("date"),
        ).drop("_calendar_date")

        demand_forecast = demand_forecast.with_columns(
            pl.col("order_day_flag").fill_null(1),
        )
        demand_forecast = demand_forecast.sort(["material", "location", "simulation_date"])

        # flag_summary: 按 week_start 统计 order_day_flag
        order_cal = order_cal.with_columns(pl.col("date").cast(pl.Date))
        _days = (pl.col("date") - min_ws).dt.total_days()
        _idx = _days // 7
        order_cal = order_cal.with_columns(
            pl.when((_idx >= 0) & (_idx < n_weeks))
            .then((min_ws + _idx * pl.duration(days=7)).cast(pl.Date))
            .otherwise(None)
            .alias("week_start"),
        )
        flag_summary = (
            order_cal
            .filter(pl.col("week_start").is_not_null())
            .group_by("week_start")
            .agg(pl.col("order_day_flag").sum().alias("flag_count"))
        )
        demand_forecast = demand_forecast.join(flag_summary, on="week_start", how="left")

        # 计算 base_qty + remainder（先四舍五入避免 float 精度丢失）
        demand_forecast = demand_forecast.with_columns(
            pl.col(qty_col).round(0).cast(pl.Int64).alias("_qty_int"),
        )
        demand_forecast = demand_forecast.with_columns([
            pl.when((pl.col("order_day_flag") == 1) & (pl.col("flag_count") > 0))
            .then(pl.col("_qty_int") // pl.col("flag_count"))
            .otherwise(0)
            .alias("base_qty"),
            (pl.col("_qty_int") % pl.col("flag_count")).alias("remainder"),
        ])

        rank_group = ["material", "location", "week_start"]
        if "order_type" in demand_forecast.columns:
            rank_group.append("order_type")
        ordering_rank = (
            pl.col("order_day_flag").cast(pl.Int64)
            .cum_sum()
            .over(rank_group) - 1
        )
        demand_forecast = demand_forecast.with_columns(
            (pl.col("base_qty") + (
                (pl.col("order_day_flag") == 1) & (ordering_rank < pl.col("remainder"))
            ).cast(pl.Int64)).alias("base_qty"),
        )

        drop_cols = [c for c in [qty_col, "quantity", "_qty_int"] if c in demand_forecast.columns]
        demand_forecast = demand_forecast.drop(drop_cols).rename({"base_qty": "quantity"})
        return demand_forecast

    def adjust_daily_order_ao(self, daily_order):
        result = daily_order.clone()
        result = result.join(self.ao_config, on=["material", "location", "order_type"], how="left")
        result = result.with_columns([
            pl.col("advance_days").fill_null(0),
            pl.col("percent").fill_null(pl.col("ao_percent")),
        ])
        result = result.with_columns(
            (pl.col("quantity") * pl.col("percent") / pl.col("ao_percent"))
            .fill_nan(0).fill_null(0)
            .alias("quantity"),
        )
        result = result.with_columns(
            (pl.col("date") + pl.duration(days=pl.col("advance_days").cast(pl.Int64))).alias("date"),
        )
        result = result.rename({"order_type": "demand_type"})
        return result

    def build_order_df(self, ao_detail):
        sim_date = self._o.simulation_date
        if isinstance(sim_date, pd.Timestamp):
            sim_date_py = sim_date.to_pydatetime()
        else:
            sim_date_py = pd.Timestamp(sim_date).to_pydatetime()

        ao_detail = ao_detail.with_columns([
            (pl.col("cov_quantity_ao_detail") / pl.col("flag_count")).alias("daily_quantity"),
            (pl.lit(sim_date_py) + pl.duration(days=pl.col("advance_days").cast(pl.Int64))).cast(pl.Date).alias("date"),
        ])
        order_df = (
            ao_detail
            .group_by(["date", "material", "location", "order_type", "advance_days"])
            .agg(pl.col("daily_quantity").sum())
            .rename({"daily_quantity": "quantity", "order_type": "demand_type"})
        )
        return order_df

    def merge_with_history(self, order_df):
        sim_date = self._o.simulation_date
        if isinstance(sim_date, pd.Timestamp):
            sim_date_py = sim_date.normalize().to_pydatetime().date()
        else:
            sim_date_py = pd.Timestamp(sim_date).normalize().to_pydatetime().date()

        # 今日订单: simulation_date == today
        if order_df is not None and not order_df.is_empty() and "simulation_date" in order_df.columns:
            today_orders = order_df.filter(pl.col("simulation_date") == pl.lit(sim_date_py))
        else:
            today_orders = order_df.clone() if order_df is not None else pl.DataFrame()

        # 所有有效订单: simulation_date <= today AND date >= today
        if order_df is None or order_df.is_empty():
            all_orders = pl.DataFrame()
        else:
            all_orders = order_df.with_columns(pl.col("date").cast(pl.Date))
            all_orders = all_orders.filter(
                (pl.col("simulation_date") <= pl.lit(sim_date_py))
                & (pl.col("date") >= pl.lit(sim_date_py))
            )
            all_orders = self._deduplicate_orders(all_orders)
            all_orders = self._normalize_orders(all_orders)

        return all_orders, today_orders

    @staticmethod
    def _unify_and_concat(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
        """将两个 DataFrame 统一到同一 schema 后 concat。"""
        # 取并集列名，缺失列补 null
        all_cols = list(dict.fromkeys(left.columns + right.columns))
        _ID_COLS = {"material", "location", "demand_type", "order_type"}
        _DATE_COLS = {"date", "simulation_date"}

        def _align(df: pl.DataFrame) -> pl.DataFrame:
            exprs = []
            for c in all_cols:
                if c not in df.columns:
                    exprs.append(pl.lit(None).alias(c))
                elif c in _ID_COLS:
                    exprs.append(pl.col(c).cast(pl.Utf8))
                elif c in _DATE_COLS:
                    exprs.append(pl.col(c).cast(pl.Date))
                else:
                    exprs.append(pl.col(c).cast(pl.Float64))
            return df.select(exprs)

        return pl.concat([_align(left), _align(right)], how="vertical")

    def apply_orders_consumption(self, forecast_df, orders_df):
        if forecast_df is None or forecast_df.is_empty():
            return pl.DataFrame(schema={"material": pl.Utf8, "location": pl.Utf8, "date": pl.Date, "quantity": pl.Int64})

        # 聚合 forecast
        consumed = forecast_df.with_columns([
            pl.col("material").cast(pl.Utf8),
            pl.col("location").cast(pl.Utf8),
            pl.col("date").cast(pl.Date),
            pl.col("quantity").fill_nan(0).fill_null(0),
        ])

        if orders_df is None or orders_df.is_empty():
            return _normalize_identifiers_polars(consumed)

        # 直接在 polars 上构建索引，不转 pandas
        idx_map = {}
        for idx, (mat, loc, dt) in enumerate(zip(
            consumed["material"].to_list(),
            consumed["location"].to_list(),
            consumed["date"].to_list(),
        )):
            if dt is None:
                continue
            idx_map.setdefault((mat, loc, dt), []).append(idx)

        orders_df = orders_df.with_columns(pl.col("date").cast(pl.Date))
        quantities = consumed["quantity"].to_numpy().astype(float).copy()
        offsets = [0, -1, -2, 1, 2, 3]

        # AO 订单
        ao = orders_df.filter(pl.col("demand_type") == "AO").sort(
            ["date", "advance_days", "quantity", "simulation_date"]
        )
        if not ao.is_empty():
            self._consume_polars(ao, quantities, idx_map, offsets)

        # Normal 订单
        normal = orders_df.filter(pl.col("demand_type") == "normal").sort(
            ["date", "quantity", "simulation_date"]
        )
        if not normal.is_empty():
            self._consume_polars(normal, quantities, idx_map, offsets)

        consumed = consumed.with_columns(
            pl.Series("quantity", quantities.astype(int))
        )
        return _normalize_identifiers_polars(consumed)

    @staticmethod
    def _consume_polars(orders_df, quantities, idx_map, offsets):
        """纯 polars 路径的消耗算法，无 pandas 转换。"""
        from datetime import timedelta
        for row in orders_df.iter_rows(named=True):
            qty = row["quantity"]
            if qty is None or qty != qty or qty <= 0:
                continue
            remaining = int(qty)
            mat = row["material"]
            loc = row["location"]
            order_date = row["date"]
            if order_date is None:
                continue
            if hasattr(order_date, "date"):
                order_date = order_date.date()

            for offset in offsets:
                if remaining <= 0:
                    break
                target_date = order_date + timedelta(days=offset)
                key = (mat, loc, target_date)
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

    def generate_supply_demand_log(self, demand_forecast, consumed_forecast):
        empty_cols = ["date", "material", "location", "quantity", "demand_element"]
        if consumed_forecast.is_empty() or "date" not in consumed_forecast.columns:
            return pl.DataFrame(schema={c: pl.Utf8 for c in empty_cols})

        sim_date = self._o.simulation_date
        if isinstance(sim_date, pd.Timestamp):
            sim_date_py = sim_date.normalize().to_pydatetime().date()
        else:
            sim_date_py = pd.Timestamp(sim_date).normalize().to_pydatetime().date()

        future_cutoff = pd.Timestamp(sim_date) + pd.Timedelta(days=M1_FUTURE_CUTOFF_DAYS)
        future_cutoff_py = future_cutoff.to_pydatetime().date()

        # 统一 cast date 列
        future_demand = consumed_forecast.with_columns(
            pl.col("date").cast(pl.Date),
        ).filter(
            (pl.col("date") > pl.lit(sim_date_py))
            & (pl.col("date") <= pl.lit(future_cutoff_py)),
        )

        if future_demand.is_empty():
            return pl.DataFrame(schema={c: pl.Utf8 for c in empty_cols})

        future_demand = future_demand.group_by(
            ["date", "material", "location"],
        ).agg(pl.col("quantity").sum())
        future_demand = future_demand.with_columns(
            pl.lit("forecast").alias("demand_element"),
        )
        return _normalize_identifiers_polars(future_demand.select(empty_cols))

    def build_summary(self, orders_df, shipment_df, cut_df, supply_demand_df):
        date_val = orders_df["date"][0] if not orders_df.is_empty() else None
        return pl.DataFrame([{
            "Total_Orders": orders_df.height,
            "Total_Shipments": shipment_df.height if isinstance(shipment_df, pl.DataFrame) else len(shipment_df),
            "Total_Cuts": cut_df.height if isinstance(cut_df, pl.DataFrame) else len(cut_df),
            "Total_SupplyDemand": supply_demand_df.height if isinstance(supply_demand_df, pl.DataFrame) else len(supply_demand_df),
            "Date": date_val,
        }])

    def validate_data(self, df, name="data", numeric_columns=None, required_columns=None, strict=False):
        """Polars 版数值校验。"""
        if df.is_empty():
            return df

        if required_columns:
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                raise ValueError(f"validate_data({name}): 缺失必选列 {missing}")

        if numeric_columns is not None:
            cols_to_check = [c for c in numeric_columns if c in df.columns]
        else:
            cols_to_check = [c for c, t in zip(df.columns, df.dtypes) if t.is_numeric()]

        if not cols_to_check:
            return df

        issues = {}
        for col in cols_to_check:
            series = df[col]
            null_count = series.null_count()
            # 检查非 null 值是否有限
            non_null = series.drop_nulls()
            if non_null.dtype.is_float():
                inf_count = int((non_null.is_infinite() | non_null.is_nan()).sum())
            else:
                inf_count = 0
            bad_count = null_count + inf_count
            if bad_count > 0:
                if strict:
                    raise ValueError(
                        f"validate_data({name}): 列 '{col}' 含 {bad_count} 个异常值"
                    )
                df = df.with_columns(
                    pl.col(col).fill_nan(0).fill_null(0),
                )
                issues[col] = bad_count

        if issues:
            logger.warning(
                "validate_data(%s): %d 列存在异常值 %s", name, len(issues), issues,
            )

        return df

    # ---- 私有工具 ----

    def _get_max_advance_days(self):
        ao_cfg = self.ao_config
        if not ao_cfg.is_empty() and "advance_days" in ao_cfg.columns:
            max_val = ao_cfg["advance_days"].max()
            return int(max_val) if max_val is not None else DEFAULT_MAX_ADVANCE_DAYS
        return DEFAULT_MAX_ADVANCE_DAYS

    def _filter_future_orders(self, orders):
        if not orders.is_empty() and "date" in orders.columns:
            orders = orders.with_columns(pl.col("date").cast(pl.Date))
            sim_date = self._o.simulation_date
            if isinstance(sim_date, pd.Timestamp):
                sim_date_py = sim_date.normalize().to_pydatetime().date()
            else:
                sim_date_py = pd.Timestamp(sim_date).normalize().to_pydatetime().date()
            return orders.filter(pl.col("date") >= pl.lit(sim_date_py))
        return orders

    def _deduplicate_orders(self, orders):
        if orders.is_empty():
            return orders
        dedup_keys = [
            c for c in [
                "date", "material", "location", "demand_type",
                "simulation_date", "advance_days", "quantity",
            ]
            if c in orders.columns
        ]
        if dedup_keys:
            return orders.unique(subset=dedup_keys)
        return orders

    def _normalize_orders(self, orders_df):
        if orders_df.is_empty():
            return orders_df
        # quantity 的类型转换交给 validate_data 处理（NaN/inf 安全）
        if "simulation_date" not in orders_df.columns:
            orders_df = orders_df.with_columns(
                pl.col("date").alias("simulation_date"),
            )
        return _normalize_identifiers_polars(orders_df)

    # ---- 发货 / 保存 / 辅助 ----

    def generate_shipments(self, orders_df, daily_detail):
        """Polars 原生发货生成（镜像 shipment.py 逻辑）。"""
        orch = self._o.legacy_orchestrator
        empty = pl.DataFrame()
        if orders_df.is_empty() or orch is None or orch.shipment_valid == 0:
            return empty, empty

        sim_date = self._o.simulation_date
        if isinstance(sim_date, pd.Timestamp):
            sim_date_val = sim_date.normalize()
            sim_date_py = sim_date.normalize().to_pydatetime().date()
        else:
            sim_date_val = pd.Timestamp(sim_date).normalize()
            sim_date_py = pd.Timestamp(sim_date).normalize().to_pydatetime().date()

        # 过滤当日到期订单
        today_orders = orders_df.filter(pl.col("date").cast(pl.Date) == pl.lit(sim_date_py))
        if today_orders.is_empty():
            return empty, empty

        # 构建库存
        inventory = self._build_inventory_polars(orch, sim_date_val)

        # 聚合订单
        ord_agg = today_orders.group_by(["material", "location"]).agg(
            pl.col("quantity").sum().alias("qty_ordered"),
        ).with_columns([
            pl.col("material").cast(pl.Utf8),
            pl.col("location").cast(pl.Utf8),
        ])

        # 合并库存
        merged = ord_agg.join(inventory, on=["material", "location"], how="left").with_columns(
            pl.col("qty_avail").fill_null(0).cast(pl.Int64),
            pl.col("qty_ordered").fill_nan(0).fill_null(0).cast(pl.Int64),
        )
        merged = merged.with_columns([
            pl.min_horizontal(["qty_ordered", "qty_avail"]).alias("shipped"),
            (pl.col("qty_ordered") - pl.min_horizontal(["qty_ordered", "qty_avail"])).alias("cut"),
        ])

        # 构建 shipment_df
        shipment_df = merged.select(
            pl.lit(sim_date_py).alias("date"),
            "material", "location",
            pl.col("shipped").alias("quantity"),
        ).with_columns([
            pl.lit("customer").alias("demand_type"),
        ])
        # 生成 order_id
        date_str = sim_date_val.strftime("%Y%m%d")
        shipment_df = shipment_df.with_columns(
            (pl.lit("ORD_" + date_str + "_") + pl.arange(0, shipment_df.height).cast(pl.Utf8)).alias("order_id"),
        )

        # 构建 cut_df
        cut_df = merged.filter(pl.col("cut") != 0).select(
            pl.lit(sim_date_py).alias("date"),
            "material", "location",
            pl.col("cut").alias("quantity"),
        )

        shipment_df = _normalize_identifiers_polars(shipment_df)
        cut_df = _normalize_identifiers_polars(cut_df)
        return shipment_df, cut_df

    def _build_inventory_polars(self, orch, sim_date):
        """从 orchestrator 构建库存 polars DataFrame。"""
        from ...utils.normalization import normalize_location
        date_str = sim_date.strftime("%Y-%m-%d")

        parts = []
        for view_fn, loc_col in [
            (orch.get_beginning_inventory_view, "location"),
            (orch.get_production_gr_view, "location"),
            (orch.get_delivery_gr_view, "receiving"),
        ]:
            df = view_fn(date_str)
            if df is not None and not df.empty:
                sub = df[["material", loc_col, "quantity"]].copy()
                sub.columns = ["material", "location", "quantity"]
                sub["material"] = sub["material"].astype(str)
                sub["location"] = sub["location"].apply(lambda v: normalize_location(v, mode="any"))
                sub = sub.groupby(["material", "location"], as_index=False)["quantity"].sum()
                parts.append(sub)

        if not parts:
            return pl.DataFrame(schema={"material": pl.Utf8, "location": pl.Utf8, "qty_avail": pl.Int64})

        import pandas as pd_inner
        inv_pd = parts[0]
        for p in parts[1:]:
            inv_pd = inv_pd.merge(p, on=["material", "location"], how="outer", suffixes=("", "_dup"))
            qty_cols = [c for c in inv_pd.columns if c.startswith("quantity")]
            inv_pd["quantity"] = inv_pd[qty_cols].fillna(0).sum(axis=1)
            inv_pd = inv_pd[["material", "location", "quantity"]]

        inv_pd["qty_avail"] = inv_pd["quantity"].fillna(0).astype(int)
        inv_pd = inv_pd[["material", "location", "qty_avail"]]
        return self._to_pl(inv_pd)

    def save_output(self, orders_df, shipment_df, cut_df, supply_demand_df, summary_df):
        """Polars 原生 Excel 输出，datetime 和 quantity 格式化。"""
        import os
        import xlsxwriter
        from datetime import datetime, date

        if self._o.skip_file_output:
            return None

        date_str = pd.Timestamp(self._o.simulation_date).strftime("%Y%m%d")
        output_file = os.path.join(
            self._o.output_dir,
            f"module1_output_{date_str}.xlsx",
        )

        try:
            sheets = {
                "OrderLog": (orders_df, ["date", "material", "location", "demand_type", "quantity", "simulation_date", "advance_days"]),
                "ShipmentLog": (shipment_df, ["date", "material", "location", "quantity", "demand_type", "order_id"]),
                "CutLog": (cut_df, ["date", "material", "location", "quantity"]),
                "SupplyDemandLog": (supply_demand_df, ["date", "material", "location", "quantity", "demand_element"]),
                "Summary": (summary_df, ["Total_Orders","Total_Shipments","Total_Cuts","Total_SupplyDemand","Date"])
            }

            with xlsxwriter.Workbook(output_file) as wb:
                dt_fmt = wb.add_format({"num_format": "yyyy-mm-dd"})
                bold = wb.add_format({"bold": True})

                for sheet_name, (df, cols) in sheets.items():
                    df = self._ensure_cols_pl(df, cols)
                    df = _normalize_identifiers_polars(df)
                    ws = wb.add_worksheet(sheet_name)

                    if df.is_empty():
                        for ci, c in enumerate(cols):
                            ws.write(0, ci, c, bold)
                        continue

                    # 表头
                    for ci, c in enumerate(df.columns):
                        ws.write(0, ci, c, bold)

                    # 数据行
                    for ri, row in enumerate(df.iter_rows(), start=1):
                        for ci, val in enumerate(row):
                            if val is None:
                                continue
                            # datetime / date → 用 datetime 格式写
                            if isinstance(val, (date, datetime)) and not isinstance(val, bool):
                                ws.write_datetime(ri, ci, datetime(val.year, val.month, val.day), dt_fmt)
                            elif isinstance(val, float):
                                ws.write_number(ri, ci, round(val))
                            else:
                                ws.write(ri, ci, val)
        except Exception:
            import traceback
            traceback.print_exc()

        return output_file

    @staticmethod
    def _ensure_cols_pl(df, cols):
        if df is None or (isinstance(df, pl.DataFrame) and df.is_empty()):
            return pl.DataFrame(schema={c: pl.Utf8 for c in cols})
        missing = [c for c in cols if c not in df.columns]
        if missing:
            df = df.with_columns([pl.lit(None).alias(c) for c in missing])
        return df.select(cols)

    def get_order_day_flag(self, order_cal):
        sim_date = self._o.simulation_date
        if isinstance(sim_date, pd.Timestamp):
            sim_date_py = sim_date.normalize().to_pydatetime().date()
        else:
            sim_date_py = pd.Timestamp(sim_date).normalize().to_pydatetime().date()
        filtered = order_cal.filter(
            pl.col("date") == pl.lit(sim_date_py)
        ).select("order_day_flag")
        return filtered.item() if filtered.height > 0 else 0
