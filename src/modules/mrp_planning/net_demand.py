"""
Module3 净需求计算模块。

负责计算每日净需求（AO gap、forecast gap和safety gap）。
"""

from typing import Optional, Tuple

import pandas as pd

from .constants import DEMAND_TYPE_AO


def calculate_daily_net_demand(
    material: str,
    location: str,
    date: pd.Timestamp,
    supply_demand_df: pd.DataFrame,
    safety_stock_df: pd.DataFrame,
    beginning_inventory_df: pd.DataFrame,
    in_transit_df: pd.DataFrame,
    delivery_gr_df: pd.DataFrame,
    future_production_df: pd.DataFrame,
    today_shipment_df: pd.DataFrame,
    open_deployment_df: pd.DataFrame,
    downstream_forecast_gap: float,
    downstream_safety_gap: float,
    horizon: int,
    delivery_shipment_df: Optional[pd.DataFrame] = None,
    order_df: Optional[pd.DataFrame] = None,
    downstream_ao_gap: float = 0.0
) -> Tuple[float, float, float]:
    """
    计算每日净需求（AO gap、forecast gap和safety gap）。

    Args:
        material: 物料编码
        location: 地点编码
        date: 计算日期
        supply_demand_df: 供需数据
        safety_stock_df: 安全库存数据
        beginning_inventory_df: 期初库存数据
        in_transit_df: 在途数据
        delivery_gr_df: 收货数据
        future_production_df: 生产数据
        today_shipment_df: 今日发货数据
        open_deployment_df: 开放调拨数据
        downstream_forecast_gap: 下游预测缺口
        downstream_safety_gap: 下游安全库存缺口
        horizon: 计算周期天数
        delivery_shipment_df: 发运记录
        order_df: 订单数据
        downstream_ao_gap: 下游AO缺口

    Returns:
        Tuple[float, float, float]: (ao_gap, fc_gap, ss_gap)
    """
    date = _ensure_timestamp(date)
    horizon = max(1, horizon)
    horizon_end = date + pd.Timedelta(days=horizon)

    try:
        # 过滤数据
        filtered = _filter_all_dataframes(
            material, location, date,
            beginning_inventory_df, in_transit_df,
            delivery_gr_df, future_production_df,
            today_shipment_df, open_deployment_df
        )

        # 计算供给侧
        supply = _calculate_supply_side(
            filtered, date, material, location,
            delivery_shipment_df, open_deployment_df
        )

        # 计算需求侧
        demand = _calculate_demand_side(
            material, location, date, horizon_end,
            order_df, supply_demand_df, safety_stock_df
        )

        # 计算缺口
        return _calculate_gaps(
            supply['total_available'],
            demand['ao_local'], demand['fc_local'], demand['ss_local'],
            downstream_ao_gap, downstream_forecast_gap, downstream_safety_gap
        )

    except Exception as e:
        print(f"Warning: Error in net demand for {material}-{location}: {e}")
        return 0.0, 0.0, 0.0


def _ensure_timestamp(date) -> pd.Timestamp:
    """确保日期为Timestamp类型。"""
    if not isinstance(date, pd.Timestamp):
        return pd.to_datetime(date)
    return date


def _filter_all_dataframes(
    material: str,
    location: str,
    date: pd.Timestamp,
    bi_df: pd.DataFrame,
    it_df: pd.DataFrame,
    dgr_df: pd.DataFrame,
    fp_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    od_df: pd.DataFrame
) -> dict:
    """预过滤所有DataFrame。"""
    return {
        'bi': _filter_by_material_location(bi_df, material, location),
        'it': _filter_by_material_receiving(it_df, material, location),
        'dgr': _filter_by_material_receiving(dgr_df, material, location),
        'fp': _filter_by_material_location(fp_df, material, location),
        'ts': _filter_by_material_location(ts_df, material, location),
        'od': _filter_open_deployment_out(od_df, material, location),
    }


def _filter_by_material_location(
    df: pd.DataFrame,
    material: str,
    location: str
) -> pd.DataFrame:
    """按物料和location过滤。"""
    if df is None or df.empty or 'material' not in df.columns:
        return pd.DataFrame()
    mask = (df['material'] == material) & (df['location'] == location)
    return df[mask]


def _filter_by_material_receiving(
    df: pd.DataFrame,
    material: str,
    location: str
) -> pd.DataFrame:
    """按物料和receiving过滤。"""
    if df is None or df.empty or 'material' not in df.columns:
        return pd.DataFrame()
    mask = (df['material'] == material) & (df['receiving'] == location)
    return df[mask]


def _filter_open_deployment_out(
    df: pd.DataFrame,
    material: str,
    location: str
) -> pd.DataFrame:
    """过滤开放调拨出库。"""
    if df is None or df.empty or 'material' not in df.columns:
        return pd.DataFrame()
    mask = (
        (df['material'] == material) &
        (df['sending'] == location) &
        (df['receiving'] != location)
    )
    return df[mask]


def _calculate_supply_side(
    filtered: dict,
    date: pd.Timestamp,
    material: str,
    location: str,
    delivery_shipment_df: Optional[pd.DataFrame],
    open_deployment_df: pd.DataFrame
) -> dict:
    """计算供给侧。"""
    begin_qty = _get_beginning_inventory(filtered['bi'], date)
    in_transit_qty = _get_quantity_sum(filtered['it'])
    delivery_gr_qty = _get_dated_quantity(filtered['dgr'], date)
    today_prod_qty = _get_today_production(filtered['fp'], date)
    future_prod_qty = _get_future_production(filtered['fp'], date)
    today_ship_qty = _get_dated_quantity(filtered['ts'], date)
    delivery_ship_qty = _get_delivery_shipment(
        delivery_shipment_df, material, location, date
    )
    open_deploy_out = _get_open_deployment_out(filtered['od'])
    open_deploy_in = _get_open_deployment_inbound(
        open_deployment_df, material, location, date
    )

    total = (
        begin_qty + in_transit_qty + delivery_gr_qty +
        today_prod_qty + future_prod_qty + open_deploy_in -
        today_ship_qty - delivery_ship_qty - open_deploy_out
    )

    return {'total_available': total}


def _get_beginning_inventory(df: pd.DataFrame, date: pd.Timestamp) -> float:
    """获取期初库存。"""
    if df.empty:
        return 0.0
    rows = df[pd.to_datetime(df['date']) == pd.to_datetime(date)]
    return float(rows['quantity'].sum()) if not rows.empty else 0.0


def _get_quantity_sum(df: pd.DataFrame) -> float:
    """获取quantity列求和。"""
    if df.empty or 'quantity' not in df.columns:
        return 0.0
    return float(df['quantity'].sum())


def _get_dated_quantity(df: pd.DataFrame, date: pd.Timestamp) -> float:
    """获取指定日期的quantity。"""
    if df.empty or 'date' not in df.columns:
        return 0.0
    rows = df[df['date'] == date]
    return float(rows['quantity'].sum()) if not rows.empty else 0.0


def _get_today_production(df: pd.DataFrame, date: pd.Timestamp) -> float:
    """获取当日生产收货。"""
    if df.empty or 'available_date' not in df.columns:
        return 0.0
    rows = df[df['available_date'] == date]
    if rows.empty:
        return 0.0
    col = 'produced_qty' if 'produced_qty' in rows.columns else 'quantity'
    return float(rows[col].sum()) if col in rows.columns else 0.0


def _get_future_production(df: pd.DataFrame, date: pd.Timestamp) -> float:
    """获取未来生产。"""
    if df.empty or 'available_date' not in df.columns:
        return 0.0
    rows = df[df['available_date'] > date]
    if rows.empty:
        return 0.0

    for col in ['con_planned_qty', 'produced_qty', 'quantity']:
        if col in rows.columns:
            return float(
                pd.to_numeric(rows[col], errors='coerce').fillna(0).sum()
            )
    return 0.0


def _get_delivery_shipment(
    df: Optional[pd.DataFrame],
    material: str,
    location: str,
    date: pd.Timestamp
) -> float:
    """获取发运数量。"""
    if df is None or df.empty:
        return 0.0

    qty_col = _find_column(df, ['quantity', 'shipped_qty'])
    send_col = _find_column(df, ['sending', 'location'])
    date_col = _find_column(df, ['date', 'ship_date'])

    if not all([qty_col, send_col, date_col]):
        return 0.0

    rows = df[
        (df['material'] == material) &
        (df[send_col] == location) &
        (pd.to_datetime(df[date_col]) == date)
    ]
    return float(rows[qty_col].sum()) if not rows.empty else 0.0


def _find_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """查找存在的列名。"""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _get_open_deployment_out(df: pd.DataFrame) -> float:
    """获取开放调拨出库量。"""
    if df.empty:
        return 0.0
    col = 'deployed_qty' if 'deployed_qty' in df.columns else 'quantity'
    return float(df[col].sum()) if col in df.columns else 0.0


def _get_open_deployment_inbound(
    df: pd.DataFrame,
    material: str,
    location: str,
    date: pd.Timestamp
) -> float:
    """获取开放调拨入库量。"""
    if df is None or df.empty:
        return 0.0

    qty_col = _find_column(df, ['deployed_qty', 'quantity'])
    required = ['receiving', 'date', 'material']
    if not qty_col or not all(c in df.columns for c in required):
        return 0.0

    odf = df[
        (df['material'] == material) &
        (df['receiving'] == location)
    ].copy()

    if odf.empty:
        return 0.0

    odf['date'] = pd.to_datetime(odf['date'], errors='coerce')
    future = odf[odf['date'] > date]
    return float(
        pd.to_numeric(future[qty_col], errors='coerce').fillna(0).sum()
    )


def _calculate_demand_side(
    material: str,
    location: str,
    date: pd.Timestamp,
    horizon_end: pd.Timestamp,
    order_df: Optional[pd.DataFrame],
    supply_demand_df: pd.DataFrame,
    safety_stock_df: pd.DataFrame
) -> dict:
    """计算需求侧。"""
    ao_local = _get_ao_demand(order_df, material, location, date, horizon_end)
    fc_local = _get_forecast_demand(
        supply_demand_df, material, location, date, horizon_end
    )
    ss_local = _get_safety_stock(safety_stock_df, material, location, horizon_end)

    return {
        'ao_local': ao_local,
        'fc_local': fc_local,
        'ss_local': ss_local,
    }


def _get_ao_demand(
    order_df: Optional[pd.DataFrame],
    material: str,
    location: str,
    date: pd.Timestamp,
    horizon_end: pd.Timestamp
) -> float:
    """获取AO需求。"""
    if order_df is None or order_df.empty or 'date' not in order_df.columns:
        return 0.0

    order_dates = pd.to_datetime(order_df['date'], errors='coerce')
    mask = (
        (order_df['material'].astype(str) == str(material)) &
        (order_df['location'].astype(str) == str(location)) &
        (order_df.get('demand_type') == DEMAND_TYPE_AO) &
        (order_dates >= date) &
        (order_dates <= horizon_end)
    )
    od = order_df[mask]

    if od.empty or 'quantity' not in od.columns:
        return 0.0
    return float(pd.to_numeric(od['quantity'], errors='coerce').fillna(0).sum())


def _get_forecast_demand(
    df: pd.DataFrame,
    material: str,
    location: str,
    date: pd.Timestamp,
    horizon_end: pd.Timestamp
) -> float:
    """获取预测需求。"""
    if df.empty or 'material' not in df.columns:
        return 0.0

    rows = df[
        (df['material'] == material) &
        (df['location'] == location) &
        (df['date'] >= date) &
        (df['date'] <= horizon_end)
    ]
    return float(
        pd.to_numeric(rows.get('quantity', 0), errors='coerce').fillna(0).sum()
    )


def _get_safety_stock(
    df: pd.DataFrame,
    material: str,
    location: str,
    horizon_end: pd.Timestamp
) -> float:
    """获取安全库存目标。"""
    if df.empty or 'material' not in df.columns:
        return 0.0

    rows = df[
        (df['material'] == material) &
        (df['location'] == location) &
        (df['date'] == horizon_end)
    ]
    if rows.empty or 'safety_stock_qty' not in rows.columns:
        return 0.0
    return float(
        pd.to_numeric(rows['safety_stock_qty'], errors='coerce').fillna(0).sum()
    )


def _calculate_gaps(
    total_available: float,
    ao_local: float,
    fc_local: float,
    ss_local: float,
    downstream_ao_gap: float,
    downstream_forecast_gap: float,
    downstream_safety_gap: float
) -> Tuple[float, float, float]:
    """计算缺口（AO → forecast → safety）。"""
    available = float(total_available)

    # AO缺口
    ao_total = ao_local + float(downstream_ao_gap or 0.0)
    ao_gap = max(ao_total - available, 0.0)
    available = max(available - min(available, ao_total), 0.0)

    # Forecast缺口
    fc_total = fc_local + float(downstream_forecast_gap or 0.0)
    fc_gap = max(fc_total - available, 0.0)
    available = max(available - min(available, fc_total), 0.0)

    # Safety缺口
    ss_total = ss_local + float(downstream_safety_gap or 0.0)
    ss_gap = max(ss_total - available, 0.0)

    return ao_gap, fc_gap, ss_gap
