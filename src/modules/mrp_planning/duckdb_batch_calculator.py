# -*- coding: utf-8 -*-
"""
Module3 DuckDB 批量计算优化模块

提供使用 DuckDB 向量化 SQL 进行批量净需求计算的优化实现。
当节点数量较大时，批量计算比逐节点循环更高效。
"""

import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# 尝试导入 DuckDB 集成模块
try:
    # 使用包路径导入，确保使用同一个模块实例（避免 singleton 问题）
    from pgsql_db.duckdb_integration import (
        get_duckdb_calculator,
        DuckDBConfig,
        get_perf_stats,
    )
    DUCKDB_INTEGRATION_AVAILABLE = True
except ImportError:
    # 回退到路径导入（兼容性）
    try:
        import sys
        import os
        # 添加 pgsql_db 到路径
        pgsql_db_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pgsql_db')
        if pgsql_db_path not in sys.path:
            sys.path.insert(0, pgsql_db_path)
        
        from duckdb_integration import (
            get_duckdb_calculator,
            DuckDBConfig,
            get_perf_stats,
        )
        DUCKDB_INTEGRATION_AVAILABLE = True
    except ImportError:
        DUCKDB_INTEGRATION_AVAILABLE = False


def batch_calculate_net_demand_duckdb(
    nodes: List[Tuple[str, str]],  # [(material, location), ...]
    sim_date: pd.Timestamp,
    beginning_inventory_df: pd.DataFrame,
    in_transit_df: pd.DataFrame,
    delivery_gr_df: pd.DataFrame,
    future_production_df: pd.DataFrame,
    today_shipment_df: pd.DataFrame,
    open_deployment_df: pd.DataFrame,
    supply_demand_df: pd.DataFrame,
    safety_stock_df: pd.DataFrame,
    order_df: Optional[pd.DataFrame],
    downstream_gaps: Dict[Tuple[str, str], Dict[str, float]],
    horizons: Dict[Tuple[str, str], int],
    delivery_shipment_df: Optional[pd.DataFrame] = None,
    run_id: Optional[str] = None,
) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
    """
    使用 DuckDB 批量计算多个节点的净需求
    
    参数：
        nodes: 节点列表 [(material, location), ...]
        sim_date: 模拟日期
        beginning_inventory_df: 期初库存
        in_transit_df: 在途库存
        delivery_gr_df: 收货数据
        future_production_df: 生产计划
        today_shipment_df: 当日发货
        open_deployment_df: 开放调拨
        supply_demand_df: 供需数据
        safety_stock_df: 安全库存
        order_df: 订单数据
        downstream_gaps: 下游缺口 {(material, location): {'AO': x, 'FC': y, 'SS': z}}
        horizons: 各节点horizon {(material, location): horizon_days}
        delivery_shipment_df: 发运记录
        run_id: 运行ID（用于性能统计）
    
    返回：
        节点缺口字典 {(material, location): (ao_gap, fc_gap, ss_gap)}
    """
    if not DUCKDB_INTEGRATION_AVAILABLE or not DuckDBConfig.enabled:
        return _batch_calculate_pandas(
            nodes, sim_date, beginning_inventory_df, in_transit_df,
            delivery_gr_df, future_production_df, today_shipment_df,
            open_deployment_df, supply_demand_df, safety_stock_df,
            order_df, downstream_gaps, horizons, delivery_shipment_df, run_id
        )
    
    calculator = get_duckdb_calculator()
    if calculator is None or len(nodes) < DuckDBConfig.min_rows_threshold:
        return _batch_calculate_pandas(
            nodes, sim_date, beginning_inventory_df, in_transit_df,
            delivery_gr_df, future_production_df, today_shipment_df,
            open_deployment_df, supply_demand_df, safety_stock_df,
            order_df, downstream_gaps, horizons, delivery_shipment_df, run_id
        )
    
    t0 = time.perf_counter()
    
    try:
        # 构建节点DataFrame
        nodes_df = pd.DataFrame(nodes, columns=['material', 'location'])
        nodes_df['material'] = nodes_df['material'].astype(str)
        nodes_df['location'] = nodes_df['location'].astype(str)
        
        # 添加horizon
        nodes_df['horizon'] = nodes_df.apply(
            lambda r: horizons.get((r['material'], r['location']), 1),
            axis=1
        )
        nodes_df['horizon_end'] = sim_date + pd.to_timedelta(nodes_df['horizon'], unit='D')
        
        # 添加下游缺口
        nodes_df['downstream_ao'] = nodes_df.apply(
            lambda r: downstream_gaps.get((r['material'], r['location']), {}).get('AO', 0.0),
            axis=1
        )
        nodes_df['downstream_fc'] = nodes_df.apply(
            lambda r: downstream_gaps.get((r['material'], r['location']), {}).get('FC', 0.0),
            axis=1
        )
        nodes_df['downstream_ss'] = nodes_df.apply(
            lambda r: downstream_gaps.get((r['material'], r['location']), {}).get('SS', 0.0),
            axis=1
        )
        
        # 预聚合各数据源
        # 期初库存
        bi_agg = _agg_by_ml(beginning_inventory_df, 'quantity', sim_date, date_filter='eq')
        # 在途
        it_agg = _agg_by_mr(in_transit_df, 'quantity')
        # 收货
        dgr_agg = _agg_by_mr(delivery_gr_df, 'quantity', sim_date, date_filter='eq')
        # 生产 - 当日
        fp_today_agg = _agg_production(future_production_df, sim_date, 'today')
        # 生产 - 未来
        fp_future_agg = _agg_production(future_production_df, sim_date, 'future')
        # 当日发货
        ts_agg = _agg_by_ml(today_shipment_df, 'quantity', sim_date, date_filter='eq')
        # 开放调拨出库
        od_out_agg = _agg_open_deployment_out(open_deployment_df)
        # 开放调拨入库（未来）
        od_in_agg = _agg_open_deployment_in(open_deployment_df, sim_date)
        # 发运
        ds_agg = _agg_delivery_shipment(delivery_shipment_df, sim_date)
        
        # AO 需求
        ao_agg = _agg_ao_demand(order_df, sim_date)
        # 预测需求
        fc_agg = _agg_forecast_demand(supply_demand_df, sim_date)
        # 安全库存
        ss_agg = _agg_safety_stock(safety_stock_df)
        
        # 注册所有表到 DuckDB
        calculator.conn.register('nodes', nodes_df)
        calculator.conn.register('bi', bi_agg)
        calculator.conn.register('it', it_agg)
        calculator.conn.register('dgr', dgr_agg)
        calculator.conn.register('fp_today', fp_today_agg)
        calculator.conn.register('fp_future', fp_future_agg)
        calculator.conn.register('ts', ts_agg)
        calculator.conn.register('od_out', od_out_agg)
        calculator.conn.register('od_in', od_in_agg)
        calculator.conn.register('ds', ds_agg)
        calculator.conn.register('ao', ao_agg)
        calculator.conn.register('fc', fc_agg)
        calculator.conn.register('ss', ss_agg)
        
        # 执行批量计算 SQL
        # 注意：AO 和 FC 需要按各节点的 horizon_end 进行过滤
        # ao 和 fc 表现在保留了 date 维度，需要在 SQL 中进行聚合和过滤
        # 使用 CAST 确保日期类型一致性，避免 datetime64 vs Timestamp 比较问题
        result = calculator.conn.execute("""
            WITH 
            -- 先按 horizon_end 聚合 AO 需求
            ao_filtered AS (
                SELECT 
                    n.material,
                    n.location,
                    COALESCE(SUM(ao.qty), 0) as qty
                FROM nodes n
                LEFT JOIN ao ON n.material = ao.material 
                    AND n.location = ao.location 
                    AND CAST(ao.date AS TIMESTAMP) <= CAST(n.horizon_end AS TIMESTAMP)
                GROUP BY n.material, n.location
            ),
            -- 先按 horizon_end 聚合 FC 需求
            fc_filtered AS (
                SELECT 
                    n.material,
                    n.location,
                    COALESCE(SUM(fc.qty), 0) as qty
                FROM nodes n
                LEFT JOIN fc ON n.material = fc.material 
                    AND n.location = fc.location 
                    AND CAST(fc.date AS TIMESTAMP) <= CAST(n.horizon_end AS TIMESTAMP)
                GROUP BY n.material, n.location
            ),
            supply_calc AS (
                SELECT 
                    n.material,
                    n.location,
                    n.horizon_end,
                    n.downstream_ao,
                    n.downstream_fc,
                    n.downstream_ss,
                    COALESCE(bi.qty, 0) as begin_inv,
                    COALESCE(it.qty, 0) as in_transit,
                    COALESCE(dgr.qty, 0) as delivery_gr,
                    COALESCE(fp_t.qty, 0) as prod_today,
                    COALESCE(fp_f.qty, 0) as prod_future,
                    COALESCE(ts.qty, 0) as shipment,
                    COALESCE(od_o.qty, 0) as od_out,
                    COALESCE(od_i.qty, 0) as od_in,
                    COALESCE(ds.qty, 0) as del_ship
                FROM nodes n
                LEFT JOIN bi ON n.material = bi.material AND n.location = bi.location
                LEFT JOIN it ON n.material = it.material AND n.location = it.receiving
                LEFT JOIN dgr ON n.material = dgr.material AND n.location = dgr.receiving
                LEFT JOIN fp_today fp_t ON n.material = fp_t.material AND n.location = fp_t.location
                LEFT JOIN fp_future fp_f ON n.material = fp_f.material AND n.location = fp_f.location
                LEFT JOIN ts ON n.material = ts.material AND n.location = ts.location
                LEFT JOIN od_out od_o ON n.material = od_o.material AND n.location = od_o.sending
                LEFT JOIN od_in od_i ON n.material = od_i.material AND n.location = od_i.receiving
                LEFT JOIN ds ON n.material = ds.material AND n.location = ds.sending
            ),
            demand_calc AS (
                SELECT 
                    s.*,
                    -- 总供给
                    s.begin_inv + s.in_transit + s.delivery_gr + 
                    s.prod_today + s.prod_future + s.od_in -
                    s.shipment - s.od_out - s.del_ship as total_supply,
                    -- 获取需求（已按horizon_end过滤）
                    COALESCE(ao_f.qty, 0) as ao_local,
                    COALESCE(fc_f.qty, 0) as fc_local,
                    COALESCE(ss.qty, 0) as ss_local
                FROM supply_calc s
                LEFT JOIN ao_filtered ao_f ON s.material = ao_f.material AND s.location = ao_f.location
                LEFT JOIN fc_filtered fc_f ON s.material = fc_f.material AND s.location = fc_f.location
                LEFT JOIN ss ON s.material = ss.material 
                    AND s.location = ss.location 
                    AND s.horizon_end = ss.date
            ),
            gap_calc AS (
                SELECT 
                    material,
                    location,
                    total_supply,
                    ao_local + downstream_ao as total_ao,
                    fc_local + downstream_fc as total_fc,
                    ss_local + downstream_ss as total_ss,
                    -- AO gap
                    GREATEST(0, (ao_local + downstream_ao) - total_supply) as ao_gap,
                    -- Remaining after AO
                    GREATEST(0, total_supply - (ao_local + downstream_ao)) as remaining_after_ao
                FROM demand_calc
            )
            SELECT 
                material,
                location,
                ao_gap,
                GREATEST(0, total_fc - remaining_after_ao) as fc_gap,
                GREATEST(0, total_ss - GREATEST(0, remaining_after_ao - total_fc)) as ss_gap
            FROM gap_calc
        """).fetchdf()
        
        # 清理注册的表
        for tbl in ['nodes', 'bi', 'it', 'dgr', 'fp_today', 'fp_future', 
                    'ts', 'od_out', 'od_in', 'ds', 'ao', 'fc', 'ss']:
            calculator.conn.unregister(tbl)
        
        # 转换为字典
        gaps = {}
        for _, row in result.iterrows():
            key = (str(row['material']), str(row['location']))
            gaps[key] = (float(row['ao_gap']), float(row['fc_gap']), float(row['ss_gap']))
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        if run_id and DuckDBConfig.collect_stats:
            get_perf_stats().record(run_id, 'batch_net_demand', 'duckdb', 
                                   len(nodes), elapsed_ms)
        
        return gaps
        
    except Exception as e:
        import traceback
        if DuckDBConfig.fallback_on_error:
            return _batch_calculate_pandas(
                nodes, sim_date, beginning_inventory_df, in_transit_df,
                delivery_gr_df, future_production_df, today_shipment_df,
                open_deployment_df, supply_demand_df, safety_stock_df,
                order_df, downstream_gaps, horizons, delivery_shipment_df, run_id
            )
        raise


def _batch_calculate_pandas(
    nodes: List[Tuple[str, str]],
    sim_date: pd.Timestamp,
    beginning_inventory_df: pd.DataFrame,
    in_transit_df: pd.DataFrame,
    delivery_gr_df: pd.DataFrame,
    future_production_df: pd.DataFrame,
    today_shipment_df: pd.DataFrame,
    open_deployment_df: pd.DataFrame,
    supply_demand_df: pd.DataFrame,
    safety_stock_df: pd.DataFrame,
    order_df: Optional[pd.DataFrame],
    downstream_gaps: Dict[Tuple[str, str], Dict[str, float]],
    horizons: Dict[Tuple[str, str], int],
    delivery_shipment_df: Optional[pd.DataFrame] = None,
    run_id: Optional[str] = None,
) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
    """Pandas 批量计算（作为回退方案）"""
    t0 = time.perf_counter()
    
    # 导入原始计算函数
    from .net_demand import calculate_daily_net_demand
    
    gaps = {}
    for material, location in nodes:
        horizon = horizons.get((material, location), 1)
        ds_gap = downstream_gaps.get((material, location), {})
        
        ao_gap, fc_gap, ss_gap = calculate_daily_net_demand(
            material, location, sim_date,
            supply_demand_df, safety_stock_df,
            beginning_inventory_df, in_transit_df,
            delivery_gr_df, future_production_df,
            today_shipment_df, open_deployment_df,
            ds_gap.get('FC', 0.0), ds_gap.get('SS', 0.0), horizon,
            delivery_shipment_df=delivery_shipment_df,
            order_df=order_df,
            downstream_ao_gap=ds_gap.get('AO', 0.0)
        )
        gaps[(material, location)] = (ao_gap, fc_gap, ss_gap)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    if run_id and DUCKDB_INTEGRATION_AVAILABLE and DuckDBConfig.collect_stats:
        get_perf_stats().record(run_id, 'batch_net_demand', 'pandas', 
                               len(nodes), elapsed_ms)
    
    return gaps


# ============================================================================
# 辅助聚合函数
# ============================================================================

def _empty_df_with_date() -> pd.DataFrame:
    """返回具有正确类型的空 DataFrame（包含 date 列）"""
    result = pd.DataFrame({
        'material': pd.Series([], dtype='str'),
        'location': pd.Series([], dtype='str'),
        'date': pd.Series([], dtype='datetime64[ns]'),
        'qty': pd.Series([], dtype='float64')
    })
    return result


def _empty_df_ml() -> pd.DataFrame:
    """返回具有正确类型的空 DataFrame（material, location, qty）"""
    return pd.DataFrame({
        'material': pd.Series([], dtype='str'),
        'location': pd.Series([], dtype='str'),
        'qty': pd.Series([], dtype='float64')
    })


def _empty_df_mr() -> pd.DataFrame:
    """返回具有正确类型的空 DataFrame（material, receiving, qty）"""
    return pd.DataFrame({
        'material': pd.Series([], dtype='str'),
        'receiving': pd.Series([], dtype='str'),
        'qty': pd.Series([], dtype='float64')
    })


def _empty_df_ms() -> pd.DataFrame:
    """返回具有正确类型的空 DataFrame（material, sending, qty）"""
    return pd.DataFrame({
        'material': pd.Series([], dtype='str'),
        'sending': pd.Series([], dtype='str'),
        'qty': pd.Series([], dtype='float64')
    })


def _agg_by_ml(df: pd.DataFrame, qty_col: str, 
               date: pd.Timestamp = None, date_filter: str = None) -> pd.DataFrame:
    """按 (material, location) 聚合"""
    if df is None or df.empty:
        return _empty_df_ml()
    
    df = df.copy()
    df['material'] = df['material'].astype(str)
    df['location'] = df['location'].astype(str)
    
    if date_filter and date is not None and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if date_filter == 'eq':
            df = df[df['date'] == date]
        elif date_filter == 'gt':
            df = df[df['date'] > date]
        elif date_filter == 'gte':
            df = df[df['date'] >= date]
    
    if qty_col not in df.columns:
        return _empty_df_ml()
    
    agg = df.groupby(['material', 'location'])[qty_col].sum().reset_index()
    agg.columns = ['material', 'location', 'qty']
    return agg


def _agg_by_mr(df: pd.DataFrame, qty_col: str,
               date: pd.Timestamp = None, date_filter: str = None) -> pd.DataFrame:
    """按 (material, receiving) 聚合"""
    if df is None or df.empty or 'receiving' not in df.columns:
        return _empty_df_mr()
    
    df = df.copy()
    df['material'] = df['material'].astype(str)
    df['receiving'] = df['receiving'].astype(str)
    
    if date_filter and date is not None and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if date_filter == 'eq':
            df = df[df['date'] == date]
    
    if qty_col not in df.columns:
        return _empty_df_mr()
    
    agg = df.groupby(['material', 'receiving'])[qty_col].sum().reset_index()
    agg.columns = ['material', 'receiving', 'qty']
    return agg


def _agg_production(df: pd.DataFrame, date: pd.Timestamp, mode: str) -> pd.DataFrame:
    """聚合生产数据"""
    if df is None or df.empty or 'available_date' not in df.columns:
        return _empty_df_ml()
    
    df = df.copy()
    df['material'] = df['material'].astype(str)
    df['location'] = df['location'].astype(str)
    df['available_date'] = pd.to_datetime(df['available_date'], errors='coerce')
    
    if mode == 'today':
        df = df[df['available_date'] == date]
        qty_col = 'produced_qty' if 'produced_qty' in df.columns else 'quantity'
    else:  # future
        df = df[df['available_date'] > date]
        for col in ['con_planned_qty', 'produced_qty', 'quantity']:
            if col in df.columns:
                qty_col = col
                break
        else:
            return _empty_df_ml()
    
    if qty_col not in df.columns:
        return _empty_df_ml()
    
    df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
    agg = df.groupby(['material', 'location'])[qty_col].sum().reset_index()
    agg.columns = ['material', 'location', 'qty']
    return agg


def _agg_open_deployment_out(df: pd.DataFrame) -> pd.DataFrame:
    """聚合开放调拨出库"""
    if df is None or df.empty:
        return _empty_df_ms()
    
    df = df.copy()
    df['material'] = df['material'].astype(str)
    df['sending'] = df['sending'].astype(str)
    
    # 只计算跨节点的
    if 'receiving' in df.columns:
        df['receiving'] = df['receiving'].astype(str)
        df = df[df['sending'] != df['receiving']]
    
    qty_col = 'deployed_qty' if 'deployed_qty' in df.columns else 'quantity'
    if qty_col not in df.columns:
        return _empty_df_ms()
    
    agg = df.groupby(['material', 'sending'])[qty_col].sum().reset_index()
    agg.columns = ['material', 'sending', 'qty']
    return agg


def _agg_open_deployment_in(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """聚合开放调拨入库（未来）
    
    说明： 必须与 Pandas 行为保持一致 in net_demand.py _get_open_deployment_inbound()
    - Filter for rows where date > sim_date (仅未来数据)
    - 若未找到日期列，则返回空结果 (Pandas returns 0 when 'date' not in columns)
    """
    if df is None or df.empty or 'receiving' not in df.columns:
        return _empty_df_mr()
    
    df = df.copy()
    df['material'] = df['material'].astype(str)
    df['receiving'] = df['receiving'].astype(str)
    
    # 查找日期列：同时检查 `date` 与 `planned_deployment_date`
    date_col = None
    for col in ['date', 'planned_deployment_date']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        # 未找到日期列时按 Pandas 行为返回空结果
        return _empty_df_mr()
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df[df[date_col] > date]  # 仅保留未来日期
    
    if df.empty:
        return _empty_df_mr()
    
    qty_col = 'deployed_qty' if 'deployed_qty' in df.columns else 'quantity'
    if qty_col not in df.columns:
        return _empty_df_mr()
    
    df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
    agg = df.groupby(['material', 'receiving'])[qty_col].sum().reset_index()
    agg.columns = ['material', 'receiving', 'qty']
    return agg


def _agg_delivery_shipment(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """聚合发运数据"""
    if df is None or df.empty:
        return _empty_df_ms()
    
    df = df.copy()
    df['material'] = df['material'].astype(str)
    
    send_col = 'sending' if 'sending' in df.columns else 'location'
    if send_col not in df.columns:
        return _empty_df_ms()
    
    df['sending'] = df[send_col].astype(str)
    
    date_col = 'date' if 'date' in df.columns else 'ship_date'
    if date_col in df.columns:
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[df['date'] == date]
    
    qty_col = 'quantity' if 'quantity' in df.columns else 'shipped_qty'
    if qty_col not in df.columns:
        return _empty_df_ms()
    
    agg = df.groupby(['material', 'sending'])[qty_col].sum().reset_index()
    agg.columns = ['material', 'sending', 'qty']
    return agg


def _agg_ao_demand(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """聚合AO 需求 - 保留日期维度用于后续horizon_end过滤"""
    if df is None or df.empty:
        return _empty_df_with_date()
    
    df = df.copy()
    df['material'] = df['material'].astype(str)
    df['location'] = df['location'].astype(str)
    
    # 过滤AO类型
    if 'demand_type' in df.columns:
        df = df[df['demand_type'] == 'AO']
    
    # 如果过滤后为空，返回正确类型的空 DataFrame
    if df.empty:
        return _empty_df_with_date()
    
    # 日期过滤 - 只取 >= sim_date 的数据
    # horizon_end 过滤将在 SQL 中按各节点进行
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['date'] >= date]
    else:
        return _empty_df_with_date()
    
    # 如果过滤后为空，返回正确类型的空 DataFrame
    if df.empty:
        return _empty_df_with_date()
    
    if 'quantity' not in df.columns:
        return _empty_df_with_date()
    
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
    # 保留日期维度，用于后续按 horizon_end 过滤
    agg = df.groupby(['material', 'location', 'date'])['quantity'].sum().reset_index()
    agg.columns = ['material', 'location', 'date', 'qty']
    return agg


def _agg_forecast_demand(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """聚合预测需求 - 保留日期维度用于后续horizon_end过滤"""
    if df is None or df.empty:
        return _empty_df_with_date()
    
    df = df.copy()
    df['material'] = df['material'].astype(str)
    df['location'] = df['location'].astype(str)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['date'] >= date]
    else:
        return _empty_df_with_date()
    
    # 如果过滤后为空，返回正确类型的空 DataFrame
    if df.empty:
        return _empty_df_with_date()
    
    if 'quantity' not in df.columns:
        return _empty_df_with_date()
    
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
    # 保留日期维度，用于后续按 horizon_end 过滤
    agg = df.groupby(['material', 'location', 'date'])['quantity'].sum().reset_index()
    agg.columns = ['material', 'location', 'date', 'qty']
    return agg


def _agg_safety_stock(df: pd.DataFrame) -> pd.DataFrame:
    """聚合安全库存（保留日期维度）"""
    if df is None or df.empty:
        return _empty_df_with_date()
    
    df = df.copy()
    df['material'] = df['material'].astype(str)
    df['location'] = df['location'].astype(str)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        return _empty_df_with_date()
    
    qty_col = 'safety_stock_qty' if 'safety_stock_qty' in df.columns else 'quantity'
    if qty_col not in df.columns:
        return _empty_df_with_date()
    
    df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
    agg = df.groupby(['material', 'location', 'date'])[qty_col].sum().reset_index()
    agg.columns = ['material', 'location', 'date', 'qty']
    return agg
