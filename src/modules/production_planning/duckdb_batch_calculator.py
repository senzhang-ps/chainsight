# -*- coding: utf-8 -*-
"""Module4 DuckDB optimization"""
import time
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

try:
    import sys, os
    pgsql_db_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pgsql_db')
    if pgsql_db_path not in sys.path:
        sys.path.insert(0, pgsql_db_path)
    from duckdb_integration import get_duckdb_calculator, DuckDBConfig, get_perf_stats
    DUCKDB_INTEGRATION_AVAILABLE = True
except ImportError:
    DUCKDB_INTEGRATION_AVAILABLE = False

def simulate_production_batch_duckdb(plan, pr_cfg, seed=None, run_id=None):
    if plan.empty or 'con_planned_qty' not in plan.columns:
        plan['produced_qty'] = []
        return plan
    if not DUCKDB_INTEGRATION_AVAILABLE or not DuckDBConfig.enabled or len(plan) < DuckDBConfig.min_rows_threshold:
        return _simulate_production_pandas(plan, pr_cfg, seed, run_id)
    calculator = get_duckdb_calculator()
    if calculator is None:
        return _simulate_production_pandas(plan, pr_cfg, seed, run_id)
    t0 = time.perf_counter()
    try:
        pr_map = pr_cfg.set_index(['location', 'line'])['pr'].to_dict()
        
        # 注意：不要对plan进行排序！源码的simulate_production直接按原始顺序处理
        # 排序会导致随机数分配顺序不同，产生不同的produced_qty结果
        
        plan_df = plan.copy()
        plan_df['pr'] = plan_df.apply(lambda r: pr_map.get((r['location'], r['line']), 1.0), axis=1)
        
        n_vals = plan_df['con_planned_qty'].astype(int).values
        p_vals = plan_df['pr'].values
        
        # Vectorized binomial sampling using NumPy
        rng = np.random.RandomState(seed)
        produced = np.array([rng.binomial(int(n), float(p)) if n > 0 else 0 for n, p in zip(n_vals, p_vals)])
        
        plan['produced_qty'] = produced
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f'[M4-Pandas] Production simulation {len(plan)} records: {elapsed_ms:.1f}ms')
        if run_id and DuckDBConfig.collect_stats:
            get_perf_stats().record(run_id, 'simulate_production', 'pandas', len(plan), elapsed_ms)
        return plan
    except Exception as e:
        print(f'[M4-DuckDB] Error, fallback to Pandas: {e}')
        if DuckDBConfig.fallback_on_error:
            return _simulate_production_pandas(plan, pr_cfg, seed, run_id)
        raise

def _simulate_production_pandas(plan, pr_cfg, seed=None, run_id=None):
    if plan.empty or 'con_planned_qty' not in plan.columns:
        plan['produced_qty'] = []
        return plan
    t0 = time.perf_counter()
    
    # 注意：不要对plan进行排序！源码的simulate_production直接按原始顺序处理
    # 排序会导致随机数分配顺序不同，产生不同的produced_qty结果
    
    rng = np.random.RandomState(seed)
    pr_map = pr_cfg.set_index(['location', 'line'])['pr'].to_dict()
    def simulate_row(row):
        pr = pr_map.get((row['location'], row['line']), 1)
        return rng.binomial(int(row['con_planned_qty']), pr)
    plan['produced_qty'] = plan.apply(simulate_row, axis=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f'[M4-Pandas] Production simulation {len(plan)} records: {elapsed_ms:.1f}ms')
    if run_id and DUCKDB_INTEGRATION_AVAILABLE and DuckDBConfig.collect_stats:
        get_perf_stats().record(run_id, 'simulate_production', 'pandas', len(plan), elapsed_ms)
    return plan

def is_duckdb_available():
    return DUCKDB_INTEGRATION_AVAILABLE and DuckDBConfig.enabled

def get_duckdb_config():
    if not DUCKDB_INTEGRATION_AVAILABLE:
        return {'available': False}
    return {'available': True, 'enabled': DuckDBConfig.enabled}
