# -*- coding: utf-8 -*-
"""Module6 DuckDB batch delay sampling optimization"""
import time
from typing import List, Tuple, Optional
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

def batch_sample_delivery_delays_duckdb(
    routes: List[Tuple[str, str]],
    dist_df: pd.DataFrame,
    seed: Optional[int] = None,
    run_id: Optional[str] = None
) -> np.ndarray:
    """
    Batch sample delivery delays for multiple routes using vectorized operations.
    
    Args:
        routes: List of (sending, receiving) tuples
        dist_df: Delay distribution DataFrame
        seed: Random seed for reproducibility
        run_id: Run ID for performance tracking
        
    Returns:
        Array of sampled delays (one per route)
    """
    if not routes:
        return np.array([])
    
    # Check if optimization should be used
    if not DUCKDB_INTEGRATION_AVAILABLE or not DuckDBConfig.enabled or len(routes) < 10:
        return _batch_sample_delays_pandas(routes, dist_df, seed, run_id)
    
    calculator = get_duckdb_calculator()
    if calculator is None:
        return _batch_sample_delays_pandas(routes, dist_df, seed, run_id)
    
    t0 = time.perf_counter()
    try:
        delays = _vectorized_delay_sampling(routes, dist_df, seed)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f'[M6-DuckDB] Batch delay sampling {len(routes)} routes: {elapsed_ms:.1f}ms')
        
        if run_id and DuckDBConfig.collect_stats:
            get_perf_stats().record(run_id, 'batch_sample_delays', 'duckdb', len(routes), elapsed_ms)
        
        return delays
    except Exception as e:
        print(f'[M6-DuckDB] Error, fallback to Pandas: {e}')
        if DuckDBConfig.fallback_on_error:
            return _batch_sample_delays_pandas(routes, dist_df, seed, run_id)
        raise

def _vectorized_delay_sampling(
    routes: List[Tuple[str, str]],
    dist_df: pd.DataFrame,
    seed: Optional[int]
) -> np.ndarray:
    """
    Vectorized delay sampling implementation.
    
    Args:
        routes: List of (sending, receiving) tuples
        dist_df: Delay distribution DataFrame
        seed: Random seed
        
    Returns:
        Array of sampled delays
    """
    if dist_df is None or dist_df.empty:
        return np.zeros(len(routes), dtype=int)
    
    required_cols = {'delay_days', 'probability', 'sending', 'receiving'}
    if not required_cols.issubset(set(dist_df.columns)):
        return np.zeros(len(routes), dtype=int)
    
    rng = np.random.RandomState(seed)
    delays = np.zeros(len(routes), dtype=int)
    
    # Build lookup cache for delay distributions
    dist_cache = {}
    for _, row in dist_df.iterrows():
        key = (str(row['sending']), str(row['receiving']))
        if key not in dist_cache:
            dist_cache[key] = {'delays': [], 'probs': []}
        dist_cache[key]['delays'].append(int(row['delay_days']))
        dist_cache[key]['probs'].append(float(row['probability']))
    
    # Check for global fallback rule
    global_key = ('ALL', 'ALL')
    has_global = any(
        str(row['sending']).upper() == 'ALL' and str(row['receiving']).upper() == 'ALL'
        for _, row in dist_df.iterrows()
    )
    if has_global:
        global_dist = {'delays': [], 'probs': []}
        for _, row in dist_df.iterrows():
            if str(row['sending']).upper() == 'ALL' and str(row['receiving']).upper() == 'ALL':
                global_dist['delays'].append(int(row['delay_days']))
                global_dist['probs'].append(float(row['probability']))
        dist_cache[global_key] = global_dist
    
    # Convert probs to numpy arrays and normalize
    for key in dist_cache:
        probs = np.array(dist_cache[key]['probs'], dtype=float)
        if probs.sum() > 0:
            probs = probs / probs.sum()
        dist_cache[key]['probs'] = probs
        dist_cache[key]['delays'] = np.array(dist_cache[key]['delays'], dtype=np.int32)
    
    # NumPy vectorized sampling
    rng = np.random.RandomState(seed)
    delays = np.zeros(len(routes), dtype=int)
    
    for i, (sending, receiving) in enumerate(routes):
        key = (sending, receiving)
        
        # Try exact match first
        if key in dist_cache:
            dist = dist_cache[key]
        # Fall back to global rule
        elif global_key in dist_cache:
            dist = dist_cache[global_key]
        else:
            delays[i] = 0
            continue
        
        # Sample from distribution
        probs = dist['probs']
        if probs.sum() > 0:
            delays[i] = rng.choice(dist['delays'], p=probs)
        else:
            delays[i] = 0
    
    return delays

def _batch_sample_delays_pandas(
    routes: List[Tuple[str, str]],
    dist_df: pd.DataFrame,
    seed: Optional[int],
    run_id: Optional[str]
) -> np.ndarray:
    """
    Pandas fallback for batch delay sampling.
    
    Args:
        routes: List of (sending, receiving) tuples
        dist_df: Delay distribution DataFrame
        seed: Random seed
        run_id: Run ID for performance tracking
        
    Returns:
        Array of sampled delays
    """
    t0 = time.perf_counter()
    
    # Import the original single-record function
    from .delivery_processor import sample_delivery_delay
    
    # Sample delays one by one (DO NOT modify random state)
    # Module 6 relies on global random state continuity
    delays = np.array([
        sample_delivery_delay(sending, receiving, dist_df)
        for sending, receiving in routes
    ])
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f'[M6-Pandas] Batch delay sampling {len(routes)} routes: {elapsed_ms:.1f}ms')
    
    if run_id and DUCKDB_INTEGRATION_AVAILABLE and DuckDBConfig.collect_stats:
        get_perf_stats().record(run_id, 'batch_sample_delays', 'pandas', len(routes), elapsed_ms)
    
    return delays

def is_duckdb_available():
    """Check if DuckDB integration is available and enabled."""
    return DUCKDB_INTEGRATION_AVAILABLE and DuckDBConfig.enabled

def get_duckdb_config():
    """Get DuckDB configuration info."""
    if not DUCKDB_INTEGRATION_AVAILABLE:
        return {'available': False}
    return {'available': True, 'enabled': DuckDBConfig.enabled}
