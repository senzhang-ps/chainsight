# -*- coding: utf-8 -*-
"""
高性能数据处理层
结合PostgreSQL存储和DuckDB向量化计算的优化实现
"""

import duckdb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import time
from functools import lru_cache
from contextlib import contextmanager


class OptimizedDataProcessor:
    """
    优化的数据处理器
    
    核心优化策略：
    1. DuckDB直连PostgreSQL，减少数据复制
    2. 预建索引视图，避免重复过滤
    3. 向量化计算替代Python循环
    4. Parquet缓存热点数据
    """
    
    def __init__(
        self,
        pg_connection_string: str,
        cache_dir: Optional[str] = None,
        memory_limit: str = "4GB",
        threads: int = 4
    ):
        """
        初始化处理器
        
        Args:
            pg_connection_string: PostgreSQL连接字符串
            cache_dir: Parquet缓存目录
            memory_limit: DuckDB内存限制
            threads: 并行线程数
        """
        self.pg_conn_str = pg_connection_string
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.memory_limit = memory_limit
        self.threads = threads
        
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._index_cache: Dict[str, Any] = {}
        self._pg_attached = False
        
        # 性能统计
        self._stats = {
            'queries_executed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_query_time': 0.0,
            'rows_processed': 0
        }
    
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """获取DuckDB连接"""
        if self._conn is None:
            self._conn = duckdb.connect(":memory:")
            self._configure_duckdb()
        return self._conn
    
    def _configure_duckdb(self):
        """配置DuckDB优化参数"""
        self._conn.execute(f"SET threads TO {self.threads}")
        self._conn.execute(f"SET memory_limit = '{self.memory_limit}'")
        # 启用进度条（大查询时有用）
        self._conn.execute("SET enable_progress_bar = true")
        # 优化字符串处理
        self._conn.execute("SET preserve_insertion_order = false")
    
    def attach_postgres(self, schema: str = "public"):
        """
        直接附加PostgreSQL数据库
        
        这是关键优化：DuckDB可以直接查询PostgreSQL，
        无需先将数据加载到Pandas
        """
        if self._pg_attached:
            return
        
        # 安装并加载postgres扩展
        self.conn.execute("INSTALL postgres")
        self.conn.execute("LOAD postgres")
        
        # 附加PostgreSQL数据库
        self.conn.execute(f"""
            ATTACH '{self.pg_conn_str}' AS pg (TYPE postgres, SCHEMA '{schema}')
        """)
        self._pg_attached = True
        print(f"✅ PostgreSQL已附加到DuckDB")
    
    def close(self):
        """关闭连接"""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._pg_attached = False
        self._index_cache.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    # ==================== 索引构建 ====================
    
    def build_groupby_index(
        self,
        df: pd.DataFrame,
        key_columns: List[str],
        index_name: str
    ) -> Dict[tuple, pd.DataFrame]:
        """
        预建GroupBy索引，避免重复过滤
        
        Args:
            df: 源DataFrame
            key_columns: 索引键列
            index_name: 索引名称（用于缓存）
        
        Returns:
            键到DataFrame的映射字典
        """
        if index_name in self._index_cache:
            self._stats['cache_hits'] += 1
            return self._index_cache[index_name]
        
        self._stats['cache_misses'] += 1
        
        # 使用DuckDB构建索引（比Pandas groupby更快）
        self.conn.register(f'_idx_{index_name}', df)
        
        # 获取所有唯一键组合
        key_cols_str = ', '.join(key_columns)
        unique_keys = self.conn.execute(f"""
            SELECT DISTINCT {key_cols_str} 
            FROM _idx_{index_name}
        """).fetchdf()
        
        # 构建索引字典
        index_dict = {}
        grouped = df.groupby(key_columns, sort=False)
        for key in unique_keys.itertuples(index=False):
            key_tuple = tuple(key)
            try:
                index_dict[key_tuple] = grouped.get_group(key_tuple)
            except KeyError:
                continue
        
        self.conn.unregister(f'_idx_{index_name}')
        self._index_cache[index_name] = index_dict
        
        return index_dict
    
    def build_lookup_dict(
        self,
        df: pd.DataFrame,
        key_columns: List[str],
        value_column: str,
        index_name: str
    ) -> Dict[tuple, Any]:
        """
        构建快速查找字典
        
        Args:
            df: 源DataFrame
            key_columns: 键列
            value_column: 值列
            index_name: 索引名称
        
        Returns:
            键到值的映射字典
        """
        cache_key = f"{index_name}_lookup"
        if cache_key in self._index_cache:
            self._stats['cache_hits'] += 1
            return self._index_cache[cache_key]
        
        self._stats['cache_misses'] += 1
        
        # 使用DuckDB构建（更快）
        self.conn.register('_lookup_src', df)
        key_cols_str = ', '.join(key_columns)
        
        result = self.conn.execute(f"""
            SELECT {key_cols_str}, {value_column}
            FROM _lookup_src
        """).fetchdf()
        
        self.conn.unregister('_lookup_src')
        
        # 构建字典
        if len(key_columns) == 1:
            lookup_dict = dict(zip(result[key_columns[0]], result[value_column]))
        else:
            keys = [tuple(row) for row in result[key_columns].values]
            lookup_dict = dict(zip(keys, result[value_column]))
        
        self._index_cache[cache_key] = lookup_dict
        return lookup_dict
    
    # ==================== 向量化计算 ====================
    
    def vectorized_filter(
        self,
        df: pd.DataFrame,
        conditions: Dict[str, Any],
        table_name: str = "_filter_temp"
    ) -> pd.DataFrame:
        """
        向量化过滤操作
        
        比Pandas的布尔索引更快，特别是对于多条件过滤
        """
        self.conn.register(table_name, df)
        
        where_clauses = []
        for col, val in conditions.items():
            if isinstance(val, (list, tuple)):
                val_str = ', '.join([f"'{v}'" if isinstance(v, str) else str(v) for v in val])
                where_clauses.append(f"{col} IN ({val_str})")
            elif isinstance(val, str):
                where_clauses.append(f"{col} = '{val}'")
            else:
                where_clauses.append(f"{col} = {val}")
        
        where_str = ' AND '.join(where_clauses)
        
        result = self.conn.execute(f"""
            SELECT * FROM {table_name}
            WHERE {where_str}
        """).fetchdf()
        
        self.conn.unregister(table_name)
        return result
    
    def vectorized_net_demand(
        self,
        gross_demand: pd.DataFrame,
        beginning_inventory: pd.DataFrame,
        intransit: pd.DataFrame,
        open_deployment: pd.DataFrame,
        future_production: pd.DataFrame,
        safety_stock: pd.DataFrame,
        target_date: str
    ) -> pd.DataFrame:
        """
        向量化净需求计算
        
        替代模块中逐行计算的Python循环
        """
        t0 = time.perf_counter()
        
        # 注册所有表
        self.conn.register('gross_demand', gross_demand)
        self.conn.register('beginning_inventory', beginning_inventory)
        self.conn.register('intransit', intransit)
        self.conn.register('open_deployment', open_deployment)
        self.conn.register('future_production', future_production)
        self.conn.register('safety_stock', safety_stock)
        
        result = self.conn.execute(f"""
            WITH supply AS (
                -- 期初库存
                SELECT material, location, COALESCE(SUM(qty), 0) as bi_qty
                FROM beginning_inventory
                GROUP BY material, location
            ),
            transit AS (
                -- 在途库存
                SELECT material, receiving as location, COALESCE(SUM(qty), 0) as it_qty
                FROM intransit
                WHERE arrival_date <= '{target_date}'
                GROUP BY material, receiving
            ),
            deployment AS (
                -- 开放部署入站
                SELECT material, receiving as location, COALESCE(SUM(deployed_qty), 0) as od_qty
                FROM open_deployment
                WHERE planned_arrival_date <= '{target_date}'
                GROUP BY material, receiving
            ),
            production AS (
                -- 未来生产
                SELECT material, location, COALESCE(SUM(planned_qty), 0) as fp_qty
                FROM future_production
                WHERE production_date <= '{target_date}'
                GROUP BY material, location
            ),
            demand AS (
                -- 毛需求
                SELECT material, location, COALESCE(SUM(quantity), 0) as gross_qty
                FROM gross_demand
                WHERE date = '{target_date}'
                GROUP BY material, location
            ),
            ss AS (
                -- 安全库存
                SELECT material, location, COALESCE(MAX(safety_stock_qty), 0) as ss_qty
                FROM safety_stock
                WHERE date = '{target_date}'
                GROUP BY material, location
            ),
            combined AS (
                SELECT 
                    COALESCE(d.material, s.material, t.material, od.material, p.material, ss.material) as material,
                    COALESCE(d.location, s.location, t.location, od.location, p.location, ss.location) as location,
                    COALESCE(d.gross_qty, 0) as gross_demand,
                    COALESCE(s.bi_qty, 0) as beginning_inventory,
                    COALESCE(t.it_qty, 0) as intransit,
                    COALESCE(od.od_qty, 0) as open_deployment_inbound,
                    COALESCE(p.fp_qty, 0) as future_production,
                    COALESCE(ss.ss_qty, 0) as safety_stock
                FROM demand d
                FULL OUTER JOIN supply s ON d.material = s.material AND d.location = s.location
                FULL OUTER JOIN transit t ON COALESCE(d.material, s.material) = t.material 
                    AND COALESCE(d.location, s.location) = t.location
                FULL OUTER JOIN deployment od ON COALESCE(d.material, s.material, t.material) = od.material 
                    AND COALESCE(d.location, s.location, t.location) = od.location
                FULL OUTER JOIN production p ON COALESCE(d.material, s.material, t.material, od.material) = p.material 
                    AND COALESCE(d.location, s.location, t.location, od.location) = p.location
                FULL OUTER JOIN ss ON COALESCE(d.material, s.material, t.material, od.material, p.material) = ss.material 
                    AND COALESCE(d.location, s.location, t.location, od.location, p.location) = ss.location
            )
            SELECT 
                material,
                location,
                gross_demand,
                beginning_inventory,
                intransit,
                open_deployment_inbound,
                future_production,
                safety_stock,
                beginning_inventory + intransit + open_deployment_inbound + future_production as total_supply,
                GREATEST(0, gross_demand + safety_stock - 
                    (beginning_inventory + intransit + open_deployment_inbound + future_production)
                ) as net_demand
            FROM combined
            WHERE material IS NOT NULL
            ORDER BY material, location
        """).fetchdf()
        
        # 清理
        for table in ['gross_demand', 'beginning_inventory', 'intransit', 
                      'open_deployment', 'future_production', 'safety_stock']:
            self.conn.unregister(table)
        
        elapsed = time.perf_counter() - t0
        self._stats['queries_executed'] += 1
        self._stats['total_query_time'] += elapsed
        self._stats['rows_processed'] += len(result)
        
        return result
    
    def vectorized_moq_rv(
        self,
        quantities: pd.DataFrame,
        moq_rv_config: pd.DataFrame
    ) -> pd.DataFrame:
        """
        向量化MOQ/RV应用
        
        替代循环中的逐行计算
        """
        self.conn.register('quantities', quantities)
        self.conn.register('config', moq_rv_config)
        
        result = self.conn.execute("""
            SELECT 
                q.*,
                COALESCE(c.moq, 0) as moq,
                COALESCE(c.rv, 1) as rv,
                CASE 
                    WHEN q.quantity <= 0 THEN 0
                    WHEN q.quantity < COALESCE(c.moq, 0) THEN 
                        CASE WHEN q.quantity > 0 THEN COALESCE(c.moq, 0) ELSE 0 END
                    WHEN COALESCE(c.rv, 1) > 0 THEN 
                        CEIL(q.quantity::FLOAT / COALESCE(c.rv, 1)) * COALESCE(c.rv, 1)
                    ELSE q.quantity
                END as adjusted_quantity
            FROM quantities q
            LEFT JOIN config c 
                ON q.material = c.material 
                AND (q.sending = c.sending OR q.location = c.sending)
        """).fetchdf()
        
        self.conn.unregister('quantities')
        self.conn.unregister('config')
        
        return result
    
    def vectorized_priority_allocation(
        self,
        demands: pd.DataFrame,
        available_inventory: pd.DataFrame,
        priority_config: pd.DataFrame
    ) -> pd.DataFrame:
        """
        向量化优先级分配
        
        使用窗口函数替代Python循环
        """
        self.conn.register('demands', demands)
        self.conn.register('inventory', available_inventory)
        self.conn.register('priority', priority_config)
        
        result = self.conn.execute("""
            WITH ranked_demands AS (
                SELECT 
                    d.*,
                    COALESCE(p.priority, 999) as priority_rank,
                    ROW_NUMBER() OVER (
                        PARTITION BY d.material, d.sending
                        ORDER BY COALESCE(p.priority, 999), d.requirement_date
                    ) as alloc_order
                FROM demands d
                LEFT JOIN priority p ON d.demand_element = p.demand_element
            ),
            inventory_available AS (
                SELECT 
                    material,
                    location as sending,
                    SUM(qty) as available_qty
                FROM inventory
                GROUP BY material, location
            ),
            cumulative_demand AS (
                SELECT 
                    rd.*,
                    ia.available_qty,
                    SUM(rd.demand_qty) OVER (
                        PARTITION BY rd.material, rd.sending
                        ORDER BY rd.alloc_order
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) as cumulative_demand
                FROM ranked_demands rd
                LEFT JOIN inventory_available ia 
                    ON rd.material = ia.material AND rd.sending = ia.sending
            )
            SELECT 
                material,
                sending,
                receiving,
                demand_element,
                requirement_date,
                demand_qty,
                priority_rank,
                available_qty,
                cumulative_demand,
                GREATEST(0, LEAST(
                    demand_qty,
                    available_qty - (cumulative_demand - demand_qty)
                )) as allocated_qty,
                demand_qty - GREATEST(0, LEAST(
                    demand_qty,
                    available_qty - (cumulative_demand - demand_qty)
                )) as unmet_qty
            FROM cumulative_demand
            ORDER BY material, sending, alloc_order
        """).fetchdf()
        
        self.conn.unregister('demands')
        self.conn.unregister('inventory')
        self.conn.unregister('priority')
        
        return result
    
    def vectorized_order_consumption(
        self,
        orders: pd.DataFrame,
        forecast: pd.DataFrame,
        consume_window_days: int = 7
    ) -> pd.DataFrame:
        """
        向量化订单消耗计算
        
        替代demand_planning模块中的循环消耗逻辑
        """
        self.conn.register('orders', orders)
        self.conn.register('forecast', forecast)
        
        result = self.conn.execute(f"""
            WITH order_window AS (
                SELECT 
                    o.*,
                    o.date as order_date,
                    o.date + INTERVAL '{consume_window_days} days' as window_end
                FROM orders o
            ),
            forecast_in_window AS (
                SELECT 
                    ow.material,
                    ow.location,
                    ow.order_date,
                    ow.quantity as order_qty,
                    f.date as forecast_date,
                    f.quantity as forecast_qty,
                    ROW_NUMBER() OVER (
                        PARTITION BY ow.material, ow.location, ow.order_date
                        ORDER BY f.date
                    ) as day_offset
                FROM order_window ow
                JOIN forecast f 
                    ON ow.material = f.material 
                    AND ow.location = f.location
                    AND f.date >= ow.order_date 
                    AND f.date <= ow.window_end
            ),
            consumption AS (
                SELECT 
                    material,
                    location,
                    forecast_date,
                    forecast_qty,
                    SUM(LEAST(order_qty, forecast_qty)) OVER (
                        PARTITION BY material, location, forecast_date
                    ) as consumed
                FROM forecast_in_window
            )
            SELECT 
                f.material,
                f.location,
                f.date,
                f.quantity as original_qty,
                COALESCE(c.consumed, 0) as consumed_qty,
                GREATEST(0, f.quantity - COALESCE(c.consumed, 0)) as remaining_qty
            FROM forecast f
            LEFT JOIN consumption c 
                ON f.material = c.material 
                AND f.location = c.location 
                AND f.date = c.forecast_date
            ORDER BY f.material, f.location, f.date
        """).fetchdf()
        
        self.conn.unregister('orders')
        self.conn.unregister('forecast')
        
        return result
    
    # ==================== 缓存管理 ====================
    
    def cache_to_parquet(self, df: pd.DataFrame, cache_name: str):
        """将DataFrame缓存到Parquet文件"""
        if self.cache_dir is None:
            return
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{cache_name}.parquet"
        
        self.conn.register('_cache_df', df)
        self.conn.execute(f"""
            COPY _cache_df TO '{cache_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        self.conn.unregister('_cache_df')
    
    def load_from_parquet(self, cache_name: str) -> Optional[pd.DataFrame]:
        """从Parquet缓存加载"""
        if self.cache_dir is None:
            return None
        
        cache_path = self.cache_dir / f"{cache_name}.parquet"
        if not cache_path.exists():
            return None
        
        return self.conn.execute(f"""
            SELECT * FROM read_parquet('{cache_path}')
        """).fetchdf()
    
    # ==================== PostgreSQL直接操作 ====================
    
    def query_postgres_direct(self, sql: str) -> pd.DataFrame:
        """
        直接查询PostgreSQL（通过DuckDB）
        
        这比psycopg + pandas更快，因为DuckDB可以直接解析PostgreSQL的行格式
        """
        if not self._pg_attached:
            raise RuntimeError("PostgreSQL未附加，请先调用attach_postgres()")
        
        t0 = time.perf_counter()
        result = self.conn.execute(sql).fetchdf()
        elapsed = time.perf_counter() - t0
        
        self._stats['queries_executed'] += 1
        self._stats['total_query_time'] += elapsed
        self._stats['rows_processed'] += len(result)
        
        return result
    
    def bulk_insert_to_postgres(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: str = "public",
        if_exists: str = "append"
    ):
        """
        批量插入到PostgreSQL
        
        使用DuckDB的批量写入能力，比逐行INSERT快得多
        """
        if not self._pg_attached:
            raise RuntimeError("PostgreSQL未附加，请先调用attach_postgres()")
        
        self.conn.register('_insert_df', df)
        
        if if_exists == "replace":
            self.conn.execute(f"DROP TABLE IF EXISTS pg.{schema}.{table_name}")
            self.conn.execute(f"""
                CREATE TABLE pg.{schema}.{table_name} AS 
                SELECT * FROM _insert_df
            """)
        else:
            self.conn.execute(f"""
                INSERT INTO pg.{schema}.{table_name}
                SELECT * FROM _insert_df
            """)
        
        self.conn.unregister('_insert_df')
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self._stats.copy()
        if stats['queries_executed'] > 0:
            stats['avg_query_time'] = stats['total_query_time'] / stats['queries_executed']
        else:
            stats['avg_query_time'] = 0
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
        )
        return stats
    
    def print_stats(self):
        """打印性能统计"""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("📊 OptimizedDataProcessor 性能统计")
        print("=" * 50)
        print(f"查询次数: {stats['queries_executed']}")
        print(f"处理行数: {stats['rows_processed']:,}")
        print(f"总查询时间: {stats['total_query_time']:.2f}s")
        print(f"平均查询时间: {stats['avg_query_time']*1000:.2f}ms")
        print(f"缓存命中率: {stats['cache_hit_rate']*100:.1f}%")
        print("=" * 50)
