"""
DuckDB数据处理模块
使用DuckDB进行高效的数据处理，然后将结果写入PostgreSQL
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import time
from contextlib import contextmanager


class DuckDBProcessor:
    """DuckDB数据处理器 - 用于高效的数据转换和分析"""
    
    def __init__(self, db_path: str = ":memory:"):
        """
        初始化DuckDB处理器
        
        Args:
            db_path: DuckDB数据库路径，默认使用内存数据库
        """
        self.db_path = db_path
        self._conn = None
        self._registered_tables = set()
    
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """获取DuckDB连接"""
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
            # 设置优化参数
            self._conn.execute("SET threads TO 4")
            self._conn.execute("SET memory_limit = '2GB'")
        return self._conn
    
    def close(self):
        """关闭连接"""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._registered_tables.clear()
    
    def register_dataframe(self, df: pd.DataFrame, table_name: str):
        """
        将DataFrame注册为DuckDB表
        
        Args:
            df: Pandas DataFrame
            table_name: 表名
        """
        self.conn.register(table_name, df)
        self._registered_tables.add(table_name)
    
    def unregister_table(self, table_name: str):
        """注销表"""
        if table_name in self._registered_tables:
            self.conn.unregister(table_name)
            self._registered_tables.discard(table_name)
    
    def execute(self, sql: str, params: tuple = None) -> duckdb.DuckDBPyRelation:
        """执行SQL查询"""
        if params:
            return self.conn.execute(sql, params)
        return self.conn.execute(sql)
    
    def query(self, sql: str) -> pd.DataFrame:
        """执行查询并返回DataFrame"""
        return self.conn.execute(sql).fetchdf()
    
    def query_to_arrow(self, sql: str):
        """执行查询并返回Arrow表（用于高效传输到PostgreSQL）"""
        return self.conn.execute(sql).fetch_arrow_table()
    
    # ==================== 数据转换操作 ====================
    
    def transform_demand_forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换需求预测数据 - 周度到日度拆分
        
        Args:
            df: 原始需求预测DataFrame (week, material, location, quantity)
        
        Returns:
            日度需求预测DataFrame
        """
        self.register_dataframe(df, 'demand_forecast')
        
        result = self.query("""
            WITH date_range AS (
                SELECT 
                    week,
                    material,
                    location,
                    quantity,
                    -- 将周转换为日期范围（假设周从周一开始）
                    CAST(week AS DATE) as week_start,
                    CAST(week AS DATE) + INTERVAL 6 DAY as week_end
                FROM demand_forecast
            ),
            daily_split AS (
                SELECT 
                    material,
                    location,
                    generate_series::DATE as date,
                    quantity / 7.0 as daily_quantity
                FROM date_range,
                LATERAL generate_series(week_start, week_end, INTERVAL '1 day')
            )
            SELECT 
                date,
                material,
                location,
                daily_quantity as quantity
            FROM daily_split
            ORDER BY material, location, date
        """)
        
        self.unregister_table('demand_forecast')
        return result
    
    def calculate_net_demand(
        self,
        gross_demand: pd.DataFrame,
        inventory: pd.DataFrame,
        safety_stock: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算净需求
        
        Args:
            gross_demand: 毛需求 (material, location, date, quantity)
            inventory: 库存 (material, location, quantity)
            safety_stock: 安全库存 (material, location, date, safety_stock_qty)
        
        Returns:
            净需求DataFrame
        """
        self.register_dataframe(gross_demand, 'gross_demand')
        self.register_dataframe(inventory, 'inventory')
        self.register_dataframe(safety_stock, 'safety_stock')
        
        result = self.query("""
            SELECT 
                gd.material,
                gd.location,
                gd.date,
                gd.quantity as gross_demand,
                COALESCE(inv.quantity, 0) as available_inventory,
                COALESCE(ss.safety_stock_qty, 0) as safety_stock,
                GREATEST(0, 
                    gd.quantity - COALESCE(inv.quantity, 0) + COALESCE(ss.safety_stock_qty, 0)
                ) as net_demand
            FROM gross_demand gd
            LEFT JOIN inventory inv 
                ON gd.material = inv.material AND gd.location = inv.location
            LEFT JOIN safety_stock ss 
                ON gd.material = ss.material 
                AND gd.location = ss.location 
                AND gd.date = ss.date
            ORDER BY gd.material, gd.location, gd.date
        """)
        
        self.unregister_table('gross_demand')
        self.unregister_table('inventory')
        self.unregister_table('safety_stock')
        return result
    
    def aggregate_orders(self, orders: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
        """
        聚合订单数据
        
        Args:
            orders: 订单DataFrame
            group_by: 分组字段列表
        
        Returns:
            聚合后的DataFrame
        """
        self.register_dataframe(orders, 'orders')
        
        group_cols = ', '.join(group_by)
        result = self.query(f"""
            SELECT 
                {group_cols},
                COUNT(*) as order_count,
                SUM(quantity) as total_quantity,
                AVG(quantity) as avg_quantity,
                MIN(quantity) as min_quantity,
                MAX(quantity) as max_quantity
            FROM orders
            GROUP BY {group_cols}
            ORDER BY {group_cols}
        """)
        
        self.unregister_table('orders')
        return result
    
    def join_config_tables(
        self,
        network: pd.DataFrame,
        leadtime: pd.DataFrame,
        deploy_config: pd.DataFrame
    ) -> pd.DataFrame:
        """
        关联配置表
        
        Args:
            network: 网络配置
            leadtime: 前置时间配置
            deploy_config: 部署配置
        
        Returns:
            关联后的配置DataFrame
        """
        self.register_dataframe(network, 'network')
        self.register_dataframe(leadtime, 'leadtime')
        self.register_dataframe(deploy_config, 'deploy_config')
        
        result = self.query("""
            SELECT 
                n.material,
                n.location,
                n.sourcing,
                n.location_type,
                lt.PDT as production_time,
                lt.GR as gr_time,
                lt.MCT as mct,
                dc.moq,
                dc.rv,
                dc.lsk
            FROM network n
            LEFT JOIN leadtime lt 
                ON n.location = lt.sending OR n.location = lt.receiving
            LEFT JOIN deploy_config dc 
                ON n.material = dc.material AND n.location = dc.sending
        """)
        
        self.unregister_table('network')
        self.unregister_table('leadtime')
        self.unregister_table('deploy_config')
        return result
    
    def calculate_deployment_allocation(
        self,
        demand: pd.DataFrame,
        supply: pd.DataFrame,
        priority_config: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算部署分配（Module5核心逻辑）
        
        Args:
            demand: 需求数据
            supply: 供应数据
            priority_config: 优先级配置
        
        Returns:
            分配结果DataFrame
        """
        self.register_dataframe(demand, 'demand')
        self.register_dataframe(supply, 'supply')
        self.register_dataframe(priority_config, 'priority')
        
        result = self.query("""
            WITH ranked_demand AS (
                SELECT 
                    d.*,
                    COALESCE(p.priority, 999) as priority_rank,
                    ROW_NUMBER() OVER (
                        PARTITION BY d.material, d.source_location 
                        ORDER BY COALESCE(p.priority, 999), d.date
                    ) as demand_rank
                FROM demand d
                LEFT JOIN priority p ON d.demand_element = p.demand_element
            ),
            supply_available AS (
                SELECT 
                    material,
                    location as source_location,
                    SUM(quantity) as available_qty
                FROM supply
                GROUP BY material, location
            ),
            allocation AS (
                SELECT 
                    rd.material,
                    rd.source_location,
                    rd.destination_location,
                    rd.date,
                    rd.quantity as requested_qty,
                    rd.priority_rank,
                    sa.available_qty,
                    LEAST(rd.quantity, 
                        GREATEST(0, sa.available_qty - 
                            SUM(rd.quantity) OVER (
                                PARTITION BY rd.material, rd.source_location 
                                ORDER BY rd.demand_rank 
                                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                            )
                        )
                    ) as allocated_qty
                FROM ranked_demand rd
                LEFT JOIN supply_available sa 
                    ON rd.material = sa.material 
                    AND rd.source_location = sa.source_location
            )
            SELECT 
                material,
                source_location,
                destination_location,
                date,
                requested_qty,
                COALESCE(allocated_qty, 0) as allocated_qty,
                requested_qty - COALESCE(allocated_qty, 0) as unfulfilled_qty,
                priority_rank
            FROM allocation
            ORDER BY material, source_location, priority_rank, date
        """)
        
        self.unregister_table('demand')
        self.unregister_table('supply')
        self.unregister_table('priority')
        return result
    
    def calculate_inventory_changes(
        self,
        initial_inventory: pd.DataFrame,
        shipments: pd.DataFrame,
        receipts: pd.DataFrame,
        date: str
    ) -> pd.DataFrame:
        """
        计算库存变化
        
        Args:
            initial_inventory: 期初库存
            shipments: 发货记录
            receipts: 收货记录
            date: 日期
        
        Returns:
            库存变化记录
        """
        self.register_dataframe(initial_inventory, 'initial_inv')
        self.register_dataframe(shipments, 'shipments')
        self.register_dataframe(receipts, 'receipts')
        
        result = self.query(f"""
            WITH shipment_agg AS (
                SELECT 
                    material,
                    source_location as location,
                    SUM(quantity) as shipped_qty
                FROM shipments
                GROUP BY material, source_location
            ),
            receipt_agg AS (
                SELECT 
                    material,
                    destination_location as location,
                    SUM(quantity) as received_qty
                FROM receipts
                GROUP BY material, destination_location
            )
            SELECT 
                '{date}' as date,
                inv.material,
                inv.location,
                inv.quantity as opening_qty,
                COALESCE(s.shipped_qty, 0) as shipped_qty,
                COALESCE(r.received_qty, 0) as received_qty,
                inv.quantity - COALESCE(s.shipped_qty, 0) + COALESCE(r.received_qty, 0) as closing_qty
            FROM initial_inv inv
            LEFT JOIN shipment_agg s 
                ON inv.material = s.material AND inv.location = s.location
            LEFT JOIN receipt_agg r 
                ON inv.material = r.material AND inv.location = r.location
            ORDER BY inv.material, inv.location
        """)
        
        self.unregister_table('initial_inv')
        self.unregister_table('shipments')
        self.unregister_table('receipts')
        return result
    
    def apply_moq_rv(
        self,
        quantities: pd.DataFrame,
        config: pd.DataFrame
    ) -> pd.DataFrame:
        """
        应用MOQ和RV规则
        
        Args:
            quantities: 数量数据 (material, location, quantity)
            config: MOQ/RV配置 (material, location, moq, rv)
        
        Returns:
            调整后的数量
        """
        self.register_dataframe(quantities, 'quantities')
        self.register_dataframe(config, 'config')
        
        result = self.query("""
            SELECT 
                q.material,
                q.location,
                q.quantity as original_qty,
                COALESCE(c.moq, 0) as moq,
                COALESCE(c.rv, 1) as rv,
                CASE 
                    WHEN q.quantity <= 0 THEN 0
                    WHEN q.quantity < COALESCE(c.moq, 0) THEN COALESCE(c.moq, 0)
                    ELSE CEIL(q.quantity / COALESCE(c.rv, 1)) * COALESCE(c.rv, 1)
                END as adjusted_qty
            FROM quantities q
            LEFT JOIN config c 
                ON q.material = c.material 
                AND q.location = c.location
        """)
        
        self.unregister_table('quantities')
        self.unregister_table('config')
        return result
    
    # ==================== 批量数据处理 ====================
    
    def bulk_transform(
        self,
        dataframes: Dict[str, pd.DataFrame],
        sql: str
    ) -> pd.DataFrame:
        """
        批量数据转换
        
        Args:
            dataframes: 表名到DataFrame的映射
            sql: SQL查询语句
        
        Returns:
            转换结果DataFrame
        """
        # 注册所有表
        for name, df in dataframes.items():
            self.register_dataframe(df, name)
        
        # 执行查询
        result = self.query(sql)
        
        # 清理
        for name in dataframes.keys():
            self.unregister_table(name)
        
        return result
    
    def read_excel_to_duckdb(self, excel_path: str, sheet_name: str = None) -> pd.DataFrame:
        """
        直接用DuckDB读取Excel（通过Pandas中转）
        
        Args:
            excel_path: Excel文件路径
            sheet_name: Sheet名称
        
        Returns:
            DataFrame
        """
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        return df
    
    def read_csv_direct(self, csv_path: str) -> pd.DataFrame:
        """
        使用DuckDB直接读取CSV（更高效）
        
        Args:
            csv_path: CSV文件路径
        
        Returns:
            DataFrame
        """
        return self.query(f"SELECT * FROM read_csv_auto('{csv_path}')")
    
    def export_to_parquet(self, df: pd.DataFrame, output_path: str, table_name: str = 'data'):
        """
        导出DataFrame到Parquet格式
        
        Args:
            df: DataFrame
            output_path: 输出路径
            table_name: 临时表名
        """
        self.register_dataframe(df, table_name)
        self.execute(f"COPY {table_name} TO '{output_path}' (FORMAT PARQUET)")
        self.unregister_table(table_name)


class DuckDBToPostgres:
    """DuckDB到PostgreSQL的数据传输器"""
    
    def __init__(self, duckdb_processor: DuckDBProcessor, pg_connection):
        """
        初始化
        
        Args:
            duckdb_processor: DuckDB处理器
            pg_connection: PostgreSQL连接（DatabaseConnection实例）
        """
        self.duck = duckdb_processor
        self.pg = pg_connection
    
    def transfer_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = 'replace'
    ) -> int:
        """
        将DataFrame传输到PostgreSQL
        
        Args:
            df: 要传输的DataFrame
            table_name: 目标表名
            if_exists: 如果表存在的处理方式
        
        Returns:
            写入的行数
        """
        if df.empty:
            return 0
        
        # 使用PostgreSQL连接写入
        self.pg.create_table_from_df(df, table_name, if_exists)
        return len(df)
    
    def process_and_transfer(
        self,
        source_dfs: Dict[str, pd.DataFrame],
        sql: str,
        target_table: str,
        if_exists: str = 'replace'
    ) -> int:
        """
        处理数据并传输到PostgreSQL
        
        Args:
            source_dfs: 源数据表字典
            sql: 处理SQL
            target_table: 目标表名
            if_exists: 如果表存在的处理方式
        
        Returns:
            写入的行数
        """
        # DuckDB处理
        result_df = self.duck.bulk_transform(source_dfs, sql)
        
        # 传输到PostgreSQL
        return self.transfer_dataframe(result_df, target_table, if_exists)


def test_duckdb_processor():
    """测试DuckDB处理器"""
    print("=" * 60)
    print("DuckDB处理器测试")
    print("=" * 60)
    
    processor = DuckDBProcessor()
    
    # 测试数据
    df = pd.DataFrame({
        'material': ['M001', 'M001', 'M002', 'M002'],
        'location': ['L1', 'L2', 'L1', 'L2'],
        'quantity': [100, 200, 150, 250]
    })
    
    print("\n原始数据:")
    print(df)
    
    # 测试聚合
    result = processor.aggregate_orders(df, ['material'])
    print("\n按material聚合:")
    print(result)
    
    # 测试SQL查询
    processor.register_dataframe(df, 'test_table')
    result2 = processor.query("""
        SELECT 
            material,
            SUM(quantity) as total,
            COUNT(*) as count
        FROM test_table
        GROUP BY material
    """)
    print("\nSQL查询结果:")
    print(result2)
    
    processor.close()
    print("\n✅ 测试完成")


if __name__ == "__main__":
    test_duckdb_processor()
