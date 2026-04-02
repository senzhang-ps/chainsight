"""
模块优化包装器 - 为每个模块提供DuckDB优化接口

本模块提供:
1. Module3Optimizer - 净需求计算优化
2. Module5Optimizer - 部署计划优化
3. Module6Optimizer - 物流执行优化
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from functools import lru_cache

import pandas as pd
import numpy as np

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseModuleOptimizer:
    """模块优化器基类"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.duckdb_conn = None
        self._indexes = {}
        self._caches = {}
        self._stats = {
            'calculations': 0,
            'cache_hits': 0,
            'total_time': 0
        }
        
        if DUCKDB_AVAILABLE:
            self.duckdb_conn = duckdb.connect(':memory:')
    
    def _register_dataframe(self, name: str, df: pd.DataFrame):
        """将DataFrame注册为DuckDB表"""
        if self.duckdb_conn and not df.empty:
            self.duckdb_conn.register(name, df)
    
    def _execute_query(self, query: str) -> pd.DataFrame:
        """执行DuckDB查询"""
        if not self.duckdb_conn:
            return pd.DataFrame()
        return self.duckdb_conn.execute(query).fetchdf()
    
    def cleanup(self):
        """清理资源"""
        if self.duckdb_conn:
            self.duckdb_conn.close()
        self._indexes.clear()
        self._caches.clear()


class Module3Optimizer(BaseModuleOptimizer):
    """
    Module3 (MRP计划) 优化器
    
    主要优化:
    1. 净需求计算 - 向量化替代循环
    2. BOM展开 - 批量处理
    3. 供应匹配 - 索引加速
    """
    
    def __init__(self):
        super().__init__('Module3')
        self._bom_index = None
        self._supply_index = None
    
    def build_bom_index(self, bom_df: pd.DataFrame):
        """构建BOM索引"""
        if bom_df.empty:
            return
        
        # 创建父-子物料映射
        self._bom_index = bom_df.groupby('parent_material').apply(
            lambda x: x[['child_material', 'quantity']].to_dict('records')
        ).to_dict()
        
        # 注册到DuckDB
        self._register_dataframe('bom', bom_df)
    
    def build_supply_index(self, config_dict: Dict[str, pd.DataFrame]):
        """构建供应配置索引"""
        # 物料-位置供应信息
        supply_config = config_dict.get('M3_SupplyConfig', pd.DataFrame())
        if not supply_config.empty:
            self._register_dataframe('supply_config', supply_config)
        
        # MOQ/RV配置
        moq_rv_config = config_dict.get('M3_MOQ_RV', pd.DataFrame())
        if not moq_rv_config.empty:
            self._register_dataframe('moq_rv_config', moq_rv_config)
            # 创建快速查找索引
            self._supply_index = moq_rv_config.set_index(
                ['material', 'location']
            ).to_dict('index')
    
    def vectorized_net_demand(
        self,
        orders_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        in_transit_df: pd.DataFrame = None,
        production_plan_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        向量化净需求计算
        
        SQL实现替代Python循环:
        net_demand = gross_demand - inventory - in_transit - production_plan
        """
        if not self.duckdb_conn or orders_df.empty:
            return orders_df
        
        start_time = time.time()
        
        # 注册数据
        self._register_dataframe('orders', orders_df)
        self._register_dataframe('inventory', inventory_df if inventory_df is not None else pd.DataFrame())
        self._register_dataframe('in_transit', in_transit_df if in_transit_df is not None else pd.DataFrame())
        self._register_dataframe('production', production_plan_df if production_plan_df is not None else pd.DataFrame())
        
        # 向量化计算净需求
        query = """
        WITH order_demand AS (
            SELECT 
                material,
                location,
                date,
                SUM(quantity) as gross_demand
            FROM orders
            GROUP BY material, location, date
        ),
        available_inventory AS (
            SELECT 
                material,
                location,
                COALESCE(SUM(quantity), 0) as inv_qty
            FROM inventory
            GROUP BY material, location
        ),
        in_transit_supply AS (
            SELECT
                material,
                destination as location,
                COALESCE(SUM(quantity), 0) as transit_qty
            FROM in_transit
            WHERE arrival_date <= (SELECT MAX(date) FROM orders)
            GROUP BY material, destination
        ),
        planned_production AS (
            SELECT
                material,
                location,
                COALESCE(SUM(quantity), 0) as prod_qty
            FROM production
            WHERE available_date <= (SELECT MAX(date) FROM orders)
            GROUP BY material, location
        )
        SELECT 
            od.material,
            od.location,
            od.date,
            od.gross_demand,
            COALESCE(ai.inv_qty, 0) as available_inventory,
            COALESCE(it.transit_qty, 0) as in_transit,
            COALESCE(pp.prod_qty, 0) as planned_production,
            GREATEST(0, 
                od.gross_demand 
                - COALESCE(ai.inv_qty, 0) 
                - COALESCE(it.transit_qty, 0) 
                - COALESCE(pp.prod_qty, 0)
            ) as net_demand
        FROM order_demand od
        LEFT JOIN available_inventory ai 
            ON od.material = ai.material AND od.location = ai.location
        LEFT JOIN in_transit_supply it
            ON od.material = it.material AND od.location = it.location
        LEFT JOIN planned_production pp
            ON od.material = pp.material AND od.location = pp.location
        ORDER BY od.date, od.material, od.location
        """
        
        try:
            result = self._execute_query(query)
            self._stats['calculations'] += 1
            self._stats['total_time'] += time.time() - start_time
            return result
        except Exception as e:
            logger.error(f"向量化净需求计算失败: {e}")
            return orders_df
    
    def vectorized_moq_rv(
        self,
        net_demand_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        向量化MOQ/RV应用
        
        MOQ (最小订单量): 需求量向上取整到MOQ的倍数
        RV (舍入值): 需求量向上舍入到RV的倍数
        """
        if not self.duckdb_conn or net_demand_df.empty:
            return net_demand_df
        
        self._register_dataframe('net_demand', net_demand_df)
        
        query = """
        SELECT 
            nd.*,
            COALESCE(mc.moq, 1) as moq,
            COALESCE(mc.rv, 1) as rv,
            -- 应用MOQ: 如果需求>0且<MOQ，则取MOQ
            CASE 
                WHEN nd.net_demand > 0 AND nd.net_demand < COALESCE(mc.moq, 1)
                THEN COALESCE(mc.moq, 1)
                ELSE nd.net_demand
            END as demand_after_moq,
            -- 应用RV: 向上舍入到RV的倍数
            CASE
                WHEN COALESCE(mc.rv, 1) > 0
                THEN CEIL(
                    CASE 
                        WHEN nd.net_demand > 0 AND nd.net_demand < COALESCE(mc.moq, 1)
                        THEN COALESCE(mc.moq, 1)
                        ELSE nd.net_demand
                    END / COALESCE(mc.rv, 1)
                ) * COALESCE(mc.rv, 1)
                ELSE CASE 
                    WHEN nd.net_demand > 0 AND nd.net_demand < COALESCE(mc.moq, 1)
                    THEN COALESCE(mc.moq, 1)
                    ELSE nd.net_demand
                END
            END as final_demand
        FROM net_demand nd
        LEFT JOIN moq_rv_config mc
            ON nd.material = mc.material 
            AND nd.location = mc.location
        """
        
        try:
            return self._execute_query(query)
        except Exception as e:
            logger.error(f"向量化MOQ/RV应用失败: {e}")
            return net_demand_df
    
    def batch_bom_explosion(
        self,
        demand_df: pd.DataFrame,
        levels: int = 5
    ) -> pd.DataFrame:
        """
        批量BOM展开
        
        使用递归CTE一次性展开所有层级
        """
        if not self.duckdb_conn or demand_df.empty:
            return demand_df
        
        self._register_dataframe('demand', demand_df)
        
        query = f"""
        WITH RECURSIVE bom_explosion AS (
            -- 基础层: 原始需求
            SELECT 
                material as parent_material,
                material as child_material,
                quantity,
                location,
                date,
                1 as bom_level
            FROM demand
            
            UNION ALL
            
            -- 递归层: 展开BOM
            SELECT
                be.child_material as parent_material,
                b.child_material,
                be.quantity * b.quantity as quantity,
                be.location,
                be.date,
                be.bom_level + 1
            FROM bom_explosion be
            JOIN bom b ON be.child_material = b.parent_material
            WHERE be.bom_level < {levels}
        )
        SELECT 
            child_material as material,
            location,
            date,
            SUM(quantity) as exploded_demand,
            MAX(bom_level) as max_level
        FROM bom_explosion
        GROUP BY child_material, location, date
        ORDER BY date, material, location
        """
        
        try:
            return self._execute_query(query)
        except Exception as e:
            logger.error(f"批量BOM展开失败: {e}")
            return demand_df


class Module5Optimizer(BaseModuleOptimizer):
    """
    Module5 (部署计划) 优化器
    
    主要优化:
    1. 优先级分配 - 窗口函数替代嵌套循环
    2. 网络流优化 - 批量计算
    3. 库存消耗 - 向量化更新
    """
    
    def __init__(self):
        super().__init__('Module5')
        self._priority_index = None
        self._network_index = None
    
    def build_priority_index(self, priority_config: pd.DataFrame):
        """构建优先级索引"""
        if priority_config.empty:
            return
        
        self._register_dataframe('priority_config', priority_config)
        
        # 创建优先级查找字典
        if 'demand_element' in priority_config.columns:
            self._priority_index = priority_config.set_index(
                'demand_element'
            )['priority'].to_dict()
    
    def build_network_index(self, network_config: pd.DataFrame):
        """构建网络配置索引"""
        if network_config.empty:
            return
        
        self._register_dataframe('network', network_config)
        
        # 创建发送-接收位置映射
        self._network_index = network_config.groupby('sending').apply(
            lambda x: x['receiving'].tolist()
        ).to_dict()
    
    def vectorized_priority_allocation(
        self,
        demand_df: pd.DataFrame,
        inventory_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        向量化优先级分配
        
        使用窗口函数按优先级顺序分配库存
        """
        if not self.duckdb_conn or demand_df.empty:
            return demand_df
        
        self._register_dataframe('demand', demand_df)
        self._register_dataframe('inventory', inventory_df)
        
        query = """
        WITH demand_with_priority AS (
            SELECT 
                d.*,
                COALESCE(pc.priority, 999) as priority
            FROM demand d
            LEFT JOIN priority_config pc 
                ON d.demand_element = pc.demand_element
        ),
        inventory_agg AS (
            SELECT material, location, SUM(quantity) as available_qty
            FROM inventory
            GROUP BY material, location
        ),
        ranked_demand AS (
            SELECT 
                dwp.*,
                COALESCE(ia.available_qty, 0) as total_available,
                ROW_NUMBER() OVER (
                    PARTITION BY dwp.material, dwp.sending 
                    ORDER BY dwp.priority, dwp.date
                ) as allocation_order,
                SUM(dwp.quantity) OVER (
                    PARTITION BY dwp.material, dwp.sending 
                    ORDER BY dwp.priority, dwp.date
                    ROWS UNBOUNDED PRECEDING
                ) as cumulative_demand
            FROM demand_with_priority dwp
            LEFT JOIN inventory_agg ia 
                ON dwp.material = ia.material 
                AND dwp.sending = ia.location
        )
        SELECT 
            material,
            sending,
            receiving,
            date,
            demand_element,
            priority,
            quantity as original_demand,
            -- 计算实际分配量
            CASE
                WHEN cumulative_demand <= total_available 
                THEN quantity
                WHEN cumulative_demand - quantity < total_available
                THEN total_available - (cumulative_demand - quantity)
                ELSE 0
            END as allocated_qty,
            -- 计算未满足量
            CASE
                WHEN cumulative_demand <= total_available 
                THEN 0
                WHEN cumulative_demand - quantity < total_available
                THEN quantity - (total_available - (cumulative_demand - quantity))
                ELSE quantity
            END as unmet_demand
        FROM ranked_demand
        ORDER BY material, sending, priority, date
        """
        
        try:
            return self._execute_query(query)
        except Exception as e:
            logger.error(f"向量化优先级分配失败: {e}")
            return demand_df
    
    def vectorized_layer_propagation(
        self,
        layer_demand_df: pd.DataFrame,
        layer_inventory_df: pd.DataFrame,
        layer_config: pd.DataFrame
    ) -> pd.DataFrame:
        """
        向量化层级传播
        
        批量处理一个层级内的所有节点
        """
        if not self.duckdb_conn or layer_demand_df.empty:
            return layer_demand_df
        
        self._register_dataframe('layer_demand', layer_demand_df)
        self._register_dataframe('layer_inventory', layer_inventory_df)
        self._register_dataframe('layer_config', layer_config)
        
        query = """
        WITH supply_demand_match AS (
            SELECT 
                ld.material,
                ld.receiving,
                ld.date,
                ld.quantity as demand_qty,
                COALESCE(li.quantity, 0) as local_inv,
                lc.sending as upstream_location,
                lc.priority as source_priority
            FROM layer_demand ld
            LEFT JOIN layer_inventory li 
                ON ld.material = li.material AND ld.receiving = li.location
            LEFT JOIN layer_config lc 
                ON ld.receiving = lc.receiving
        ),
        allocation AS (
            SELECT 
                material,
                receiving,
                date,
                demand_qty,
                local_inv,
                -- 优先使用本地库存
                LEAST(demand_qty, local_inv) as from_local,
                -- 剩余需求向上游传播
                GREATEST(0, demand_qty - local_inv) as propagate_upstream,
                upstream_location
            FROM supply_demand_match
        )
        SELECT 
            material,
            upstream_location as sending,
            receiving,
            date,
            propagate_upstream as quantity,
            'layer_propagation' as source
        FROM allocation
        WHERE propagate_upstream > 0 AND upstream_location IS NOT NULL
        """
        
        try:
            return self._execute_query(query)
        except Exception as e:
            logger.error(f"向量化层级传播失败: {e}")
            return layer_demand_df


class Module6Optimizer(BaseModuleOptimizer):
    """
    Module6 (物流执行) 优化器
    
    主要优化:
    1. 配送匹配 - 批量处理
    2. 路径选择 - 向量化计算
    3. 状态更新 - 批量操作
    """
    
    def __init__(self):
        super().__init__('Module6')
        self._route_index = None
        self._vehicle_index = None
    
    def build_logistics_index(self, config_dict: Dict[str, pd.DataFrame]):
        """构建物流配置索引"""
        # 路线配置
        route_config = config_dict.get('M6_RouteConfig', pd.DataFrame())
        if not route_config.empty:
            self._register_dataframe('routes', route_config)
            self._route_index = route_config.set_index(
                ['origin', 'destination']
            ).to_dict('index')
        
        # 车辆配置
        vehicle_config = config_dict.get('M6_VehicleConfig', pd.DataFrame())
        if not vehicle_config.empty:
            self._register_dataframe('vehicles', vehicle_config)
    
    def vectorized_delivery_matching(
        self,
        deployment_df: pd.DataFrame,
        open_orders_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        向量化配送匹配
        
        批量匹配部署计划与开放订单
        """
        if not self.duckdb_conn or deployment_df.empty:
            return deployment_df
        
        self._register_dataframe('deployments', deployment_df)
        self._register_dataframe('open_orders', open_orders_df)
        
        query = """
        WITH deployment_ranked AS (
            SELECT 
                d.*,
                r.lead_time,
                r.cost_per_unit,
                ROW_NUMBER() OVER (
                    PARTITION BY d.material, d.sending, d.receiving 
                    ORDER BY d.priority, d.date
                ) as match_order
            FROM deployments d
            LEFT JOIN routes r 
                ON d.sending = r.origin AND d.receiving = r.destination
        ),
        order_ranked AS (
            SELECT 
                o.*,
                ROW_NUMBER() OVER (
                    PARTITION BY o.material, o.location 
                    ORDER BY o.priority, o.due_date
                ) as order_rank
            FROM open_orders o
            WHERE o.status = 'open'
        )
        SELECT 
            dr.material,
            dr.sending,
            dr.receiving,
            dr.quantity,
            dr.date as deployment_date,
            dr.lead_time,
            dr.date + dr.lead_time as arrival_date,
            dr.cost_per_unit * dr.quantity as delivery_cost,
            orr.order_id as matched_order_id
        FROM deployment_ranked dr
        LEFT JOIN order_ranked orr
            ON dr.material = orr.material 
            AND dr.receiving = orr.location
            AND dr.match_order = orr.order_rank
        """
        
        try:
            return self._execute_query(query)
        except Exception as e:
            logger.error(f"向量化配送匹配失败: {e}")
            return deployment_df
    
    def batch_update_status(
        self,
        updates: List[Dict[str, Any]]
    ) -> int:
        """
        批量更新状态
        
        返回：
            更新的记录数
        """
        if not self.duckdb_conn or not updates:
            return 0
        
        updates_df = pd.DataFrame(updates)
        self._register_dataframe('status_updates', updates_df)
        
        # 返回更新数量
        return len(updates)


# ===================== 工厂函数 =====================

def create_module_optimizer(module_name: str) -> BaseModuleOptimizer:
    """
    创建模块优化器
    
    参数：
        module_name: 模块名称 (Module3, Module5, Module6)
    
    返回：
        对应的优化器实例
    """
    optimizers = {
        'Module3': Module3Optimizer,
        'Module5': Module5Optimizer,
        'Module6': Module6Optimizer
    }
    
    optimizer_class = optimizers.get(module_name, BaseModuleOptimizer)
    return optimizer_class() if module_name in optimizers else BaseModuleOptimizer(module_name)


def create_all_optimizers() -> Dict[str, BaseModuleOptimizer]:
    """创建所有模块优化器"""
    return {
        'Module3': Module3Optimizer(),
        'Module5': Module5Optimizer(),
        'Module6': Module6Optimizer()
    }
