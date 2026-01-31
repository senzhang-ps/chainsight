"""
Module3 节点处理器模块。

负责单个节点的净需求计算和缺口传递。
"""

from typing import Dict, Optional, Tuple

import pandas as pd

from .constants import (
    DEMAND_ELEMENT_AO,
    DEMAND_ELEMENT_FORECAST,
    DEMAND_ELEMENT_SAFETY,
)
from .data_indexer import DataIndexer
from .lead_time import (
    compute_root_horizon,
    determine_lead_time,
    infer_sending_location_type,
)
from .net_demand import calculate_daily_net_demand, calculate_daily_net_demand_indexed
from .utils import (
    apply_moq_rv,
    apportion_largest_remainder,
    lookup_moq_rv_three_keys,
)


class NodeProcessor:
    """节点处理器类。"""

    def __init__(
        self,
        ctx: dict,
        downstream_gaps: dict,
        current_layer: int,
        sim_date: pd.Timestamp,
        **data_dfs
    ):
        """
        初始化节点处理器。

        Args:
            ctx: 模拟上下文
            downstream_gaps: 下游缺口字典
            current_layer: 当前层级
            sim_date: 模拟日期
            **data_dfs: 数据DataFrame字典
        """
        self.ctx = ctx
        self.downstream_gaps = downstream_gaps
        self.current_layer = current_layer
        self.sim_date = sim_date
        self.data = data_dfs

    def process(self, ml_row) -> Tuple[list, Optional[tuple], dict]:
        """
        处理单个节点。

        Args:
            ml_row: material-location行数据

        Returns:
            Tuple: (记录列表, 父节点键, 父节点缺口)
        """
        material = str(ml_row.material)
        location = str(ml_row.location)

        upstream, horizon = self._get_upstream_and_horizon(material, location)
        lower_gaps = self._get_downstream_gaps(material, location)
        
        # 使用索引版本的净需求计算（如果有索引器）
        data_indexer = self.ctx.get('data_indexer')
        if data_indexer is not None:
            ao_gap, fc_gap, ss_gap = calculate_daily_net_demand_indexed(
                material, location, self.sim_date,
                data_indexer,
                self.data['daily_supply_demand_df'],
                self.data['safety_stock_df'],
                self.data['open_deployment_df'],
                lower_gaps['FC'], lower_gaps['SS'], horizon,
                order_df=self.data.get('daily_order_df'),
                delivery_shipment_df=self.data.get('delivery_shipment_df'),
                downstream_ao_gap=lower_gaps['AO']
            )
        else:
            ao_gap, fc_gap, ss_gap = calculate_daily_net_demand(
                material, location, self.sim_date,
                self.data['daily_supply_demand_df'],
                self.data['safety_stock_df'],
                self.data['beginning_inventory_df'],
                self.data['in_transit_df'],
                self.data['delivery_gr_df'],
                self.data['future_production_df'],
                self.data['daily_shipment_df'],
                self.data['open_deployment_df'],
                lower_gaps['FC'], lower_gaps['SS'], horizon,
                delivery_shipment_df=self.data.get('delivery_shipment_df'),
                order_df=self.data.get('daily_order_df'),
                downstream_ao_gap=lower_gaps['AO']
            )

        records = self._build_records(
            material, location, horizon, ao_gap, fc_gap, ss_gap
        )
        parent_key, parent_gaps = self._compute_parent_gaps(
            material, location, upstream, ao_gap, fc_gap, ss_gap
        )

        return records, parent_key, parent_gaps

    def _get_upstream_and_horizon(
        self,
        material: str,
        location: str
    ) -> Tuple[Optional[str], int]:
        """获取上游节点和horizon。"""
        network_candidates = self.ctx['active_network'][
            (self.ctx['active_network']['material'] == material) &
            (self.ctx['active_network']['location'] == location)
        ]

        if not network_candidates.empty:
            row = network_candidates.iloc[0]
            upstream = row['sourcing']

            if pd.isna(upstream) or str(upstream).strip() == '':
                return None, self._get_root_horizon(material, location)

            location_type = infer_sending_location_type(
                self.ctx['active_network'],
                self.ctx['location_layer_map'],
                str(upstream), material, self.sim_date
            )
            horizon, _ = determine_lead_time(
                str(upstream), location, location_type,
                self.data['lead_time_df'],
                self.data.get('m4_mlcfg_df'),
                material, self.ctx['ptf_lsk_cache']
            )
            return str(upstream), max(1, horizon)

        return None, self._get_root_horizon(material, location)

    def _get_root_horizon(self, material: str, location: str) -> int:
        """获取根节点horizon。"""
        if self.ctx['location_layer_map'].get((str(material), str(location)), -1) == 0:
            return compute_root_horizon(
                material, location,
                self.data['lead_time_df'],
                self.data.get('m4_mlcfg_df'),
                self.ctx['ptf_lsk_cache']
            )
        return 1

    def _get_downstream_gaps(self, material: str, location: str) -> dict:
        """获取下游缺口。"""
        return self.downstream_gaps[(material, location)]

    def _build_records(
        self,
        material: str,
        location: str,
        horizon: int,
        ao_gap: float,
        fc_gap: float,
        ss_gap: float
    ) -> list:
        """构建记录列表。
        
        说明：仅在gap > 0时创建记录，与code_v0逻辑保持一致。
        """
        records = []
        req_date = self.sim_date + pd.Timedelta(days=1)

        # 仅在gap > 0时创建记录（与code_v0逻辑一致）
        if ao_gap > 0:
            records.append(self._make_record(
                material, location, req_date, -ao_gap,
                DEMAND_ELEMENT_AO, horizon
            ))
        if fc_gap > 0:
            records.append(self._make_record(
                material, location, req_date, -fc_gap,
                DEMAND_ELEMENT_FORECAST, horizon
            ))
        if ss_gap > 0:
            records.append(self._make_record(
                material, location, req_date, -ss_gap,
                DEMAND_ELEMENT_SAFETY, horizon
            ))

        return records

    def _make_record(
        self,
        material: str,
        location: str,
        req_date: pd.Timestamp,
        quantity: float,
        demand_element: str,
        horizon: int
    ) -> dict:
        """创建单条记录。"""
        return {
            'material': material,
            'location': location,
            'requirement_date': req_date,
            'quantity': quantity,
            'demand_element': demand_element,
            'layer': self.current_layer,
            'simulation_date': self.sim_date,
            'horizon_days': horizon,
        }

    def _compute_parent_gaps(
        self,
        material: str,
        location: str,
        upstream: Optional[str],
        ao_gap: float,
        fc_gap: float,
        ss_gap: float
    ) -> Tuple[Optional[tuple], dict]:
        """计算父节点缺口。"""
        parent_gaps = {'AO': 0.0, 'FC': 0.0, 'SS': 0.0}
        if not upstream or not pd.notna(upstream):
            return None, parent_gaps

        parent_key = (material, str(upstream))
        components = [
            ('AO', max(0.0, ao_gap)),
            ('FC', max(0.0, fc_gap)),
            ('SS', max(0.0, ss_gap)),
        ]
        total = sum(v for _, v in components)

        if total <= 0:
            return parent_key, parent_gaps

        moq, rv = lookup_moq_rv_three_keys(
            self.data.get('deploy_config_df'),
            material, upstream, location
        )
        target = apply_moq_rv(total, moq, rv, is_cross_node=True)

        if target <= 0:
            return parent_key, parent_gaps

        base_vals = [v for _, v in components]
        apportion = apportion_largest_remainder(base_vals, target)

        for (de, _), q in zip(components, apportion):
            parent_gaps[de] = float(q)

        return parent_key, parent_gaps
