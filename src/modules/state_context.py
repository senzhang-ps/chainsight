"""仿真状态总管模块。

职责：
- 持有所有可变状态
- 提供 view（读取状态 → DataFrame）
- 提供 processor（模块结果 → 状态变更）
- 每日操作（快照/清理/GR入库）

不管持久化（交给 Orchestrator.save_daily_state）。
不管调度（交给 Orchestrator）。

生命周期：
1. initialize()       → 仿真开始前初始化
2. day_start(date)    → 每日开始（快照/清理/GR/views）
3. apply_xxx()        → 模块结果写回状态
4. day_end(date)      → 每日结束（期末快照）
5. [Orch 负责持久化]
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .module import Module
from ..core.orchestrator.models import DeploymentUID
from ..utils.normalization import (
    normalize_identifiers,
    normalize_location,
    normalize_material,
    normalize_receiving,
    normalize_sending,
)

logger = logging.getLogger("SupplyChainSimulation")


class StateContext(Module):
    """仿真状态总管 — 持有所有可变状态，提供读写接口。

    不管持久化（交给 Orchestrator）。
    不管调度（交给 Orchestrator）。
    只管：状态持有 + 状态读取(view) + 状态写入(processor) + 每日操作。

    生命周期：
    1. initialize()       → 仿真开始前初始化
    2. day_start(date)    → 每日开始（快照/清理/GR/views）
    3. apply_xxx()        → 模块结果写回状态
    4. day_end(date)      → 每日结束（期末快照）
    5. [Orch 负责持久化]
    """

    module_config = 'StateContext'

    def __init__(self, simulation_date, orchestrator=None, orch=None,
                 data_source='live', db=None, run_id=None, **kwargs):
        super().__init__(simulation_date, orch, 'StateContext', **kwargs)

        # ── 全部可变状态（原 Orchestrator.__init__ 的内容） ──
        self.unrestricted_inventory: Dict[Tuple[str, str], int] = {}
        self.open_deployment: Dict[str, Dict] = {}
        self.in_transit: Dict[str, Dict] = {}
        self.production_gr: List[Dict] = []
        self.delivery_gr: List[Dict] = []
        self.shipment_log: List[Dict] = []
        self.delivery_shipment_log: List[Dict] = []
        self.production_plan_backlog: List[Dict] = []
        self.space_capacity: pd.DataFrame = pd.DataFrame()

        # M4 跨天状态（按日期字符串索引，与 RuntimeState 数据结构对齐）
        self.m4_line_states: Dict[str, dict] = {}           # {date_str: {line: state_dict}}
        self.m4_allocated_capacity: Dict[str, dict] = {}    # {date_str: {key: hours}}

        # 按日期索引（O(1) 查询）
        self.production_gr_by_date: Dict[str, List[Dict]] = {}
        self.delivery_gr_by_date: Dict[str, List[Dict]] = {}
        self.shipment_log_by_date: Dict[str, List[Dict]] = {}
        self.delivery_shipment_log_by_date: Dict[str, List[Dict]] = {}

        # 快照
        self.daily_beginning_inventory: Dict[str, Dict[Tuple[str, str], int]] = {}
        self.daily_ending_inventory: Dict[str, Dict[Tuple[str, str], int]] = {}
        self.initial_inventory: Dict[Tuple[str, str], int] = {}

        # 控制
        self.uid_sequence = 0
        self.cleanup_grace_days: int = 100
        self.daily_logs: List[Dict] = []
        self._last_cleanup_date = None
        self.current_date = None

        # 兼容旧模块的属性（如 ModuleOne 写入 shipment_valid）
        self.shipment_valid = 0

        # 数据源
        self._data_source = data_source
        self._db = db
        self._run_id = run_id

        # views dict — 每日 day_start 后计算
        self.views: Dict[str, pd.DataFrame] = {}

    # ══════════════════════════════════════════
    # 初始化（仿真开始前调用一次）
    # ══════════════════════════════════════════

    def initialize(self, config_dict: dict):
        """从配置字典初始化状态。

        来源: Orchestrator.initialize_inventory + set_space_capacity

        续跑：orch 已注入快照（_resuming）时，从 ViewContext views 恢复 ctx 状态，
        跳过从配置重算（库存/在途/调拨等从 viewcontext_* 表读取还原）。
        """
        orch = self.orchestrator
        if getattr(orch, '_resuming', False) and getattr(orch, '_restore_date', None):
            # 从上一完成周期（progress_date - 1）的 viewcontext 恢复 ctx；
            # 断点当天 progress_date 的 view 尚未落库，故不能用 _resume_date。
            orch.persistence.restore_state_from_views(self, orch.run_id, orch._restore_date)
            logger.info("🔁 StateContext 已从 ViewContext 恢复，跳过 initialize 重算")
            return

        if 'M1_InitialInventory' in config_dict and not config_dict['M1_InitialInventory'].empty:
            self._init_inventory(config_dict['M1_InitialInventory'])
        else:
            logger.warning("⚠️ 未找到初始库存配置，使用空库存")
            self._init_inventory(
                pd.DataFrame(columns=['material', 'location', 'quantity'])
            )

        if 'Global_SpaceCapacity' in config_dict and not config_dict['Global_SpaceCapacity'].empty:
            self._init_space_capacity(config_dict['Global_SpaceCapacity'])

    def _init_inventory(self, df: pd.DataFrame):
        """从 M1_InitialInventory 配置初始化实物库存。

        源: orchestrator_main.py:132-157 Orchestrator.initialize_inventory()
        """
        self.unrestricted_inventory.clear()
        self.initial_inventory.clear()

        normalized_df = normalize_identifiers(df)

        for row in normalized_df.itertuples():
            key = (row.material, row.location)
            quantity = int(row.quantity)
            self.unrestricted_inventory[key] = quantity
            self.initial_inventory[key] = quantity

        msg = f"Initialized {len(normalized_df)} inventory records"
        self._log_event("INIT_INVENTORY", msg)

    def _init_space_capacity(self, df: pd.DataFrame):
        """从 Global_SpaceCapacity 配置设置空间容量。

        源: orchestrator_main.py:159-194 Orchestrator.set_space_capacity()
        """
        self.space_capacity = normalize_identifiers(df.copy())
        self.space_capacity["eff_from"] = pd.to_datetime(
            self.space_capacity["eff_from"].astype(str),
            format="%Y-%m-%d", errors="coerce",
        )
        self.space_capacity["eff_to"] = pd.to_datetime(
            self.space_capacity["eff_to"].astype(str),
            format="%Y-%m-%d", errors="coerce",
        )

        msg = f"Configured {len(df)} space capacity records"
        self._log_event("SET_SPACE_CAPACITY", msg)

    # ══════════════════════════════════════════
    # 每日开始（任何模块运行之前）
    # ══════════════════════════════════════════

    def day_start(self, date_str: str):
        """快照 → 清理 → 到货入库 → 历史生产入库 → 计算 views。

        来源: simulation_db.py:358-406 循环体内日始段
        """
        logger.info("🌅 每日开始: %s", date_str)
        self.current_date = pd.to_datetime(date_str).normalize()

        # step1: 期初快照
        # 源: persistence.py:192 OrchestratorPersistenceMixin.save_beginning_inventory()
        self.daily_beginning_inventory[date_str] = (
            self.unrestricted_inventory.copy()
        )

        # step2: 清理过期调拨
        # 源: daily_ops.py:220 OrchestratorDailyOpsMixin.cleanup_past_due_open_deployments()
        self._cleanup_past_due(date_str)

        # step3: 处理当日到货
        # 源: daily_ops.py:111 OrchestratorDailyOpsMixin._process_delivery_arrivals()
        self._process_delivery_arrivals(date_str)

        # step4: 历史生产当日入库
        # 源: simulation_db.py:368-398 循环体内联逻辑
        self._process_backlog_gr(date_str)

        # step5: 计算 views
        # 源: views.py OrchestratorViewsMixin 全部 get_xxx_view()
        self.views = self._compute_views(date_str)

    def _cleanup_past_due(self, date_str: str):
        """清理过期的 open deployment。

        源: daily_ops.py:220-330 OrchestratorDailyOpsMixin.cleanup_past_due_open_deployments()
        """
        cleanup_date = pd.to_datetime(date_str).normalize()
        threshold_date = cleanup_date - pd.Timedelta(
            days=int(self.cleanup_grace_days)
        )

        removed = []
        to_delete = []

        for uid, rec in self.open_deployment.items():
            pdd = pd.to_datetime(
                rec.get('planned_deployment_date')
            ).normalize()
            remaining_qty = int(rec.get('deployed_qty', 0))

            if pdd < threshold_date:
                to_delete.append(uid)
                removed.append({
                    'cleanup_date': cleanup_date,
                    'grace_days': int(self.cleanup_grace_days),
                    'ori_deployment_uid': uid,
                    'material': normalize_material(rec.get('material')),
                    'sending': normalize_sending(rec.get('sending')),
                    'receiving': normalize_receiving(rec.get('receiving')),
                    'planned_deployment_date': pdd,
                    'remaining_qty': remaining_qty,
                    'demand_element': rec.get('demand_element', ''),
                    'creation_date': rec.get('creation_date', ''),
                    'reason': f"past_due>{int(self.cleanup_grace_days)}d",
                })

        for uid in to_delete:
            del self.open_deployment[uid]

        # 存储清理审计结果，供 Orch 持久化到 DB
        cleanup_df = pd.DataFrame(removed)
        if cleanup_df.empty:
            cleanup_df = pd.DataFrame(columns=[
                'cleanup_date', 'grace_days', 'ori_deployment_uid',
                'material', 'sending', 'receiving',
                'planned_deployment_date', 'remaining_qty',
                'demand_element', 'creation_date', 'reason',
            ])
        self.cleanup_audit_df = normalize_identifiers(cleanup_df)

        msg = (
            f"Removed {len(to_delete)} past-due open deployments "
            f"(grace_days={self.cleanup_grace_days})"
        )
        self._log_event("OPEN_DEPLOYMENT_CLEANUP", msg)

    def _process_delivery_arrivals(self, date_str: str):
        """处理当天到达的在途交付。

        源: daily_ops.py:111-218 OrchestratorDailyOpsMixin._process_delivery_arrivals()
        """
        date_obj = pd.to_datetime(date_str).normalize()

        completed_transits = []
        for transit_uid, transit_record in self.in_transit.items():
            actual_dd = pd.to_datetime(
                transit_record['actual_delivery_date']
            ).normalize()
            if actual_dd == date_obj:
                # 增加收货地库存
                receiving_key = (
                    transit_record['material'],
                    transit_record['receiving'],
                )
                old_inv = self.unrestricted_inventory.get(receiving_key, 0)
                self.unrestricted_inventory[receiving_key] = (
                    old_inv + transit_record['quantity']
                )

                # 记录 delivery GR
                gr_record = {
                    'date': date_obj,
                    'material': normalize_material(
                        transit_record['material']
                    ),
                    'receiving': normalize_receiving(
                        transit_record['receiving']
                    ),
                    'quantity': transit_record['quantity'],
                    'ori_deployment_uid': transit_record.get(
                        'ori_deployment_uid'
                    ),
                    'vehicle_uid': transit_record.get('vehicle_uid'),
                    'actual_ship_date': transit_record.get(
                        'actual_ship_date'
                    ),
                }

                # 去重检查
                existing_key = (
                    date_obj,
                    transit_record['material'],
                    transit_record['receiving'],
                    transit_record.get('ori_deployment_uid'),
                    transit_record.get('vehicle_uid'),
                )
                ds = date_obj.strftime('%Y-%m-%d')
                is_dup = any(
                    (
                        rec['date'],
                        rec['material'],
                        rec['receiving'],
                        rec['ori_deployment_uid'],
                        rec['vehicle_uid'],
                    ) == existing_key
                    for rec in self.delivery_gr_by_date.get(ds, [])
                )

                if not is_dup:
                    self.delivery_gr.append(gr_record)
                    if ds not in self.delivery_gr_by_date:
                        self.delivery_gr_by_date[ds] = []
                    self.delivery_gr_by_date[ds].append(gr_record)

                completed_transits.append(transit_uid)

        for transit_uid in completed_transits:
            del self.in_transit[transit_uid]

        if completed_transits:
            msg = (
                f"Processed {len(completed_transits)} delivery arrivals"
            )
            self._log_event("DELIVERY_ARRIVALS", msg)

    def _process_backlog_gr(self, date_str: str):
        """从 production_plan_backlog 中处理当日到货的生产计划。

        源: simulation_db.py:368-398 循环体内联逻辑。

        StateContext 维护 backlog 在内存中，无需文件系统访问。
        只处理 available_date == today 的记录，处理后从 backlog 中移除。
        """
        date_obj = pd.to_datetime(date_str).normalize()

        if not self.production_plan_backlog:
            return

        backlog_df = pd.DataFrame(self.production_plan_backlog)
        backlog_df['available_date'] = pd.to_datetime(
            backlog_df['available_date']
        ).dt.normalize()

        mask = backlog_df['available_date'] == date_obj
        arriving = backlog_df[mask]

        if arriving.empty:
            return

        for row in arriving.itertuples():
            key = (
                normalize_material(row.material),
                normalize_location(row.location),
            )
            quantity = int(row.quantity)

            old = self.unrestricted_inventory.get(key, 0)
            self.unrestricted_inventory[key] = old + quantity

            record = {
                'date': date_obj,
                'material': normalize_material(row.material),
                'location': normalize_location(row.location),
                'quantity': quantity,
            }
            self.production_gr.append(record)
            ds = date_obj.strftime('%Y-%m-%d')
            if ds not in self.production_gr_by_date:
                self.production_gr_by_date[ds] = []
            self.production_gr_by_date[ds].append(record)

        # 移除已处理的记录
        remaining = backlog_df[~mask]
        self.production_plan_backlog = (
            remaining.to_dict('records') if not remaining.empty else []
        )

        msg = f"Processed {len(arriving)} backlog production arrivals"
        self._log_event("BACKLOG_GR", msg)

    def _compute_views(self, date_str: str) -> Dict[str, pd.DataFrame]:
        """计算所有 view 并返回 dict。

        源: views.py OrchestratorViewsMixin 的全部 get_xxx_view()
        """
        return {
            'beginning_inventory': self._beginning_inventory_view(date_str),
            'unrestricted_inventory': self._unrestricted_inventory_view(date_str),
            'production_gr': self._production_gr_view(date_str),
            'delivery_gr': self._delivery_gr_view(date_str),
            'planning_intransit': self._planning_intransit_view(date_str),
            'open_deployment': self._open_deployment_view(date_str),
            'all_production': self._all_production_view(date_str),
            'shipment_log': self._shipment_log_view(date_str),
            'delivery_shipment_log': self._delivery_shipment_log_view(date_str),
            'space_quota': self._space_quota_view(date_str),
            'production_plan_backlog': self._production_plan_backlog_view(date_str),
        }

    # ── View 方法（从 OrchestratorViewsMixin 端口） ──

    def _beginning_inventory_view(self, date: str) -> pd.DataFrame:
        """源: views.py:450-483 get_beginning_inventory_view()"""
        date_obj = pd.to_datetime(date).normalize()

        if date in self.daily_beginning_inventory:
            inv = self.daily_beginning_inventory[date]
        else:
            inv = self.unrestricted_inventory

        records = []
        for (mat, loc), qty in inv.items():
            records.append({
                'date': date_obj,
                'material': normalize_material(mat),
                'location': normalize_location(loc),
                'quantity': qty,
            })

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'date', 'material', 'location', 'quantity',
            ])
        return df

    def _unrestricted_inventory_view(self, date: str) -> pd.DataFrame:
        """源: views.py:24-56 get_unrestricted_inventory_view()"""
        date_obj = pd.to_datetime(date).normalize()

        records = []
        for (mat, loc), qty in self.unrestricted_inventory.items():
            records.append({
                'date': date_obj,
                'material': normalize_material(mat),
                'location': normalize_location(loc),
                'quantity': qty,
            })

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'date', 'material', 'location', 'quantity',
            ])
        return df

    def _production_gr_view(self, date: str) -> pd.DataFrame:
        """源: views.py:338-361 get_production_gr_view()"""
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        records = self.production_gr_by_date.get(date_str, [])
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'date', 'material', 'location', 'quantity',
            ])
        return df

    def _delivery_gr_view(self, date: str) -> pd.DataFrame:
        """源: views.py:363-388 get_delivery_gr_view()"""
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        records = self.delivery_gr_by_date.get(date_str, [])
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'date', 'material', 'receiving', 'quantity',
                'ori_deployment_uid', 'vehicle_uid',
            ])
        return df

    def _planning_intransit_view(self, date: str) -> pd.DataFrame:
        """源: views.py:79-128 get_planning_intransit_view()"""
        date_obj = pd.to_datetime(date).normalize()
        records = []
        for uid, rec in self.in_transit.items():
            records.append({
                'transit_uid': uid,
                'date': date_obj,
                'material': normalize_material(rec['material']),
                'sending': rec.get('sending', ''),
                'receiving': rec['receiving'],
                'actual_ship_date': rec.get('actual_ship_date', ''),
                'actual_delivery_date': rec['actual_delivery_date'],
                'quantity': rec['quantity'],
                'ori_deployment_uid': rec.get('ori_deployment_uid', ''),
                'vehicle_uid': rec.get('vehicle_uid', ''),
            })

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'transit_uid', 'date', 'material', 'sending',
                'receiving', 'actual_ship_date',
                'actual_delivery_date', 'quantity',
                'ori_deployment_uid', 'vehicle_uid',
            ])
        return df

    def _open_deployment_view(self, date: str) -> pd.DataFrame:
        """源: views.py:147-191 get_open_deployment_view()"""
        records = []
        for uid, rec in self.open_deployment.items():
            records.append({
                'material': normalize_material(rec['material']),
                'sending': normalize_sending(rec['sending']),
                'receiving': normalize_receiving(rec['receiving']),
                'planned_deployment_date': pd.to_datetime(
                    rec['planned_deployment_date']
                ),
                'deployed_qty': rec['deployed_qty'],
                'demand_element': rec['demand_element'],
                'ori_deployment_uid': uid,
            })
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'material', 'sending', 'receiving',
                'planned_deployment_date', 'deployed_qty',
                'demand_element', 'ori_deployment_uid',
            ])
        return df

    def _all_production_view(self, date: str) -> pd.DataFrame:
        """源: views.py:280-336 get_all_production_view()"""
        cols = ['material', 'location', 'available_date', 'quantity']
        date_obj = pd.to_datetime(date).normalize()

        today_gr = self._production_gr_view(date)
        if not today_gr.empty:
            today_gr = today_gr.rename(
                columns={'date': 'available_date'}
            )[cols]
        else:
            today_gr = pd.DataFrame(columns=cols)

        future = pd.DataFrame(self.production_plan_backlog)
        if not future.empty:
            future['available_date'] = pd.to_datetime(
                future['available_date']
            ).dt.normalize()
            future = future[future['available_date'] >= date_obj][cols]
        else:
            future = pd.DataFrame(columns=cols)

        dfs = [df for df in [today_gr, future] if not df.empty]
        if not dfs:
            return pd.DataFrame(columns=cols)

        out = pd.concat(dfs, ignore_index=True)
        if out.empty:
            return out

        out = out.groupby(
            ['material', 'location', 'available_date'],
            as_index=False,
        ).agg({'quantity': 'sum'})
        out['quantity'] = out['quantity'].astype(int)
        return out

    def _shipment_log_view(self, date: str) -> pd.DataFrame:
        """源: views.py:392-415 get_shipment_log_view()"""
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        records = self.shipment_log_by_date.get(date_str, [])
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'date', 'material', 'location', 'quantity',
            ])
        return df

    def _delivery_shipment_log_view(self, date: str) -> pd.DataFrame:
        """源: views.py:417-446 get_delivery_shipment_log_view()"""
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        rows = self.delivery_shipment_log_by_date.get(date_str, [])
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=[
                'date', 'material', 'sending', 'receiving',
                'quantity', 'ori_deployment_uid',
                'actual_ship_date', 'actual_delivery_date', 'type',
            ])
        return df

    def _space_quota_view(self, date: str) -> pd.DataFrame:
        """源: views.py:195-245 get_space_quota_view()"""
        date_obj = pd.to_datetime(date).normalize()

        if (self.space_capacity.empty
                or 'eff_from' not in self.space_capacity.columns):
            return pd.DataFrame(columns=['receiving', 'date', 'max_qty'])

        mask = (
            (self.space_capacity['eff_from'] <= date_obj)
            & (self.space_capacity['eff_to'] >= date_obj)
        )
        effective = self.space_capacity[mask]

        records = []
        for row in effective.itertuples():
            loc = row.location
            cap = row.capacity
            loc_inv = sum(
                qty for (_, l), qty in self.unrestricted_inventory.items()
                if l == loc
            )
            records.append({
                'receiving': loc,
                'date': date_obj,
                'max_qty': max(0, cap - loc_inv),
            })

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=['receiving', 'date', 'max_qty'])
        return df

    def _production_plan_backlog_view(self, date: str) -> pd.DataFrame:
        """源: views.py:249-278 get_production_plan_backlog_view()"""
        cols = ['material', 'location', 'available_date', 'quantity']
        if not self.production_plan_backlog:
            return pd.DataFrame(columns=cols)

        backlog_df = pd.DataFrame(self.production_plan_backlog)
        if backlog_df.empty:
            return pd.DataFrame(columns=cols)

        for col in cols:
            if col not in backlog_df.columns:
                backlog_df[col] = ''

        return backlog_df[cols]

    # ══════════════════════════════════════════
    # Processor（模块运行后写回状态）
    # ══════════════════════════════════════════

    def apply_shipments(self, shipment_df: pd.DataFrame, date: str):
        """处理 Module1 发货数据。

        源: processors.py:23-84 OrchestratorProcessorsMixin.process_module1_shipments()
        """
        if shipment_df is None or shipment_df.empty:
            return

        date_obj = pd.to_datetime(date).normalize()

        # 筛选当日发货记录
        mask = (
            pd.to_datetime(shipment_df['date']).dt.normalize() == date_obj
        )
        daily_shipments = shipment_df[mask]

        for row in daily_shipments.itertuples():
            key = (
                normalize_material(row.material),
                normalize_location(row.location),
            )
            if key in self.unrestricted_inventory:
                old_qty = self.unrestricted_inventory[key]
                new_qty = max(0, old_qty - int(row.quantity))
                self.unrestricted_inventory[key] = new_qty

            # 记录发货日志
            record = {
                'date': date_obj,
                'material': normalize_material(row.material),
                'location': normalize_location(row.location),
                'quantity': int(row.quantity),
                'type': 'customer_shipment',
            }
            self.shipment_log.append(record)
            ds = date_obj.strftime('%Y-%m-%d')
            if ds not in self.shipment_log_by_date:
                self.shipment_log_by_date[ds] = []
            self.shipment_log_by_date[ds].append(record)

        if len(daily_shipments) > 0:
            logger.info("🚚 已扣减 %d 个 shipment 的库存", len(daily_shipments))
            self._log_event("M1_SHIPMENTS", f"Processed {len(daily_shipments)} shipments")

    def apply_production(self, production_df: pd.DataFrame, date: str):
        """处理 Module4 生产数据。

        源: processors.py:86-246 OrchestratorProcessorsMixin.process_module4_production()
        """
        if production_df is None or production_df.empty:
            return

        date_obj = pd.to_datetime(date).normalize()

        # === A) 缓存当日 GR 的生产计划到 backlog 中 ===
        tmp = production_df.copy()
        if 'available_date' in tmp.columns:
            tmp['available_date'] = pd.to_datetime(
                tmp['available_date']
            ).dt.normalize()
        if 'quantity' not in tmp.columns and 'produced_qty' in tmp.columns:
            tmp = tmp.rename(columns={'produced_qty': 'quantity'})

        keep = ['material', 'location', 'available_date', 'quantity']
        tmp = tmp[keep].copy()
        tmp['material'] = tmp['material'].astype(str)
        tmp['location'] = tmp['location'].apply(normalize_location)
        tmp['quantity'] = tmp['quantity'].fillna(0).astype(int)

        # 用于精确去重的维度
        tmp['simulation_date'] = date_obj
        if 'production_plan_date' in production_df.columns:
            tmp['production_plan_date'] = pd.to_datetime(
                production_df['production_plan_date']
            ).dt.normalize()
        else:
            tmp['production_plan_date'] = tmp['available_date']

        # 追加到 backlog
        if self.production_plan_backlog:
            existing_df = pd.DataFrame(self.production_plan_backlog)
        else:
            existing_df = pd.DataFrame()

        for col in ['simulation_date', 'production_plan_date']:
            if col not in existing_df.columns:
                existing_df[col] = pd.NaT

        combined = pd.concat([existing_df, tmp], ignore_index=True)

        # 第一阶段：按 5 维去重
        combined = combined.drop_duplicates(
            subset=[
                'material', 'location', 'simulation_date',
                'production_plan_date', 'available_date',
            ],
            keep='first',
        )

        # 第二阶段：按 3 维汇总数量
        aggregated = combined.groupby(
            ['material', 'location', 'available_date'],
            as_index=False,
        ).agg({'quantity': 'sum'})
        aggregated['quantity'] = (
            aggregated['quantity'].fillna(0).astype(int)
        )
        self.production_plan_backlog = aggregated.to_dict('records')

        # === B) 只对"今天到货"的进行 GR 入库 ===
        mask = (
            pd.to_datetime(production_df['available_date']).dt.normalize()
            == date_obj
        )
        daily_production = production_df[mask]

        for row in daily_production.itertuples():
            key = (
                normalize_material(row.material),
                normalize_location(row.location),
            )
            quantity = int(row.produced_qty)

            old = self.unrestricted_inventory.get(key, 0)
            self.unrestricted_inventory[key] = old + quantity

            record = {
                'date': date_obj,
                'material': normalize_material(row.material),
                'location': normalize_location(row.location),
                'quantity': quantity,
            }
            self.production_gr.append(record)
            ds = date_obj.strftime('%Y-%m-%d')
            if ds not in self.production_gr_by_date:
                self.production_gr_by_date[ds] = []
            self.production_gr_by_date[ds].append(record)

        if len(daily_production) > 0:
            msg = (
                f"Processed {len(daily_production)} production receipts"
            )
            self._log_event("M4_PRODUCTION", msg)

    def apply_deployment(self, deployment_df: pd.DataFrame, date: str):
        """处理 Module5 部署计划。

        源: processors.py:248-330 OrchestratorProcessorsMixin.process_module5_deployment()
        """
        if deployment_df is None or deployment_df.empty:
            return

        date_obj = pd.to_datetime(date).normalize()

        # 为保证可复现，稳定排序
        sort_cols = [
            col for col in [
                'material', 'sending', 'receiving',
                'planned_deployment_date', 'demand_element', 'deployed_qty',
            ]
            if col in deployment_df.columns
        ]
        if sort_cols:
            deployment_df = deployment_df.sort_values(
                by=sort_cols, kind='mergesort'
            )

        for row in deployment_df.itertuples():
            self.uid_sequence += 1
            pdd = pd.to_datetime(
                row.planned_deployment_date
            ).strftime('%Y-%m-%d')
            uid_obj = DeploymentUID(
                material=str(row.material),
                sending=str(row.sending),
                receiving=str(row.receiving),
                planned_deploy_date=pdd,
                demand_element=str(row.demand_element),
                sequence=self.uid_sequence,
            )
            uid = uid_obj.to_string()

            converted_qty = self._safe_convert_to_int(row.deployed_qty)

            self.open_deployment[uid] = {
                'material': normalize_material(row.material),
                'sending': normalize_sending(row.sending),
                'receiving': normalize_receiving(row.receiving),
                'planned_deployment_date': pdd,
                'deployed_qty': converted_qty,
                'demand_element': str(row.demand_element),
                'creation_date': date_obj.strftime('%Y-%m-%d'),
            }

        if len(deployment_df) > 0:
            msg = f"Added {len(deployment_df)} deployment plans"
            self._log_event("M5_DEPLOYMENT", msg)

    def apply_delivery(self, delivery_df: pd.DataFrame, date: str):
        """处理 Module6 交付计划。

        源: processors.py:332-527 OrchestratorProcessorsMixin.process_module6_delivery()
        """
        if delivery_df is None or delivery_df.empty:
            return

        date_obj = pd.to_datetime(date).normalize()

        for row in delivery_df.itertuples():
            uid = str(row.ori_deployment_uid)
            vehicle_uid = str(row.vehicle_uid)
            material = str(row.material)
            sending = str(row.sending)
            receiving = str(row.receiving)

            norm_mat = normalize_material(material)
            norm_rcv = normalize_receiving(receiving)
            ship_date = pd.to_datetime(row.actual_ship_date)
            delivery_date = pd.to_datetime(row.actual_delivery_date)
            quantity = self._safe_convert_to_int(row.delivery_qty)

            # 只处理当天发运
            if ship_date.normalize() != date_obj:
                continue

            # 减少开放调拨数量
            if uid in self.open_deployment:
                old_qty = self.open_deployment[uid]['deployed_qty']
                self.open_deployment[uid]['deployed_qty'] = old_qty - quantity
                if self.open_deployment[uid]['deployed_qty'] <= 0:
                    del self.open_deployment[uid]

            # 减少发货地非限制库存
            sending_key = (
                normalize_material(material),
                normalize_location(sending),
            )
            if sending_key in self.unrestricted_inventory:
                old_inv = self.unrestricted_inventory[sending_key]
                self.unrestricted_inventory[sending_key] = max(
                    0, old_inv - quantity
                )

            # 记录发运出库日志
            shipment_record = {
                'date': date_obj,
                'material': norm_mat,
                'sending': normalize_sending(sending),
                'receiving': norm_rcv,
                'quantity': quantity,
                'ori_deployment_uid': uid,
                'actual_ship_date': ship_date.strftime('%Y-%m-%d'),
                'actual_delivery_date': delivery_date.strftime('%Y-%m-%d'),
                'type': 'delivery_shipment',
            }
            self.delivery_shipment_log.append(shipment_record)
            ds = date_obj.strftime('%Y-%m-%d')
            if ds not in self.delivery_shipment_log_by_date:
                self.delivery_shipment_log_by_date[ds] = []
            self.delivery_shipment_log_by_date[ds].append(shipment_record)

            # 判断处理逻辑
            if delivery_date.normalize() > date_obj:
                # 为未来交付创建在途记录
                transit_uid = f"{uid}_transit_{vehicle_uid}"
                self.in_transit[transit_uid] = {
                    'material': norm_mat,
                    'sending': normalize_sending(sending),
                    'receiving': norm_rcv,
                    'actual_ship_date': ship_date.strftime('%Y-%m-%d'),
                    'actual_delivery_date': delivery_date.strftime('%Y-%m-%d'),
                    'quantity': quantity,
                    'ori_deployment_uid': uid,
                    'vehicle_uid': vehicle_uid,
                }
            elif delivery_date.normalize() == date_obj:
                # 当天交付：创建 delivery GR
                receiving_key = (material, receiving)
                old_inv = self.unrestricted_inventory.get(
                    receiving_key, 0
                )
                self.unrestricted_inventory[receiving_key] = (
                    old_inv + quantity
                )

                # 记录 delivery GR（含去重检查）
                gr_record = {
                    'date': date_obj,
                    'material': norm_mat,
                    'receiving': norm_rcv,
                    'quantity': quantity,
                    'ori_deployment_uid': uid,
                    'vehicle_uid': vehicle_uid,
                }

                existing_key = (
                    date_obj, material, receiving, uid, vehicle_uid,
                )
                ds = date_obj.strftime('%Y-%m-%d')
                is_dup = any(
                    (
                        rec['date'],
                        rec['material'],
                        rec['receiving'],
                        rec['ori_deployment_uid'],
                        rec['vehicle_uid'],
                    ) == existing_key
                    for rec in self.delivery_gr_by_date.get(ds, [])
                )

                if not is_dup:
                    self.delivery_gr.append(gr_record)
                    if ds not in self.delivery_gr_by_date:
                        self.delivery_gr_by_date[ds] = []
                    self.delivery_gr_by_date[ds].append(gr_record)

        if len(delivery_df) > 0:
            msg = f"Processed {len(delivery_df)} delivery plans"
            self._log_event("M6_DELIVERY", msg)

    # ══════════════════════════════════════════
    # 每日结束
    # ══════════════════════════════════════════

    def day_end(self, date_str: str):
        """期末快照。

        源: persistence.py:208 OrchestratorPersistenceMixin.save_ending_inventory()
        """
        self.daily_ending_inventory[date_str] = (
            self.unrestricted_inventory.copy()
        )
        logger.info("📋 %s 当日处理完成", date_str)

    # ══════════════════════════════════════════
    # M4 跨天状态托管（产线状态 + 已分配产能）
    # ══════════════════════════════════════════

    def apply_line_state(self, line_states: dict, date_str: str):
        """存储当日 M4 产线状态（换产连续性）。

        驱动循环在 m4.run() 后调用此方法，将 current_line_states 写入 ctx。
        外部 persistence_manager 在 save_daily_state 时落盘。
        """
        if line_states:
            self.m4_line_states[date_str] = line_states

    def apply_allocated_capacity(self, allocated_capacity: dict, date_str: str):
        """存储当日 M4 已分配产能（防重复分配）。

        驱动循环在 m4.run() 后调用此方法，将 current_allocated_capacity 写入 ctx。
        """
        if allocated_capacity:
            self.m4_allocated_capacity[date_str] = allocated_capacity

    def get_previous_line_state(self, date_str: str) -> dict:
        """返回前一日 M4 产线状态（date_str - 1 day），不存在则 {}。

        驱动循环在每日 m4.prepare() 前调用此方法，将结果注入
        m4.previous_line_states_override。
        """
        prev_date = (pd.Timestamp(date_str) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        return self.m4_line_states.get(prev_date, {})

    def get_all_previous_allocated_capacity(self, date_str: str) -> dict:
        """返回 date_str 之前所有日期的已分配产能汇总。

        逻辑同 RuntimeState.load_all_previous_capacity，但数据来源是
        内存中的 ctx.m4_allocated_capacity（而非文件）。
        重跑恢复后数据来自 DB viewcontext_m4_allocated_capacity 表。

        驱动循环在每日 m4.prepare() 前调用此方法，将结果注入
        m4.allocated_capacity_override。
        """
        consolidated = {}
        target = pd.Timestamp(date_str)
        for d_str, daily_cap in self.m4_allocated_capacity.items():
            if pd.Timestamp(d_str) < target:
                for key, value in daily_cap.items():
                    if key not in consolidated:
                        consolidated[key] = 0
                    consolidated[key] += value
        return consolidated

    # ══════════════════════════════════════════
    # 向后兼容的旧接口方法
    # ══════════════════════════════════════════

    def get_unrestricted_inventory_view(self, date: str) -> pd.DataFrame:
        """向后兼容：get_unrestricted_inventory_view()"""
        return self._unrestricted_inventory_view(date)

    def get_current_unrestricted_inventory(self) -> Dict[Tuple[str, str], int]:
        """向后兼容：get_current_unrestricted_inventory()"""
        result: Dict[Tuple[str, str], int] = {}
        for (mat, loc), qty in self.unrestricted_inventory.items():
            key = (normalize_material(mat), normalize_location(loc))
            result[key] = qty
        return result

    def get_beginning_inventory_view(self, date: str) -> pd.DataFrame:
        return self._beginning_inventory_view(date)

    def get_production_gr_view(self, date: str) -> pd.DataFrame:
        return self._production_gr_view(date)

    def get_delivery_gr_view(self, date: str) -> pd.DataFrame:
        return self._delivery_gr_view(date)

    def get_planning_intransit_view(self, date: str) -> pd.DataFrame:
        return self._planning_intransit_view(date)

    def get_open_deployment_view(self, date: str) -> pd.DataFrame:
        return self._open_deployment_view(date)

    def get_open_deployment(self, current_date: pd.Timestamp) -> pd.DataFrame:
        return self._open_deployment_view(current_date.strftime('%Y-%m-%d'))

    def get_space_quota_view(self, date: str) -> pd.DataFrame:
        return self._space_quota_view(date)

    def get_production_plan_backlog_view(self, date: str) -> pd.DataFrame:
        return self._production_plan_backlog_view(date)

    def get_all_production_view(self, date: str) -> pd.DataFrame:
        return self._all_production_view(date)

    def get_shipment_log_view(self, date: str) -> pd.DataFrame:
        return self._shipment_log_view(date)

    def get_delivery_shipment_log_view(self, date: str) -> pd.DataFrame:
        return self._delivery_shipment_log_view(date)

    def get_summary_statistics(self, date: str) -> dict:
        """获取指定日期的汇总统计。

        源: views.py:487-528 get_summary_statistics()
        """
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        return {
            'date': date,
            'total_inventory_items': len(self.unrestricted_inventory),
            'total_inventory_quantity': sum(
                self.unrestricted_inventory.values()
            ),
            'open_deployment_count': len(self.open_deployment),
            'in_transit_count': len(self.in_transit),
            'production_gr_count': len(
                self.production_gr_by_date.get(date_str, [])
            ),
            'delivery_gr_count': len(
                self.delivery_gr_by_date.get(date_str, [])
            ),
            'shipment_count': len(
                self.shipment_log_by_date.get(date_str, [])
            ),
        }

    def generate_inventory_change_log(self, date: str) -> pd.DataFrame:
        """生成指定日期的库存变动日志。

        源: inventory_log.py:35-243 OrchestratorInventoryLogMixin.generate_inventory_change_log()
        """
        date_obj = pd.to_datetime(date).normalize()
        date_str = date_obj.strftime('%Y-%m-%d')

        # 获取所有涉及的物料-地点组合
        all_keys = set()

        if date in self.daily_beginning_inventory:
            all_keys.update(self.daily_beginning_inventory[date].keys())
        if date in self.daily_ending_inventory:
            all_keys.update(self.daily_ending_inventory[date].keys())

        for record in self.production_gr_by_date.get(date_str, []):
            all_keys.add((record['material'], record['location']))

        for record in self.delivery_gr_by_date.get(date_str, []):
            all_keys.add((record['material'], record['receiving']))

        for record in self.shipment_log_by_date.get(date_str, []):
            all_keys.add((record['material'], record['location']))

        # 发运出库数据
        delivery_ship_data = {}
        for record in self.delivery_shipment_log_by_date.get(date_str, []):
            material = record['material']
            sending = record['sending']
            quantity = float(record['quantity'])
            key = (material, sending)
            old_val = delivery_ship_data.get(key, 0)
            delivery_ship_data[key] = old_val + quantity
            all_keys.add(key)

        change_log = []
        for material, location in all_keys:
            beginning_qty = 0
            if date in self.daily_beginning_inventory:
                beginning_qty = self.daily_beginning_inventory[
                    date
                ].get((material, location), 0)

            production_qty = sum(
                rec['quantity']
                for rec in self.production_gr_by_date.get(date_str, [])
                if rec['material'] == material and rec['location'] == location
            )

            delivery_qty = sum(
                rec['quantity']
                for rec in self.delivery_gr_by_date.get(date_str, [])
                if rec['material'] == material and rec['receiving'] == location
            )

            shipment_qty = sum(
                rec['quantity']
                for rec in self.shipment_log_by_date.get(date_str, [])
                if rec['material'] == material and rec['location'] == location
            )

            delivery_ship_qty = delivery_ship_data.get(
                (material, location), 0
            )

            ending_qty = 0
            if date in self.daily_ending_inventory:
                ending_qty = self.daily_ending_inventory[
                    date
                ].get((material, location), 0)

            if (
                beginning_qty != 0 or production_qty != 0
                or delivery_qty != 0 or shipment_qty != 0
                or delivery_ship_qty != 0 or ending_qty != 0
            ):
                calculated_ending = (
                    beginning_qty
                    + production_qty
                    + delivery_qty
                    - shipment_qty
                    - delivery_ship_qty
                )
                if calculated_ending < 0:
                    calculated_ending = 0

                change_log.append({
                    'date': date_obj,
                    'material': material,
                    'location': location,
                    'beginning_inventory': beginning_qty,
                    'production_gr': production_qty,
                    'delivery_gr': delivery_qty,
                    'shipment': shipment_qty,
                    'delivery_ship': delivery_ship_qty,
                    'ending_inventory': ending_qty,
                    'calculated_ending': calculated_ending,
                    'balance_diff': ending_qty - calculated_ending,
                })

        df = pd.DataFrame(change_log)
        if df.empty:
            df = pd.DataFrame(columns=[
                'date', 'material', 'location',
                'beginning_inventory', 'production_gr',
                'delivery_gr', 'shipment', 'delivery_ship',
                'ending_inventory', 'calculated_ending', 'balance_diff',
            ])
        return self._stable_sort_inventory_change_log(df)

    def save_beginning_inventory(self, date: str):
        """向后兼容：保存期初库存。"""
        self.daily_beginning_inventory[date] = (
            self.unrestricted_inventory.copy()
        )

    def save_ending_inventory(self, date: str):
        """向后兼容：保存期末库存。"""
        self.daily_ending_inventory[date] = (
            self.unrestricted_inventory.copy()
        )

    # 旧接口兼容：process_module1_shipments = apply_shipments
    process_module1_shipments = apply_shipments
    process_module4_production = apply_production
    process_module5_deployment = apply_deployment
    process_module6_delivery = apply_delivery

    def set_past_due_cleanup_grace_days(self, days: int):
        """向后兼容：设置过期清理宽限天数。"""
        try:
            self.cleanup_grace_days = max(0, int(days))
        except Exception:
            self.cleanup_grace_days = 0

    def initialize_inventory(self, df: pd.DataFrame):
        """向后兼容：初始化库存。"""
        self._init_inventory(df)

    def set_space_capacity(self, df: pd.DataFrame):
        """向后兼容：设置空间容量。"""
        self._init_space_capacity(df)

    def cleanup_past_due_open_deployments(self, date, grace_days=0,
                                           write_audit=True):
        """向后兼容：清理过期调拨。"""
        self._cleanup_past_due(date)

    def _process_delivery_arrivals_compat(self, date):
        """向后兼容：处理到货。"""
        self._process_delivery_arrivals(date)

    # ══════════════════════════════════════════
    # 工具方法
    # ══════════════════════════════════════════

    @staticmethod
    def _safe_convert_to_int(value):
        """安全转换 pandas Series 或标量为整数。

        源: orchestrator_main.py:196-235 Orchestrator._safe_convert_to_int()
        """
        try:
            if hasattr(value, 'iloc') and len(value) > 0:
                value = value.iloc[0]
            elif hasattr(value, 'item'):
                value = value.item()
            elif isinstance(value, pd.Series):
                if len(value) == 1:
                    value = value.iloc[0]
                elif len(value) > 1:
                    value = value.iloc[0]
                else:
                    return 0

            if value is None or pd.isna(value):
                return 0

            return int(float(value))

        except (ValueError, TypeError, IndexError, AttributeError):
            return 0

    def _log_event(self, event_type: str, message: str):
        """记录事件用于审计追踪。

        源: persistence.py:224-240 OrchestratorPersistenceMixin._log_event()
        """
        self.daily_logs.append({
            'timestamp': datetime.now().isoformat(),
            'date': (
                self.current_date.strftime('%Y-%m-%d')
                if self.current_date is not None
                else ''
            ),
            'event_type': event_type,
            'message': message,
        })

    @staticmethod
    def _stable_sort_inventory_change_log(df: pd.DataFrame) -> pd.DataFrame:
        """稳定排序库存变动日志。

        源: inventory_log.py:12-33
        """
        if df.empty:
            return df
        sort_cols = [
            c for c in [
                'date', 'material', 'location', 'beginning_inventory',
                'production_gr', 'delivery_gr', 'shipment',
                'delivery_ship', 'ending_inventory',
                'calculated_ending', 'balance_diff',
            ] if c in df.columns
        ]
        if not sort_cols:
            return df.reset_index(drop=True)
        return df.sort_values(by=sort_cols, kind='mergesort').reset_index(
            drop=True
        )

    # ══════════════════════════════════════════
    # Module 接口（StateContext 不做业务计算）
    # ══════════════════════════════════════════

    def prepare(self):
        pass

    def run(self):
        pass
