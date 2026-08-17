"""Module5 集成模式主入口 — ``ModuleFive`` 类。

本模块只承载 M5 的生命周期和引擎选择：
- ``prepare()``：由 Orch 调用，加载一次静态配置；
- ``run()``：由 Orch 对每个仿真日调用，执行当日部署计划；
- pandas / Polars DataFrame 计算：委托给 ``.backends``。

``prepare()`` 与 ``run()`` 是相互独立的 Orch 生命周期步骤；``run()``
绝不隐式调用 ``prepare()``。
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ..module import Module
from .backends import _PandasBackend, _PolarsBackend

logger = logging.getLogger("SupplyChainSimulation")


class ModuleFive(Module):
    """M5 部署计划模块 — 静态网络准备与逐日部署计划调度。

    Parameters
    ----------
    simulation_start_date : datetime-like
        仿真起始日期，用于保留模块的既有生命周期契约。
    state_context : StateContext, optional
        当日库存、需求、在途、生产及空间约束的动态事实来源。
    orch : Orch, optional
        配置注入与计算引擎选择器；``engine`` 为 ``pandas`` 或 ``polars``。
    """

    schema = {
        "M3_SafetyStock": {
            "material": "str", "location": "str", "date": "datetime", "quantity": "float",
        },
        "Global_Network": {
            "material": "str", "location": "str", "sourcing": "str",
            "eff_from": "datetime", "eff_to": "datetime",
        },
        "Global_LeadTime": {
            "sending": "str", "receiving": "str", "PDT": "int", "GR": "int", "MCT": "int",
        },
        "Global_DemandPriority": {"demand_element": "str", "priority": "int"},
        "M5_PushPullModel": {"material": "str", "sending": "str", "model": "str"},
        "M5_DeployConfig": {"material": "str", "sending": "str"},
        "M5_SupplyDemandLog": {
            "material": "str", "location": "str", "date": "datetime",
            "demand_element": "str", "quantity": "float",
        },
        "M4_MaterialLocationLineCfg": {"material": "str", "location": "str"},
    }

    def __init__(
        self,
        simulation_date,
        simulation_start_date,
        state_context=None,
        orch=None,
        verbose: bool = False,
        config: Optional[dict] = None,
    ):
        super().__init__(simulation_date, orch, "M5", verbose, config=config)
        self.state_context = state_context
        self.simulation_start_date = pd.Timestamp(simulation_start_date).normalize()
        self._prepared = False

        # 引擎选择与 M1/M4 一致；所有 DataFrame 业务计算留在 backend。
        self._backend = (
            _PolarsBackend(self) if self._engine == "polars" else _PandasBackend(self)
        )
        self._empty_result()

    # ------------------------------------------------------------------
    # 向后兼容状态属性 — 只读代理到 backend
    # ------------------------------------------------------------------

    @property
    def static(self):
        return self._backend.static

    @property
    def static_config(self):
        return self._backend.static_config

    @property
    def layer_map(self):
        return self._backend.layer_map

    @property
    def location_to_layer(self):
        return self._backend.location_to_layer

    @property
    def layers(self):
        return self._backend.layers

    # ------------------------------------------------------------------
    # 计算步骤（委托到 backend；两个 engine 使用相同的门面编排）
    # ------------------------------------------------------------------

    def load_static_data(self):
        if self.orchestrator is None and not self.datas:
            raise RuntimeError("ModuleFive.prepare() 需要由 Orch 注入配置")
        if not self.datas:
            self.orchestrator.load_datas(self)

    def normalise_static_config(self):
        return self._backend.normalise_static_config(self.datas)

    def validate_static_network(self, static):
        return self._backend.validate_static_network(static)

    def build_network_layers(self, network):
        return self._backend.build_network_layers(network)

    def store_static_state(self, static, layer_map):
        self._backend.store_static_state(static, layer_map)

    def load_daily_inputs(self, day):
        return self._backend.daily_inputs(day)

    def build_active_network(self, config, day):
        return self._backend.active_network(config, day)

    def validate_config(self, config, active):
        return self._backend.validate(config, active)

    def build_route_parameters(self, active, config):
        return self._backend.route_parameters(active, config)

    def build_node_horizon(self, active, day, routes):
        return self._backend.node_horizon(active, day, routes)

    def build_direct_demand(self, active, day, config, routes):
        return self._backend.all_direct_demand(active, day, config, routes)

    def publish_planning_facts(self, day, active, routes, node_horizon, direct_demand):
        if self.state_context is None:
            return
        # 保持 M5 的最小独立单元测试契约：不具备 StateContext 发布接口的
        # fake state 仅用于验证 M5 本身的计算，不参与 M3 日内调度。
        if not hasattr(self.state_context, "publish_planning_facts"):
            return
        self.state_context.publish_planning_facts(day, {
            "version": 1,
            "simulation_date": day,
            "active_network": self._backend.to_pandas(active),
            "routes": self._backend.to_pandas(routes),
            "node_horizon": self._backend.to_pandas(node_horizon),
            "direct_demand": self._backend.to_pandas(direct_demand),
            "layer_map": dict(self._backend.layer_map),
            "layers": list(self._backend.layers),
        })

    def build_supply_ledger(self, config, day):
        return self._backend.supply_ledger(config, day)

    def build_layer_plan(self, day, config, active, routes, priority, available, pools,
                         node_horizon=None, direct_demand=None):
        if node_horizon is None and direct_demand is None:
            return self._backend.plan_layers(
                day, config, active, routes, priority, available, pools,
            )
        return self._backend.plan_layers(
            day, config, active, routes, priority, available, pools,
            node_horizon=node_horizon, direct_demand=direct_demand,
        )

    def build_push_plan(self, plan, direct, active, routes, config, available, projected, day):
        return self._backend.push(plan, direct, active, routes, config, self._backend.available_supply(available), self._backend.projected_supply(projected), day)

    def append_plan(self, plan, push):
        return self._backend.append_plan(plan, push)

    def apply_space_constraints(self, plan, space, priority):
        return self._backend.apply_space(plan, space, priority)

    def append_unfulfilled(self, unfulfilled, space_unfulfilled):
        return self._backend.append_unfulfilled(unfulfilled, space_unfulfilled)

    def finalise_result(self, plan, unfulfilled, available, today_transit, shipment, validation, day):
        return self._backend.finalise_result(plan, unfulfilled, available, today_transit, shipment, validation, day)

    # ------------------------------------------------------------------
    # 生命周期（由 Orch 独立调用）
    # ------------------------------------------------------------------

    def prepare(self):
        """加载一次静态配置，并建立 backend 的网络与层级状态。"""
        if self._prepared:
            return

        self.load_static_data()
        static = self.normalise_static_config()
        network = self.validate_static_network(static)
        layer_map = self.build_network_layers(network)
        self.store_static_state(static, layer_map)
        self._prepared = True
        logger.info("✅ Module5 静态配置准备完成（engine=%s）", self._engine)

    def run(self):
        """执行当前仿真日的部署计划，不执行配置准备。"""
        if not self._prepared:
            raise RuntimeError("ModuleFive.run() 需要 Orch 先调用 prepare()")

        day = pd.Timestamp(self.simulation_date).normalize()
        logger.info("3️⃣ 运行 Module5 - 部署计划：%s", day.date())
        config = self.load_daily_inputs(day)
        active = self.build_active_network(config, day)
        priority, validation = self.validate_config(config, active)
        routes = self.build_route_parameters(active, config)
        node_horizon = self.build_node_horizon(active, day, routes)
        direct_demand = self.build_direct_demand(active, day, config, routes)
        self.publish_planning_facts(
            day, active, routes, node_horizon, direct_demand
        )
        available, pools, projected, today_transit, shipment = self.build_supply_ledger(config, day)
        plan, direct, unfulfilled = self.build_layer_plan(
            day, config, active, routes, priority, available, pools,
            node_horizon=node_horizon if self.state_context is not None else None,
            direct_demand=direct_demand if self.state_context is not None else None,
        )
        push = self.build_push_plan(plan, direct, active, routes, config, available, projected, day)
        plan = self.append_plan(plan, push)
        plan, space_unfulfilled = self.apply_space_constraints(plan, config["ReceivingSpace"], priority)
        unfulfilled = self.append_unfulfilled(unfulfilled, space_unfulfilled)
        self._result = self.finalise_result(
            plan, unfulfilled, available, today_transit, shipment, validation, day
        )
        self._backend.result = self._result
        logger.info(
            "✅ Module5 完成 - 部署=%d, 未满足=%d",
            len(self._result.get('deployment_plan', pd.DataFrame())),
            len(self._result.get('unfulfilled_log', pd.DataFrame())),
        )

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------

    def _empty_result(self):
        self._result = self._backend.empty_result()


__all__ = ["ModuleFive"]
