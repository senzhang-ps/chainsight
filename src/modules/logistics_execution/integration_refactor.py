"""Module6 集成重构门面。

``ModuleSix`` 对齐当前 ``Orch`` / ``StateContext`` 生命周期：

* ``prepare()`` 只加载和验证静态 M6 配置；
* ``run()`` 只读取当天 Context view、执行物流仿真并生成输出；
* 发运结果由外层调用 ``StateContext.apply_delivery()`` 写回状态，模块本身不
  突变 Context。

旧的 ``run_daily_physical_flow`` API 保留在 :mod:`.main`，以便在完成真实数据
strict parity 前继续支撑现有集成调度。
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ...utils.defaults import M6_MAX_WAIT_DAYS
from ..module import Module
from .backends import _PandasBackend, _PolarsBackend

logger = logging.getLogger("SupplyChainSimulation")


class ModuleSix(Module):
    """M6 物流执行模块的显式门面。"""

    schema = {
        "M6_TruckReleaseCon": {
            "sending": "str", "receiving": "str", "truck_type": "str",
            "WFR": "float", "VFR": "float",
        },
        "M6_TruckCapacityPlan": {
            "sending": "str", "receiving": "str", "truck_type": "str",
        },
        "M6_TruckTypeSpecs": {"truck_type": "str"},
        "M6_MaterialMD": {"material": "str"},
        "M6_DeliveryDelayDistribution": {"sending": "str", "receiving": "str"},
        "M6_MDQBypassRules": {"sending": "str", "receiving": "str"},
        "Global_DemandPriority": {"demand_element": "str", "priority": "int"},
        "Global_LeadTime": {"sending": "str", "receiving": "str"},
    }

    def __init__(
        self,
        simulation_date,
        state_context=None,
        orch=None,
        *,
        max_wait_days: int = M6_MAX_WAIT_DAYS,
        random_seed: Optional[int] = None,
        verbose: bool = False,
        config: Optional[dict] = None,
    ):
        super().__init__(simulation_date, orch, "M6", verbose, config=config)
        self.state_context = state_context
        self.max_wait_days = max_wait_days
        self.random_seed = random_seed
        self._prepared = False
        self._backend = (
            _PolarsBackend(self) if self._engine == "polars" else _PandasBackend(self)
        )
        self._result = self._backend.empty_result()

    # Explicit, engine-neutral lifecycle steps ---------------------------------
    def load_static_data(self):
        if not self.datas:
            if self.orchestrator is None:
                raise RuntimeError("ModuleSix.prepare() 需要 Orch 注入配置")
            self.orchestrator.load_datas(self)

    def normalise_static_config(self):
        return self._backend.normalise_static_config(self.datas)

    def validate_static_config(self, static):
        return self._backend.validate_static_config(static)

    def store_static_state(self, static):
        self._backend.store_static_state(static)

    def load_daily_inputs(self, day: pd.Timestamp):
        return self._backend.load_daily_inputs(day)

    def prepare_daily_data(self, inputs, day: pd.Timestamp):
        return self._backend.prepare_daily_data(inputs, day)

    def execute_daily_flow(self, run_params, prepared_data):
        return self._backend.execute_daily_flow(run_params, prepared_data)

    def finalise_result(self, run_params, results, validation_log):
        return self._backend.finalise_result(run_params, results, validation_log)

    def prepare(self):
        """加载一次静态配置；不读取或修改每日物流状态。"""
        if self._prepared:
            return
        self.load_static_data()
        static = self.normalise_static_config()
        self.validate_static_config(static)
        self.store_static_state(static)
        self._prepared = True

    def run(self):
        """执行当前仿真日；调用方负责随后以 ``apply_delivery`` 写回 Context。"""
        if not self._prepared:
            raise RuntimeError("ModuleSix.run() 需要 Orch 先调用 prepare()")
        if self.state_context is None:
            raise RuntimeError("ModuleSix.run() 需要 StateContext")

        day = pd.Timestamp(self.simulation_date).normalize()
        logger.info("4️⃣ 运行 Module6 - 物流执行：%s", day.date())
        inputs = self.load_daily_inputs(day)
        run_params, prepared_data = self.prepare_daily_data(inputs, day)
        results = self.execute_daily_flow(run_params, prepared_data)
        self._result = self.finalise_result(
            run_params, results, prepared_data["validation_log"]
        )
        self._backend.result = self._result
        logger.info(
            "✅ Module6 完成 - 交付=%d, 车辆=%d, 未满足=%d",
            len(self._result['delivery_plan']),
            len(self._result['vehicle_log']),
            len(self._result['unsatisfied_log']),
        )

    def output(self):
        return self._result


__all__ = ["ModuleSix"]
