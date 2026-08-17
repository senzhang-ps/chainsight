"""Module3 MRP 门面：生命周期编排与引擎选择。"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ..module import Module
from .backends import _PandasBackend, _PolarsBackend

logger = logging.getLogger("SupplyChainSimulation")


class ModuleThree(Module):
    """M3 净需求模块，消费 M5→M3 的同日共享规划事实。"""

    schema = {
        "M3_SafetyStock": {"material": "str", "location": "str", "date": "datetime", "quantity": "float"},
        "Global_Network": {"material": "str", "location": "str", "sourcing": "str", "eff_from": "datetime", "eff_to": "datetime"},
        "Global_LeadTime": {"sending": "str", "receiving": "str", "PDT": "int", "GR": "int", "MCT": "int"},
        "M4_MaterialLocationLineCfg": {"material": "str", "location": "str"},
        "M5_DeployConfig": {"material": "str", "sending": "str"},
    }

    def __init__(self, simulation_date, simulation_start_date, output_dir='',
                 orchestrator=None, orch=None, skip_file_output=False,
                 state_context=None,
                 verbose=False, config=None):
        super().__init__(simulation_date, orch, 'M3', verbose, config=config)
        self.legacy_orchestrator = orchestrator
        self.state_context = state_context
        self.output_dir = output_dir
        self.skip_file_output = skip_file_output
        self.simulation_start_date = pd.Timestamp(simulation_start_date).normalize()
        self._prepared = False
        self._backend = (
            _PolarsBackend(self) if self._engine == 'polars' else _PandasBackend(self)
        )
        self._result = self._backend.empty_result()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def prepare(self):
        """加载一次静态配置；共享规划事实在 M5 日运行时才会产生。"""
        if self._prepared:
            return
        if self.orchestrator is None and not self.datas:
            raise RuntimeError('ModuleThree.prepare() 需要由 Orch 注入配置')
        if not self.datas:
            self.orchestrator.load_datas(self)
        self._backend.store_static_state(
            self._backend.normalise_static_config(self.datas)
        )
        self._prepared = True
        logger.info('✅ Module3 静态配置准备完成（engine=%s）', self._engine)

    def run(self):
        """消费 M5 PlanningFacts 和 M6 后供给状态，计算当日净需求。"""
        if not self._prepared:
            raise RuntimeError('ModuleThree.run() 需要 Orch 先调用 prepare()')
        if self.state_context is None:
            raise RuntimeError('ModuleThree.run() 需要 StateContext')
        day = pd.Timestamp(self.simulation_date).normalize()
        logger.info('5️⃣ 运行 Module3 - 净需求计算：%s', day.date())
        facts = self._backend.planning_facts(
            self.state_context.get_planning_facts(day)
        )
        supply = self._backend.supply_views(self.state_context, day)
        self._result = self._backend.finalise_result(
            self._backend.calculate_layers(day, facts, supply)
        )
        self.state_context.apply_m3_net_demand(
            self._result['net_demand_df'], day
        )
        self._backend.result = self._result
        logger.info(
            '✅ Module3 完成 - 净需求=%d',
            self._result['net_demand_count'],
        )
