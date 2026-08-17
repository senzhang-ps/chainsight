"""Module4 集成模式主入口 — ModuleFour 类。

本模块提供 M4 生产计划模块的面向对象封装：
- 无约束日计划构建（按 material/location/line 汇总，按 min_batch / rv 向上取整）
- 产能分配（含换产与跨天连续性）—— 逐日/逐批次顺序循环（照搬 CapacityAllocator）
- 生产可靠性仿真、换产指标、产线状态与已分配产能提取（向量化）

通过 ``engine`` 参数支持 pandas / polars 两种计算后端（默认 pandas）。
计算逻辑委托到 ``.backends`` 中的 _PandasBackend / _PolarsBackend。
"""

import logging
from typing import Optional

import pandas as pd

from ..module import Module
from .backends import (
    _PandasBackend,
    _PolarsBackend,
    dedup_issues,
    compute_planning_window,
    safe_float_conversion,
    UNCONSTRAINED_PLAN_COLUMNS,
)

logger = logging.getLogger("SupplyChainSimulation")


class ModuleFour(Module):
    """M4 生产计划模块 — 无约束日计划构建、产能分配与生产仿真。

    Parameters
    ----------
    net_demand : pd.DataFrame, optional
        当日净需求（M3 输出），列含 ``material / location / requirement_date / quantity``。
    net_demand_path : str, optional
        净需求文件路径；当 ``net_demand`` 未提供时使用。
    module3_result : dict, optional
        M3 内存结果（优先级最高），取 ``net_demand_df`` 作为净需求。
    previous_line_states_override : dict, optional
        前一日产线状态（由外部 StateContext 注入），缺省 {} 视为全新开始。
    allocated_capacity_override : dict, optional
        历史已分配产能（由外部 StateContext 注入），缺省 {} 视为全新开始。
    """

    schema = {
        'M4_MaterialLocationLineCfg': {
            'material': 'str',
            'location': 'str',
            'delegate_line': 'str',
            'prd_rate': 'int',
            'min_batch': 'float',
            'rv': 'float',
            'ptf': 'int',
            'lsk': 'int',
            'day': 'int',
            'mct': 'int',
        },
        'M4_LineCapacity': {
            'location': 'str',
            'line': 'str',
            'date': 'datetime',
            'capacity': 'int',
        },
        'M4_ChangeoverMatrix': {
            'from_material': 'str',
            'to_material': 'str',
            'changeover_id': 'str',
            'from line': 'str',
            'to line': 'str',
        },
        'M4_ChangeoverDefinition': {
            'changeover_id': 'str',
            'line': 'str',
            'time': 'float',
            'cost': 'float',
            'mu_loss': 'float',
        },
        'M4_ProductionReliability': {
            'location': 'str',
            'line': 'str',
            'pr': 'float',
        },
    }

    def __init__(self, simulation_date, simulation_start_date,
                 orchestrator=None, orch=None,
                 verbose=False, config=None,
                 net_demand: Optional[pd.DataFrame] = None,
                 net_demand_path: Optional[str] = None,
                 previous_line_states_override: Optional[dict] = None,
                 allocated_capacity_override: Optional[dict] = None,
                 module3_result: Optional[dict] = None):
        super().__init__(simulation_date, orch, 'M4', verbose, config=config)
        self.legacy_orchestrator = orchestrator
        self.net_demand = net_demand
        self.net_demand_path = net_demand_path
        self.simulation_start_date = simulation_start_date
        self.unconstrained_plan: Optional[pd.DataFrame] = None

        # 跨天状态与输出控制
        self.previous_line_states_override = previous_line_states_override
        self.allocated_capacity_override = allocated_capacity_override
        self.module3_result = module3_result
        self.issues: list = []

        # ── 引擎选择 ──────────────────────────────────────────────────
        if self._engine == 'polars':
            self._backend = _PolarsBackend(self)
        else:
            self._backend = _PandasBackend(self)

    # ------------------------------------------------------------------
    # 向后兼容属性 — 代理到 backend（测试 / StateContext 等外部代码
    # 可能直接读 m4.previous_line_states / previously_allocated）
    # ------------------------------------------------------------------

    @property
    def previous_line_states(self):
        return self._backend.previous_line_states

    @property
    def previously_allocated(self):
        return self._backend.previously_allocated

    # ------------------------------------------------------------------
    # 数据加载（engine-agnostic，pandas I/O）
    # ------------------------------------------------------------------

    def load_net_demand(self) -> pd.DataFrame:
        """加载净需求（M3 输出）并对齐 production_runner 预处理。"""
        if self.module3_result is not None and 'net_demand_df' in self.module3_result:
            df = self.module3_result['net_demand_df'].copy()
        elif self.net_demand is not None:
            df = self.net_demand.copy()
        elif self.net_demand_path is not None:
            df = pd.read_excel(self.net_demand_path)
        else:
            return pd.DataFrame()

        if df.empty:
            return df

        if 'layer' in df.columns:
            df = df[df['layer'] == 0].copy()
        if 'quantity' in df.columns:
            df['quantity'] = df['quantity'].abs()
        if 'material' in df.columns:
            df['material'] = df['material'].astype(str)
        if 'location' in df.columns:
            df['location'] = df['location'].astype(str)
        if 'requirement_date' in df.columns:
            df['requirement_date'] = pd.to_datetime(df['requirement_date'])

        return df

    def is_offset_review_day(self, row, days_since_start: int) -> bool:
        """判断当前 simulation_date 是否落在该 mlcfg 行的 review cycle 上。"""
        first_review_day = int(row['day']) - 1
        is_on_cycle = (days_since_start - first_review_day) % int(row['lsk']) == 0
        is_after_first = days_since_start >= first_review_day
        return is_on_cycle and is_after_first

    # ------------------------------------------------------------------
    # 步骤方法（委托到 backend）
    # ------------------------------------------------------------------

    def build_unconstrained_plan(self):
        return self._backend.build_unconstrained_plan()

    def _prepare_allocator_inputs(self):
        self._backend._prepare_allocator_inputs()

    def _reset_capacity_map(self):
        self._backend._reset_capacity_map()

    def _load_previous_states(self):
        self._backend._load_previous_states()

    def allocate_capacity(self, uncon):
        return self._backend.allocate_capacity(uncon)

    def simulate_reliability(self, plan_log):
        return self._backend.simulate_reliability(plan_log)

    def calc_changeover_metrics(self, plan_log):
        return self._backend.calc_changeover_metrics(plan_log)

    def extract_allocated_capacity(self, plan_log):
        return self._backend.extract_allocated_capacity(plan_log)

    def extract_line_states(self, plan_log):
        return self._backend.extract_line_states(plan_log)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def prepare(self):
        """一次性数据准备（循环前调用一次，日期无关）：
        1) 加载静态配置表 → self.datas；2) 标识符列归一；3) 构建分配器静态查找表。
        逐日计算（净需求、无约束计划、跨天状态、cap_map 重置）移至 run()。
        """
        # 0) 数据加载（委托给 orchestrator）
        if self.orchestrator is not None:
            self.orchestrator.load_datas(self)

        # 0.5) 按 schema 将标识符列转为 str
        self._cast_datas_identifiers()

        # 1) 分配器静态输入（mlcfg / co_def / co_mat / cap_map / rate_map / mct_map）
        self._prepare_allocator_inputs()

    def run(self):
        day = pd.Timestamp(self.simulation_date).normalize()
        logger.info("2️⃣ 运行 Module4 - 生产计划：%s", day.date())
        try:
            # ── 逐日准备（每日 run 时执行，prepare 只做一次性静态准备）──
            self.issues = []                          # 逐日重置问题清单
            self._load_previous_states()              # 跨天状态（前日产线状态 + 历史已分配产能）
            self._reset_capacity_map()                # 重置 cap_map（防跨天产能污染）
            self.unconstrained_plan = self.build_unconstrained_plan()  # 读当日 module3_result

            unconstrained_plan = self.validate_data(
                self.unconstrained_plan,
                name='unconstrained_plan',
                numeric_columns=['uncon_planned_qty', 'original_quantity'],
                required_columns=['material', 'location', 'line', 'planned_date'],
            )

            # 分配产能
            plan_log, exceed_log = self.allocate_capacity(unconstrained_plan)

            # 仿真生产可靠性
            plan_log = self.simulate_reliability(plan_log)

            # 计算换产指标
            changeover_log = self.calc_changeover_metrics(plan_log)

            # 提取当天产线状态供下一天使用
            current_line_states = self.extract_line_states(plan_log)

            # 提取当天分配的产能供后续仿真日期使用
            current_allocated_capacity = self.extract_allocated_capacity(plan_log)

            # 去重问题 → DataFrame
            issues = dedup_issues(self.issues)
            issues_df = pd.DataFrame(issues) if issues else pd.DataFrame()

            # production_df：available_date >= simulation_date 的记录
            # （plan_log 可能是 polars —— 先转 pandas 再做 pandas 风格过滤）
            plan_log_pd = self._to_pandas(plan_log)
            production_df = pd.DataFrame()
            if not plan_log_pd.empty and 'available_date' in plan_log_pd.columns:
                plan_log_pd['available_date'] = pd.to_datetime(plan_log_pd['available_date'])
                current_production = plan_log_pd[
                    plan_log_pd['available_date'] >= pd.Timestamp(self.simulation_date).normalize()
                ]
                if not current_production.empty:
                    production_df = current_production.copy()

            # ── 统一输出为 pandas（持久化由外部 Orch 负责） ────────────
            self._result = {
                'production_df': self._to_pandas(production_df),
                'exceed_log': self._to_pandas(exceed_log),
                'issues_df': self._to_pandas(issues_df),
                'changeover_log': self._to_pandas(changeover_log),
                'current_line_states': current_line_states,
                'current_allocated_capacity': current_allocated_capacity,
                'unconstrained_plan': self._to_pandas(unconstrained_plan),
            }
            logger.info(
                "✅ Module4 完成 - 生产计划=%d, 产能超限=%d, 换产=%d",
                len(self._result['production_df']),
                len(self._result['exceed_log']),
                len(self._result['changeover_log']),
            )
        except Exception as error:
            import traceback
            traceback.print_exc()
            logger.exception("❌ Module4 失败: %s", error)
            self._empty_result()

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------

    def output(self):
        """返回 run() 产出的结果字典。"""
        return self._result

    # ------------------------------------------------------------------
    # 转换辅助
    # ------------------------------------------------------------------

    def _to_pandas(self, df):
        if df is None:
            return pd.DataFrame()
        if isinstance(df, pd.DataFrame):
            return df
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
        return df

    # ------------------------------------------------------------------
    # validate_data — 委托到对应 backend
    # ------------------------------------------------------------------

    def validate_data(self, df, name='data', numeric_columns=None,
                      required_columns=None, strict=False):
        if self._engine == 'polars':
            return self._backend.validate_data(
                df, name=name, numeric_columns=numeric_columns,
                required_columns=required_columns, strict=strict,
            )
        return super().validate_data(
            df, name=name, numeric_columns=numeric_columns,
            required_columns=required_columns, strict=strict,
        )

    # ------------------------------------------------------------------
    # 私有工具方法
    # ------------------------------------------------------------------

    def _cast_datas_identifiers(self):
        """按 schema 将 self.datas 中声明为 'str' 的列强制转为 str dtype，
        并对 location 列做零填充归一化。
        """
        from .utils import cast_identifiers_to_str
        for sheet_name, col_schema in self.schema.items():
            df = self.datas.get(sheet_name)
            if df is None or df.empty:
                continue
            str_cols = [col for col, dtype in col_schema.items() if dtype == 'str']
            self.datas[sheet_name] = cast_identifiers_to_str(df, cols=str_cols)

    def _empty_result(self):
        self._result = {
            'production_df': pd.DataFrame(),
            'exceed_log': pd.DataFrame(),
            'issues_df': pd.DataFrame(),
            'changeover_log': pd.DataFrame(),
            'current_line_states': {},
            'current_allocated_capacity': {},
            'unconstrained_plan': pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS),
        }
