"""Module4 集成模式主入口 — ModuleFour 类。

本模块提供 M4 生产计划模块的面向对象封装：
- 无约束日计划构建（按 material/location/line 汇总，按 min_batch / rv 向上取整）
- 产能分配（含换产与跨天连续性）—— 逐日/逐批次顺序循环（照搬 CapacityAllocator）
- 生产可靠性仿真、换产指标、产线状态与已分配产能提取（向量化）

4 个纯计算步骤已向量化（直接对 self.datas[…] 做 merge）；
产能分配器内部是点查式顺序循环，依赖 _prepare_allocator_inputs() 构建的
dict/Series（rate_map / co_mat / co_def / cap_map / mct_map）。
"""

import logging
import os
from datetime import timedelta
from typing import Any, Optional

import pandas as pd
import numpy as np

from ..module import Module
from .constants import (
    UNCONSTRAINED_PLAN_COLUMNS,
    DEFAULT_CHANGEOVER_TIME,
    CHANGEOVER_LOG_COLUMNS,
)
from .capacity_allocator import _lookup_rate
from .plan_builder import optimal_changeover_sequence
from .utils import safe_float_conversion, dedup_issues
from .state_manager import (
    load_line_state,
    save_line_state,
    load_all_previous_capacity,
    save_allocated_capacity,
)
from .output_writer import write_output
from ...utils.defaults import M6_RANDOM_SEED
from ...utils.normalization import normalize_material
from src.utils.date_helpers import compute_planning_window

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
        前一日产线状态（内存模式），提供则跳过文件读取。
    allocated_capacity_override : dict, optional
        历史已分配产能（内存模式），提供则跳过文件读取。
    skip_state_file_output : bool
        是否跳过 line_states / allocated_capacity JSON 写盘。
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
            'MCT': 'int',
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

    def __init__(self, simulation_date, simulation_start_date, output_dir='',
                 orchestrator=None, orch=None,
                 skip_file_output=False,
                 verbose=False, config=None,
                 net_demand: Optional[pd.DataFrame] = None,
                 net_demand_path: Optional[str] = None,
                 previous_line_states_override: Optional[dict] = None,
                 allocated_capacity_override: Optional[dict] = None,
                 skip_state_file_output: bool = False,
                 module3_result: Optional[dict] = None):
        super().__init__(simulation_date, orch, 'M4', verbose, config=config)
        self.legacy_orchestrator = orchestrator
        self.output_dir = output_dir
        self.skip_file_output = skip_file_output
        self.net_demand = net_demand
        self.net_demand_path = net_demand_path
        self.simulation_start_date = simulation_start_date
        self.unconstrained_plan: Optional[pd.DataFrame] = None

        # 跨天状态与输出控制
        self.previous_line_states_override = previous_line_states_override
        self.allocated_capacity_override = allocated_capacity_override
        self.skip_state_file_output = skip_state_file_output
        self.module3_result = module3_result
        self.issues: list = []

        # 分配器输入（_prepare_allocator_inputs 填充）
        self.mlcfg: Optional[pd.DataFrame] = None
        self.co_def_df: Optional[pd.DataFrame] = None
        self.co_mat: Optional[pd.Series] = None
        self.co_def: dict = {}
        self.cap_df: Optional[pd.DataFrame] = None
        self.cap_map: dict = {}
        self.has_location: bool = False
        self.rate_map: Optional[pd.Series] = None
        self.mct_map: dict = {}
        self.previous_line_states: dict = {}
        self.previously_allocated: dict = {}

    # ------------------------------------------------------------------
    # 步骤方法（TimedMeta 自动计时）
    # ------------------------------------------------------------------

    def load_net_demand(self) -> pd.DataFrame:
        """加载净需求（M3 输出）并对齐 production_runner 预处理：

        - 优先取 ``module3_result['net_demand_df']``，其次 ``net_demand``，最后文件；
        - 仅保留 ``layer == 0``（下游需求）；
        - ``quantity`` 取绝对值（M3 约定需求侧为负，M4 需要正值）；
        - ``material`` / ``location`` 转为字符串；
        - ``requirement_date`` 规范为 datetime。
        """
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

    def build_unconstrained_plan(self) -> pd.DataFrame:
        """构建无约束日计划：

        1. 从 ``self.datas['M4_MaterialLocationLineCfg']`` 取 mlcfg；
        2. 按 ``(simulation_date - simulation_start_date)`` 计算 ``days_since_start``，
           用 :meth:`is_offset_review_day` 标记并筛选 review day；
        3. 与 net_demand（M3 输出）按 material/location 关联，按 ``requirement_date``
           过滤到 simulation_date；
        4. 按 material/location/line 汇总，并按 ``min_batch`` / ``rv`` 向上取整。
        """
        if 'M4_MaterialLocationLineCfg' not in self.datas:
            return pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS)

        mat_loc_line_config = self.datas['M4_MaterialLocationLineCfg'].copy()

        # 1) review day 标记 + 筛选
        days_since_start = (
            pd.Timestamp(self.simulation_date) - pd.Timestamp(self.simulation_start_date)
        ).days
        mat_loc_line_config['is_offset_review_day'] = mat_loc_line_config.apply(
            lambda r: self.is_offset_review_day(r, days_since_start), axis=1,
        )
        mat_loc_line_config = mat_loc_line_config[
            mat_loc_line_config['is_offset_review_day']
        ].drop(columns=['is_offset_review_day'])

        if mat_loc_line_config.empty:
            return pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS)

        # 2) 关联 net_demand
        net_demand = self.load_net_demand()
        if net_demand is None or net_demand.empty:
            return pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS)
        net_demand['material'] = net_demand['material'].astype('str')

        df = pd.merge(
            mat_loc_line_config, net_demand,
            how='inner', on=['material', 'location'],
        )
        if df.empty or 'requirement_date' not in df.columns:
            return pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS)

        # 3) 按 simulation_date 过滤
        df['requirement_date'] = pd.to_datetime(df['requirement_date']).dt.normalize()
        sim_date = pd.Timestamp(self.simulation_date).normalize()
        df = df[df['requirement_date'] == sim_date]
        if df.empty:
            return pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS)

        # 4) 按 material/location/line 汇总，向上取整
        grouped = df.groupby(['material', 'location', 'delegate_line'], as_index=False).agg(
            original_quantity=('quantity', 'sum'),
            min_batch=('min_batch', 'first'),
            rv=('rv', 'first'),
        )

        grouped['uncon_planned_qty'] = grouped.apply(
            lambda r: np.ceil(max(r['original_quantity'], r['min_batch']) / round(r['rv'])) * round(r['rv'])
                      if max(r['original_quantity'], r['min_batch']) % round(r['rv']) != 0
                      else int(max(r['original_quantity'], r['min_batch'])),
            axis=1,
        ).round().astype(int)

        grouped = grouped.rename(columns={'delegate_line': 'line'})
        grouped['planned_date'] = sim_date
        grouped['simulation_date'] = sim_date

        # 标准化 material（复刻 production_runner L124-L125）
        grouped['material'] = grouped['material'].apply(normalize_material).astype('string')

        return grouped[UNCONSTRAINED_PLAN_COLUMNS]

    def is_offset_review_day(self, row, days_since_start: int) -> bool:
        """对齐 :func:`src.utils.date_helpers.is_offset_review_day` 的语义：
        判断当前 simulation_date 是否落在该 mlcfg 行的 review cycle 上。
        """
        first_review_day = int(row['day']) - 1
        is_on_cycle = (days_since_start - first_review_day) % int(row['lsk']) == 0
        is_after_first = days_since_start >= first_review_day
        return is_on_cycle and is_after_first

    # ------------------------------------------------------------------
    # 分配器输入预处理 + 跨天状态加载
    # ------------------------------------------------------------------

    def _prepare_allocator_inputs(self):
        """构建产能分配器所需的 dict/Series（仅分配器与部分提取逻辑消费）。

        复刻 production_runner L127-L153。向量化函数（simulate /
        changeover_metrics / extract_*）不读这些 dict/Series，而是直接 merge
        原始 ``self.datas[…]`` 帧。
        """
        mlcfg = self.datas['M4_MaterialLocationLineCfg'].copy()
        self.mlcfg = mlcfg

        # ChangeoverDefinition —— changeover_id 转 str（向量化函数也用它 merge）
        co_def_df = self.datas['M4_ChangeoverDefinition'].copy()
        co_def_df['changeover_id'] = co_def_df['changeover_id'].astype(str)
        self.co_def_df = co_def_df

        # ChangeoverMatrix —— 分配器点查用 Series
        co_mat_df = self.datas['M4_ChangeoverMatrix'].copy()
        co_mat_df['from_material'] = co_mat_df['from_material'].astype(str)
        co_mat_df['to_material'] = co_mat_df['to_material'].astype(str)
        co_mat_df['changeover_id'] = co_mat_df['changeover_id'].astype(str)
        self.co_mat = co_mat_df.set_index(
            ['from_material', 'to_material']
        )['changeover_id'].sort_index()

        # co_def dict —— 分配器点查
        self.co_def = co_def_df.set_index(
            ['changeover_id', 'line']
        )['time'].to_dict()

        # LineCapacity —— cap_df（提取用）+ cap_map（分配器就地改写）
        cap_df = self.datas['M4_LineCapacity'].copy()
        cap_df['date'] = pd.to_datetime(cap_df['date'])
        self.cap_df = cap_df
        self.has_location = 'location' in cap_df.columns
        self.cap_map = self._build_capacity_map(cap_df)

        # rate_map Series —— 分配器点查（index 改名 line）
        rate_map = mlcfg.set_index(
            ['material', 'location', 'delegate_line']
        )['prd_rate']
        rate_map.index.set_names(['material', 'location', 'line'], inplace=True)
        self.rate_map = rate_map.sort_index()

        # mct_map —— 分配器 _alloc_day 点查
        self.mct_map = mlcfg.set_index(['material', 'location'])['MCT'].to_dict()

    def _build_capacity_map(self, cap_df: pd.DataFrame) -> dict:
        """构建产能映射 dict（分配器就地改写）。对齐 CapacityAllocator._build_capacity_map。"""
        cap_df = cap_df.copy()
        cap_df['capacity'] = cap_df['capacity'].astype(float)
        if 'location' in cap_df.columns:
            return cap_df.set_index(['location', 'line', 'date'])['capacity'].to_dict()
        return cap_df.set_index(['line', 'date'])['capacity'].to_dict()

    def _load_previous_states(self):
        """加载前一日产线状态与历史已分配产能。override 优先，否则从文件读。

        对齐 production_runner L155-L181。
        """
        if self.previous_line_states_override is not None:
            self.previous_line_states = self.previous_line_states_override
        else:
            self.previous_line_states = load_line_state(
                self.output_dir, self.simulation_date
            )

        if self.allocated_capacity_override is not None:
            self.previously_allocated = self.allocated_capacity_override
        else:
            self.previously_allocated = load_all_previous_capacity(
                self.output_dir, self.simulation_date
            )

    # ------------------------------------------------------------------
    # 产能分配 —— 照搬 CapacityAllocator（保留逐日/逐批次顺序循环）
    # ------------------------------------------------------------------

    def allocate_capacity(
        self, uncon: pd.DataFrame
    ) -> tuple:
        """集中式产能分配（含跨天换产连续性）。照搬 CapacityAllocator.allocate。

        返回：(plan_log DataFrame, exceed_log DataFrame)
        """
        plans_log = []
        exceed_log = []

        if uncon.empty:
            return pd.DataFrame(plans_log), pd.DataFrame(exceed_log)

        uncon = uncon.sort_values(
            ['line', 'simulation_date', 'planned_date', 'material']
        ).reset_index(drop=True)

        for (line, sim_date), group in uncon.groupby(['line', 'simulation_date']):
            line_plans, line_exceed = self._alloc_line_group(line, sim_date, group)
            plans_log.extend(line_plans)
            exceed_log.extend(line_exceed)

        return pd.DataFrame(plans_log), pd.DataFrame(exceed_log)

    def _alloc_line_group(self, line, sim_date, group):
        batch_list = group.to_dict(orient='records')

        if len(batch_list) > 1:
            sequence = optimal_changeover_sequence(
                batch_list, self.co_mat, self.co_def, line
            )
            batch_list = [batch_list[i] for i in sequence]

        state = self._alloc_init_line_state(line)
        plans = []
        exceeds = []

        for idx, batch in enumerate(batch_list):
            batch_plans, batch_exceed = self._alloc_batch(
                line, sim_date, batch, idx, state
            )
            plans.extend(batch_plans)
            if batch_exceed:
                exceeds.append(batch_exceed)
            state['prev_mat'] = batch['material']

        return plans, exceeds

    def _alloc_init_line_state(self, line):
        state = {
            'prev_mat': None,
            'initial_co_remain': 0,
            'initial_coid': None,
            'has_incomplete_changeover': False,
        }

        if line not in self.previous_line_states:
            return state

        line_state = self.previous_line_states[line]
        state['prev_mat'] = line_state.get('last_material')

        changeover_info = line_state.get('changeover_info')
        if (
            line_state.get('last_activity') == 'changeover' and
            changeover_info
        ):
            remaining = changeover_info.get('remaining_time', 0)
            if remaining > 0:
                state['initial_co_remain'] = remaining
                state['initial_coid'] = changeover_info.get('changeover_id')
                state['has_incomplete_changeover'] = True
                state['prev_mat'] = changeover_info.get(
                    'to_material', state['prev_mat']
                )
            else:
                state['prev_mat'] = changeover_info.get(
                    'to_material', state['prev_mat']
                )

        return state

    def _alloc_batch(self, line, sim_date, batch, batch_idx, state):
        material = batch['material']
        location = batch['location']

        config_row = self.mlcfg[
            (self.mlcfg['material'] == material) &
            (self.mlcfg['location'] == location)
        ].iloc[0]

        lsk = int(config_row['lsk'])
        ptf = int(config_row['ptf'])
        window_start, window_end = compute_planning_window(
            batch['simulation_date'], ptf, lsk
        )

        changeover = self._alloc_calc_changeover(batch_idx, material, state, line)

        return self._alloc_to_horizon(
            line, sim_date, batch, location, material,
            window_start, window_end, changeover
        )

    def _alloc_calc_changeover(self, batch_idx, material, state, line):
        if batch_idx == 0 and state['has_incomplete_changeover']:
            return {
                'remain': state['initial_co_remain'],
                'coid': state['initial_coid'],
                'is_first': True,
            }

        prev_mat = state['prev_mat']
        if prev_mat is not None and prev_mat != material:
            coid, co_time = self._alloc_lookup_changeover(prev_mat, material, line)
            return {
                'remain': co_time,
                'coid': coid,
                'is_first': True,
            }

        return {'remain': 0, 'coid': None, 'is_first': False}

    def _alloc_lookup_changeover(self, from_mat, to_mat, line):
        try:
            from_str = str(from_mat)
            to_str = str(to_mat)

            if (from_str, to_str) not in self.co_mat.index:
                coid = f"MISSING_CO_{from_str}_to_{to_str}"
                self.issues.append({
                    'sheet': 'M4_ChangeoverMatrix',
                    'row': '',
                    'issue': (
                        f"缺失物料切换定义: {from_str} → {to_str}，"
                        f"产线 {line}。已使用默认时间。"
                    )
                })
                return coid, DEFAULT_CHANGEOVER_TIME

            coid_result = self.co_mat.loc[(from_str, to_str)]
            if isinstance(coid_result, pd.Series):
                coid = str(coid_result.iloc[0])
            else:
                coid = str(coid_result)

            if (coid, line) not in self.co_def:
                self.issues.append({
                    'sheet': 'M4_ChangeoverDefinition',
                    'row': '',
                    'issue': (
                        f"缺失changeover定义: {coid}, 产线 {line}。"
                        f"已使用默认时间。"
                    )
                })
                return coid, DEFAULT_CHANGEOVER_TIME

            return coid, self.co_def[(coid, line)]

        except Exception:
            return None, 0

    def _alloc_to_horizon(
        self, line, sim_date, batch, location, material,
        window_start, window_end, changeover
    ):
        plans = []
        prod_remain = batch['uncon_planned_qty']
        co_remain = changeover['remain']
        coid_to_log = changeover['coid']
        is_first_co_day = changeover['is_first']

        horizon_days = pd.date_range(window_start, window_end)

        for day_dt in horizon_days:
            result = self._alloc_day(
                line, sim_date, batch, location, material,
                day_dt, prod_remain, co_remain,
                coid_to_log, is_first_co_day
            )

            if result['plan']:
                plans.append(result['plan'])

            prod_remain = result['prod_remain']
            co_remain = result['co_remain']
            coid_to_log = result['coid_to_log']
            is_first_co_day = result['is_first_co_day']

            if prod_remain <= 0:
                break

        exceed = None
        if prod_remain > 0:
            exceed = {
                'material': batch['material'],
                'location': location,
                'line': line,
                'simulation_date': sim_date,
                'production_plan_date': window_end,
                'unmet_uncon_planned_qty': prod_remain,
            }

        return plans, exceed

    def _alloc_day(
        self, line, sim_date, batch, location, material,
        day_dt, prod_remain, co_remain, coid_to_log, is_first_co_day
    ):
        cap_key = self._alloc_capacity_key(location, line, day_dt)
        current_cap = self.cap_map.get(cap_key, 0)
        today_cap = self._alloc_adjust_prev(
            current_cap, location, line, day_dt
        )

        co_used, co_remain, today_cap, changeover_done = (
            self._alloc_consume_changeover(co_remain, today_cap)
        )

        if co_remain > 0 and coid_to_log:
            self.cap_map[cap_key] = current_cap - co_used
            return {
                'plan': None,
                'prod_remain': prod_remain,
                'co_remain': co_remain,
                'coid_to_log': coid_to_log,
                'is_first_co_day': is_first_co_day,
            }

        if co_remain == 0 and is_first_co_day:
            changeover_done = True

        rate = _lookup_rate(self.rate_map, material, location, line)
        can_produce = min(prod_remain, int(today_cap * rate))
        hours_used = can_produce / rate if rate else 0

        plan = None
        if can_produce > 0:
            record_coid = (
                coid_to_log if (changeover_done or is_first_co_day)
                else None
            )
            mct = self.mct_map.get((material, location), 0)

            plan = {
                'material': material,
                'location': location,
                'line': line,
                'simulation_date': sim_date,
                'production_plan_date': day_dt,
                'available_date': day_dt + timedelta(days=int(mct)),
                'uncon_planned_qty': batch['uncon_planned_qty'],
                'con_planned_qty': can_produce,
                'changeover_id': record_coid,
            }

            if coid_to_log is not None:
                coid_to_log = None
                is_first_co_day = False

        prod_remain -= can_produce
        self.cap_map[cap_key] = current_cap - co_used - hours_used

        return {
            'plan': plan,
            'prod_remain': prod_remain,
            'co_remain': co_remain,
            'coid_to_log': coid_to_log,
            'is_first_co_day': is_first_co_day,
        }

    def _alloc_capacity_key(self, location, line, day_dt):
        if self.has_location:
            return (location, line, day_dt)
        return (line, day_dt)

    def _alloc_adjust_prev(self, current_cap, location, line, day_dt):
        if not self.previously_allocated:
            return current_cap

        key = f"{location}|{line}|{day_dt.strftime('%Y-%m-%d')}"
        prev_used = self.previously_allocated.get(key, 0)

        return max(0, current_cap - prev_used)

    def _alloc_consume_changeover(self, co_remain, today_cap):
        if co_remain <= 0:
            return 0, 0, today_cap, False

        co_used = min(today_cap, co_remain)
        co_remain -= co_used
        today_cap -= co_used
        completed = (co_remain == 0)

        return co_used, co_remain, today_cap, completed

    # ------------------------------------------------------------------
    # 生产可靠性仿真 —— 向量化（保 RNG 序列：逐行 binomial）
    # ------------------------------------------------------------------

    def simulate_reliability(self, plan_log: pd.DataFrame) -> pd.DataFrame:
        """仿真生产可靠性。对齐 capacity_allocator.simulate_production。

        关键：不排序 plan；binomial 逐行抽样以保种子复现（与 plan.apply 的
        RNG 调用序列一致）。仅 pr 查找向量化（merge）。
        """
        pr_cfg = self.datas.get('M4_ProductionReliability')

        if (
            plan_log is None or plan_log.empty or
            'con_planned_qty' not in plan_log.columns or
            pr_cfg is None or pr_cfg.empty
        ):
            plan_log['produced_qty'] = []
            return plan_log

        seed = self.config.get('RandomSeed', M6_RANDOM_SEED)
        rng = np.random.RandomState(seed)

        # 向量化 pr 查找：drop_duplicates keep='last' 对齐 to_dict() 的 last 胜出
        pr_lookup = (
            pr_cfg[['location', 'line', 'pr']]
            .drop_duplicates(['location', 'line'], keep='last')
        )
        merged = plan_log.merge(pr_lookup, on=['location', 'line'], how='left')
        p = merged['pr'].fillna(1.0).to_numpy()
        n = plan_log['con_planned_qty'].astype(int).to_numpy()

        # 逐行抽样（与 plan.apply(simulate_row) 的 RNG 调用序列完全一致）
        plan_log['produced_qty'] = [
            rng.binomial(int(ni), float(pi)) for ni, pi in zip(n, p)
        ]
        return plan_log

    # ------------------------------------------------------------------
    # 换产指标 —— 向量化
    # ------------------------------------------------------------------

    def calc_changeover_metrics(self, plan_log: pd.DataFrame) -> pd.DataFrame:
        """计算换产指标。对齐 capacity_allocator.calculate_changeover_metrics。

        groupby 计数 + merge 换产定义 + 按计数倍乘。定义缺失计 0（对齐 KeyError→0）。
        """
        cols = CHANGEOVER_LOG_COLUMNS

        if (
            plan_log is None or plan_log.empty or
            self.co_def_df is None or self.co_def_df.empty
        ):
            return pd.DataFrame(columns=cols)

        if 'changeover_id' in plan_log.columns:
            filt = plan_log[plan_log['changeover_id'].notna()]
        else:
            return pd.DataFrame(columns=cols)

        if filt.empty:
            return pd.DataFrame(columns=cols)

        summary = (
            filt.groupby(
                ['production_plan_date', 'location', 'line', 'changeover_id']
            )
            .size()
            .reset_index(name='count')
        )

        defs = self.co_def_df.drop_duplicates(
            ['changeover_id', 'line'], keep='first'
        )
        m = summary.merge(defs, on=['changeover_id', 'line'], how='left')

        for c in ['time', 'cost', 'mu_loss']:
            m[c] = m[c].fillna(0) * m['count']

        m = m.rename(columns={
            'production_plan_date': 'date',
            'changeover_id': 'changeover_type',
        })
        return m[cols]

    # ------------------------------------------------------------------
    # 已分配产能提取 —— 向量化
    # ------------------------------------------------------------------

    def extract_allocated_capacity(self, plan_log: pd.DataFrame) -> dict:
        """从生产计划提取已分配产能（小时）。对齐 extract_allocated_capacity_from_plan。

        逐行小时 = con_planned_qty/rate + 换产时间，按 (location,line,production_plan_date) 求和。
        速率：primary (mat,loc,line) 缺失则 1（对齐 _lookup_rate 在 3-key map 上回退失效→1）。
        """
        allocated = {}

        if plan_log is None or plan_log.empty:
            return allocated

        # 速率查找：直接用分配器 self.rate_map（3-key）merge，缺失 fillna(1)
        rate_df = self.rate_map.reset_index()
        g = plan_log.merge(
            rate_df.rename(columns={'prd_rate': 'rate'}),
            on=['material', 'location', 'line'], how='left',
        )
        g['rate'] = g['rate'].fillna(1.0)

        # 换产时间（按行）
        co = g.merge(
            self.co_def_df, on=['changeover_id', 'line'], how='left'
        )['time'].fillna(0)

        g['_hours'] = g['con_planned_qty'] / g['rate'] + co.to_numpy()

        agg = g.groupby(
            ['location', 'line', 'production_plan_date']
        )['_hours'].sum()

        for (loc, line, prod_date), v in agg.items():
            key = f"{loc}|{line}|{prod_date.strftime('%Y-%m-%d')}"
            allocated[key] = safe_float_conversion(v)

        return allocated

    # ------------------------------------------------------------------
    # 产线状态提取 —— 向量化聚合 + 小循环推断
    # ------------------------------------------------------------------

    def extract_line_states(self, plan_log: pd.DataFrame) -> dict:
        """从生产计划提取产线状态。对齐 extract_line_states_from_plan。

        末条状态：按 (line,simulation_date) 取 production_plan_date 末行；
        日末换产推断：聚合 (line,production_plan_date) 的已分配工时与产线产能比较。
        """
        if plan_log is None or plan_log.empty:
            return {}

        changeover_states = self._analyze_end_of_day_changeover(plan_log)

        last_rows = (
            plan_log.sort_values('production_plan_date')
            .groupby(['line', 'simulation_date']).tail(1)
        )

        line_states = {}
        for row in last_rows.itertuples(index=False):
            co_state = changeover_states.get(row.line)
            line_states[row.line] = self._build_line_state(row, co_state)

        return line_states

    def _build_line_state(self, last_prod, co_state):
        """构建单条产线状态。对齐 capacity_allocator._build_line_state。"""
        base_state = {
            'last_material': str(last_prod.material),
            'last_location': str(last_prod.location),
            'last_production_date': pd.Timestamp(
                last_prod.production_plan_date
            ).strftime('%Y-%m-%d'),
        }

        if co_state and co_state.get('last_activity') == 'changeover':
            base_state['last_activity'] = 'changeover'
            base_state['changeover_info'] = co_state['changeover_info']
        else:
            base_state['last_activity'] = 'production'
            base_state['changeover_info'] = None

        return base_state

    def _analyze_end_of_day_changeover(self, plan_log: pd.DataFrame) -> dict:
        """分析日末换产状态（向量化聚合）。对齐 capacity_allocator._analyze_end_of_day_changeover。

        把原逐行累加 qty/rate + co_time 拍平为 groupby 求和；
        remaining = 产线当日产能 - 已分配；满足 remaining>0.1 且 |remaining-1.0|<0.1
        时推断未完成换产。每条产线取最大 production_plan_date 的推断结果。
        """
        states = {}

        if self.cap_df is None or self.cap_df.empty:
            return states
        if self.rate_map is None or self.rate_map.empty:
            return states

        # 速率 + 换产时间（按行）
        rate_df = self.rate_map.reset_index()
        g = plan_log.merge(
            rate_df.rename(columns={'prd_rate': 'rate'}),
            on=['material', 'location', 'line'], how='left',
        )
        g['rate'] = g['rate'].fillna(1.0)
        g = g.merge(self.co_def_df, on=['changeover_id', 'line'], how='left')
        g['co_time'] = g['time'].fillna(0)
        g['alloc'] = g['con_planned_qty'] / g['rate'] + g['co_time']

        # 仅当日仿真
        sim_ts = pd.Timestamp(self.simulation_date)
        sg = g[g['simulation_date'] == sim_ts]
        if sg.empty:
            return states

        agg = (
            sg.groupby(['line', 'production_plan_date'])
            .agg(allocated=('alloc', 'sum'), last_material=('material', 'last'))
            .reset_index()
        )

        # 产线当日产能（跨 location 求和）
        cap_by_line = (
            self.cap_df.groupby(['line', 'date'])['capacity'].sum().reset_index()
            .rename(columns={'date': 'production_plan_date', 'capacity': 'line_cap'})
        )
        agg = agg.merge(
            cap_by_line, on=['line', 'production_plan_date'], how='left'
        )
        agg['line_cap'] = agg['line_cap'].fillna(0)

        agg['remaining'] = agg['line_cap'] - agg['allocated']
        mask = (
            (agg['remaining'] > 0.1) &
            ((agg['remaining'] - 1.0).abs() < 0.1) &
            agg['last_material'].notna()
        )
        triggered = agg[mask].sort_values('production_plan_date')

        for row in triggered.itertuples(index=False):
            states[row.line] = {
                'last_activity': 'changeover',
                'changeover_info': {
                    'changeover_id': 'INFERRED_INCOMPLETE',
                    'from_material': row.last_material,
                    'to_material': 'UNKNOWN_NEXT',
                    'total_time': 1.0,
                    'completed_time': row.remaining,
                    'remaining_time': 1.0 - row.remaining,
                },
            }

        return states

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def prepare(self):
        # 0) 数据加载（委托给 orchestrator）
        if self.orchestrator is not None:
            self.orchestrator.load_datas(self)

        # 0.5) 按 schema 将标识符列转为 str（对齐 production_runner 的 cast_identifiers_to_str）
        self._cast_datas_identifiers()

        # 1) 构建无约束日计划
        self.unconstrained_plan = self.build_unconstrained_plan()

        # 2) 分配器输入 + 跨天状态
        self._prepare_allocator_inputs()
        self._load_previous_states()

    def run(self):
        try:
            unconstrained_plan = self.validate_data(
                self.unconstrained_plan,
                name='unconstrained_plan',
                numeric_columns=['uncon_planned_qty', 'original_quantity'],
                required_columns=['material', 'location', 'line', 'planned_date'],
            )

            # 分配产能（支持跨天转产连续性和产能跟踪）
            plan_log, exceed_log = self.allocate_capacity(unconstrained_plan)

            # 仿真生产可靠性
            plan_log = self.simulate_reliability(plan_log)

            # 计算换产指标
            changeover_log = self.calc_changeover_metrics(plan_log)

            # 提取并保存当天产线状态供下一天使用（带跨天转产检测）
            current_line_states = self.extract_line_states(plan_log)
            if current_line_states and not self.skip_state_file_output:
                save_line_state(
                    self.output_dir, self.simulation_date, current_line_states
                )

            # 提取并保存当天分配的产能供后续仿真日期使用
            current_allocated_capacity = self.extract_allocated_capacity(plan_log)
            if current_allocated_capacity and not self.skip_state_file_output:
                save_allocated_capacity(
                    self.output_dir, self.simulation_date,
                    current_allocated_capacity
                )

            # 去重问题 → DataFrame
            issues = dedup_issues(self.issues)
            issues_df = pd.DataFrame(issues) if issues else pd.DataFrame()

            # 生成输出文件（通常仅文件模式需要写出）
            if not self.skip_file_output:
                base_output_file = os.path.join(self.output_dir, "Module4Output.xlsx")
                write_output(
                    plan_log, exceed_log, issues, changeover_log,
                    base_output_file, self.simulation_date,
                )

            # production_df：available_date >= simulation_date 的记录
            production_df = pd.DataFrame()
            if not plan_log.empty and 'available_date' in plan_log.columns:
                plan_log['available_date'] = pd.to_datetime(plan_log['available_date'])
                current_production = plan_log[
                    plan_log['available_date'] >= pd.Timestamp(self.simulation_date).normalize()
                ]
                if not current_production.empty:
                    production_df = current_production.copy()

            self._result = {
                'production_df': production_df,
                'exceed_log': (
                    exceed_log if isinstance(exceed_log, pd.DataFrame)
                    else pd.DataFrame(exceed_log) if exceed_log else pd.DataFrame()
                ),
                'issues_df': issues_df,
                'changeover_log': (
                    changeover_log if isinstance(changeover_log, pd.DataFrame)
                    else pd.DataFrame(changeover_log) if changeover_log else pd.DataFrame()
                ),
                'current_line_states': current_line_states,
                'current_allocated_capacity': current_allocated_capacity,
                'unconstrained_plan': unconstrained_plan,
            }
        except Exception:
            import traceback
            traceback.print_exc()
            self._empty_result()

    # ------------------------------------------------------------------
    # 私有工具方法
    # ------------------------------------------------------------------

    def _cast_datas_identifiers(self):
        """按 schema 将 self.datas 中声明为 'str' 的列强制转为 str dtype，
        并对 location 列做零填充归一化（对齐 production_runner 的
        ``module4.cast_identifiers_to_str``）。

        Excel 读入的标识符列常为 int64 / float64，与 M3 输出的 str 列 merge 时
        会因 dtype 不匹配而 ValueError；此外 mlcfg 的 location 可能为 '386' 而
        M3 为 '0386'（零填充），不归一化则 merge 空结果。此方法在 prepare() 入口
        处统一修正。
        """
        from .utils import cast_identifiers_to_str
        for sheet_name, col_schema in self.schema.items():
            df = self.datas.get(sheet_name)
            if df is None or df.empty:
                continue
            # 找出 schema 中声明为 'str' 的列名
            str_cols = [col for col, dtype in col_schema.items() if dtype == 'str']
            # cast_identifiers_to_str 会做 astype(str) + location 零填充
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
