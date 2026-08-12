"""Module4 的 pandas / polars 后端实现。

每个 backend 类实现相同的方法签名，供 ModuleFour 通过委托调用。
"""

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...utils.defaults import M6_RANDOM_SEED, DEFAULT_CHANGEOVER_TIME
from ...utils.normalization import normalize_material

logger = logging.getLogger("SupplyChainSimulation")


# ======================================================================
# 局部常量
# ======================================================================

UNCONSTRAINED_PLAN_COLUMNS: List[str] = [
    'material', 'location', 'line', 'planned_date',
    'uncon_planned_qty', 'simulation_date', 'original_quantity',
]

CHANGEOVER_LOG_COLUMNS: List[str] = [
    'date', 'location', 'line', 'changeover_type', 'count', 'time', 'cost', 'mu_loss',
]


# ======================================================================
# 局部函数（原 from .capacity_allocator / .plan_builder / src.utils / .utils import）
# ======================================================================

def _lookup_rate(
    rate_map,
    material: Any,
    location: Any,
    line: Any,
    default: float = 1.0
) -> float:
    """Look up production rate by location-aware key, with legacy fallback.

    Originally from capacity_allocator.py — inlined to avoid coupling to
    the old monolith.
    """
    for key in ((material, location, line), (material, line)):
        try:
            value = rate_map.get(key, None)
        except (KeyError, IndexError, TypeError):
            value = None

        if value is None:
            continue

        if isinstance(value, pd.Series):
            values = pd.to_numeric(value, errors='coerce').dropna().unique()
            if len(values) == 1:
                return float(values[0])
            if len(values) > 1:
                raise ValueError(
                    "Conflicting prd_rate values for rate_map key "
                    f"{key}: {values.tolist()}"
                )
            continue

        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        return float(value)

    return float(default)


def _get_changeover_time(
    from_material: str,
    to_material: str,
    co_mat,
    co_def: Dict[tuple, float],
    line: str
) -> float:
    """获取换产时间。duck-typed: co_mat 可为 pandas Series 或 dict。"""
    try:
        from_str = str(from_material)
        to_str = str(to_material)

        # dict 路径
        if isinstance(co_mat, dict):
            coid = co_mat.get((from_str, to_str))
            if coid is None:
                return 0
            coid = str(coid)
            return co_def.get((coid, line), 0)

        # pandas Series 路径
        if (from_str, to_str) not in co_mat.index:
            return 0

        coid = co_mat.loc[(from_str, to_str)]
        if isinstance(coid, pd.Series):
            coid = str(coid.iloc[0])
        else:
            coid = str(coid)

        return co_def.get((coid, line), 0)
    except Exception:
        return 0


def _select_first_batch(
    batches: List[Dict[str, Any]],
    remaining: set
) -> int:
    """选择第一个批次（最大原始数量）。"""
    return max(
        remaining,
        key=lambda i: batches[i].get(
            'original_quantity', batches[i]['uncon_planned_qty']
        )
    )


def _select_next_batch(
    batches: List[Dict[str, Any]],
    remaining: set,
    current_material: str,
    co_mat,
    co_def: Dict[tuple, float],
    line: str
) -> int:
    """选择下一个批次：优先最小换产时间，同时间用数量打破。"""
    min_cost = None
    candidates = []

    for idx in remaining:
        co_time = _get_changeover_time(
            current_material, batches[idx]['material'], co_mat, co_def, line
        )
        if min_cost is None or co_time < min_cost:
            min_cost = co_time
            candidates = [idx]
        elif co_time == min_cost:
            candidates.append(idx)

    if len(candidates) == 1:
        return candidates[0]

    return max(
        candidates,
        key=lambda i: batches[i].get(
            'original_quantity', batches[i]['uncon_planned_qty']
        )
    )


def optimal_changeover_sequence(
    batches: List[Dict[str, Any]],
    co_mat,
    co_def: Dict[tuple, float],
    line: str
) -> List[int]:
    """优化换产序列。

    首件按原始需求量最大选择；后续优先最小换产时间，
    若并列使用数量打破。

    Originally from plan_builder.py — inlined to avoid coupling to
    the old monolith. co_mat now duck-typed (Series or dict).
    """
    if not batches:
        return []

    remaining = set(range(len(batches)))

    first = _select_first_batch(batches, remaining)
    sequence = [first]
    remaining.remove(first)

    current_material = batches[first]['material']

    while remaining:
        next_idx = _select_next_batch(
            batches, remaining, current_material, co_mat, co_def, line
        )
        sequence.append(next_idx)
        remaining.remove(next_idx)
        current_material = batches[next_idx]['material']

    return sequence


def compute_planning_window(
    simulation_date: pd.Timestamp,
    ptf: int,
    lsk: int,
):
    """Return the inclusive planning window start/end pair.

    Originally from src.utils.date_helpers — inlined to avoid external import.
    """
    window_start = simulation_date + timedelta(days=ptf)
    window_end = simulation_date + timedelta(days=ptf + lsk - 1)
    return window_start, window_end


def safe_float_conversion(value: Any) -> float:
    """安全地将值转换为 float 类型，处理 numpy 数值以确保 JSON 序列化兼容。

    Originally from .utils — inlined to avoid coupling to sibling module.
    """
    if isinstance(value, (np.integer, np.int64)):
        return float(value)
    if isinstance(value, np.floating):
        return float(value)
    return float(value) if value else 0.0


def dedup_issues(issues: List[dict]) -> List[dict]:
    """去重校验问题记录。

    Originally from .utils — inlined to avoid coupling to sibling module.
    """
    if not issues:
        return issues

    df = pd.DataFrame(issues)
    df = df.drop_duplicates()
    return df.to_dict(orient='records')


def _round_up_to_batch_legacy(quantity: float, min_batch: Any, rounding_volume: Any) -> int:
    """复刻旧 M4 的批量取整：先截断配置值，再使用向上取整。"""
    base = max(quantity, int(min_batch))
    rv = int(rounding_volume)
    if base % rv == 0:
        return int(base)
    return int(np.ceil(base / rv) * rv)


# ======================================================================
# Pandas 后端 — 直接复用现有逻辑
# ======================================================================

class _PandasBackend:
    """将 ModuleFour 中原有 pandas 操作原样保留。"""

    def __init__(self, owner):
        self._o = owner
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

    @property
    def datas(self):
        return self._o.datas

    # ---- 步骤方法 ----

    def build_unconstrained_plan(self) -> pd.DataFrame:
        if 'M4_MaterialLocationLineCfg' not in self.datas:
            return pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS)

        mat_loc_line_config = self.datas['M4_MaterialLocationLineCfg'].copy()

        # 1) review day 标记 + 筛选
        days_since_start = (
            pd.Timestamp(self._o.simulation_date) - pd.Timestamp(self._o.simulation_start_date)
        ).days
        mat_loc_line_config['is_offset_review_day'] = mat_loc_line_config.apply(
            lambda r: self._o.is_offset_review_day(r, days_since_start), axis=1,
        )
        mat_loc_line_config = mat_loc_line_config[
            mat_loc_line_config['is_offset_review_day']
        ].drop(columns=['is_offset_review_day'])

        if mat_loc_line_config.empty:
            return pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS)

        # 2) 关联 net_demand
        net_demand = self._o.load_net_demand()
        if net_demand is None or net_demand.empty:
            return pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS)
        net_demand['material'] = net_demand['material'].astype('str')

        net_demand = net_demand.groupby(['material', 'location', 'requirement_date'])['quantity'].sum().reset_index()

        df = pd.merge(
            mat_loc_line_config, net_demand,
            how='inner', on=['material', 'location'],
        )
        if df.empty or 'requirement_date' not in df.columns:
            return pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS)

        # 3) 按 simulation_date 过滤
        df['requirement_date'] = pd.to_datetime(df['requirement_date']).dt.normalize()
        sim_date = pd.Timestamp(self._o.simulation_date).normalize()
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
            lambda row: _round_up_to_batch_legacy(
                row['original_quantity'], row['min_batch'], row['rv'],
            ),
            axis=1,
        )

        grouped = grouped.rename(columns={'delegate_line': 'line'})
        grouped['planned_date'] = sim_date
        grouped['simulation_date'] = sim_date

        # 标准化 material
        grouped['material'] = grouped['material'].apply(normalize_material).astype('string')

        return grouped[UNCONSTRAINED_PLAN_COLUMNS]

    # ---- 分配器输入预处理 + 跨天状态加载 ----

    def _prepare_allocator_inputs(self):
        mlcfg = self.datas['M4_MaterialLocationLineCfg'].copy()
        self.mlcfg = mlcfg

        # ChangeoverDefinition
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

        # LineCapacity
        cap_df = self.datas['M4_LineCapacity'].copy()
        cap_df['date'] = pd.to_datetime(cap_df['date'])
        self.cap_df = cap_df
        self.has_location = 'location' in cap_df.columns
        self.cap_map = self._build_capacity_map(cap_df)

        # rate_map Series
        rate_map = mlcfg.set_index(
            ['material', 'location', 'delegate_line']
        )['prd_rate']
        rate_map.index.set_names(['material', 'location', 'line'], inplace=True)
        self.rate_map = rate_map.sort_index()

        # mct_map（列名小写，对齐 ConfigReader 投影后的 db 列名）
        self.mct_map = mlcfg.set_index(['material', 'location'])['mct'].to_dict()

    def _build_capacity_map(self, cap_df: pd.DataFrame) -> dict:
        cap_df = cap_df.copy()
        cap_df['capacity'] = cap_df['capacity'].astype(float)
        if 'location' in cap_df.columns:
            return cap_df.set_index(['location', 'line', 'date'])['capacity'].to_dict()
        return cap_df.set_index(['line', 'date'])['capacity'].to_dict()

    def _reset_capacity_map(self):
        """每日 run() 开头重置 cap_map：allocate_capacity 会就地扣减 cap_map，
        一次性 prepare 后必须每日从原始 cap_df 重建，避免跨天产能污染。"""
        self.cap_map = self._build_capacity_map(self.cap_df)

    def _load_previous_states(self):
        self.previous_line_states = self._o.previous_line_states_override or {}
        self.previously_allocated = self._o.allocated_capacity_override or {}

    # ---- 产能分配 ----

    def allocate_capacity(self, uncon: pd.DataFrame) -> tuple:
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
                self._o.issues.append({
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
                self._o.issues.append({
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

    # ---- 生产可靠性仿真 ----

    def simulate_reliability(self, plan_log: pd.DataFrame) -> pd.DataFrame:
        pr_cfg = self.datas.get('M4_ProductionReliability')

        if (
            plan_log is None or plan_log.empty or
            'con_planned_qty' not in plan_log.columns or
            pr_cfg is None or pr_cfg.empty
        ):
            # 空表 / 无 pr 配置：补 produced_qty 列并保持 0 行（与 polars 后端一致，
            # 避免下游 extract_line_states / _analyze_end_of_day_changeover 在空产日
            # 因列缺失或行数异常而抛错）。
            if plan_log is None:
                return pd.DataFrame(columns=['produced_qty'])
            if plan_log.empty:
                plan_log['produced_qty'] = pd.Series(dtype='int64')
            else:
                plan_log['produced_qty'] = []
            return plan_log

        seed = self._o.config.get('RandomSeed', M6_RANDOM_SEED)
        rng = np.random.RandomState(seed)

        # 向量化 pr 查找：drop_duplicates keep='last'
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

    # ---- 换产指标 ----

    def calc_changeover_metrics(self, plan_log: pd.DataFrame) -> pd.DataFrame:
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

    # ---- 已分配产能提取 ----

    def extract_allocated_capacity(self, plan_log: pd.DataFrame) -> dict:
        allocated = {}

        if plan_log is None or plan_log.empty:
            return allocated

        # 速率查找：直接用分配器 self.rate_map（3-key）merge
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

    # ---- 产线状态提取 ----

    def extract_line_states(self, plan_log: pd.DataFrame) -> dict:
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
        sim_ts = pd.Timestamp(self._o.simulation_date)
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

    # ---- validate_data ----

    def validate_data(self, df, name='data', numeric_columns=None,
                      required_columns=None, strict=False):
        from ..module import Module
        return Module.validate_data(self._o, df, name=name,
                                    numeric_columns=numeric_columns,
                                    required_columns=required_columns,
                                    strict=strict)


# ======================================================================
# Polars 后端
# ======================================================================

class _PolarsBackend:
    """ModuleFour 的 polars 实现 — 方法签名与 _PandasBackend 一致。"""

    def __init__(self, owner):
        self._o = owner
        # 缓存转换后的 polars 数据（供 extract 步骤使用）
        self._mlcfg_pl = None
        self._co_def_df_pl = None
        self._cap_df_pl = None

        # 分配器输入 — 全部为 dict 形式（分配器点查用 dict，纯 Python 循环）
        self.mlcfg: Optional[pd.DataFrame] = None  # 保留 pandas 格式供 _alloc_batch 查询
        self.co_def_df: Optional[pd.DataFrame] = None  # 保留 pandas 格式供 extract
        self.co_mat_dict: dict = {}  # dict 形式的 co_mat（first-wins）
        self.co_def: dict = {}
        self.cap_df: Optional[pd.DataFrame] = None  # 保留 pandas 格式供 extract
        self.cap_map: dict = {}
        self.has_location: bool = False
        self.rate_map_dict: dict = {}  # dict 形式的 rate_map
        self.mct_map: dict = {}
        self.previous_line_states: dict = {}
        self.previously_allocated: dict = {}
        # _alloc_batch 行查询用 dict
        self._mlcfg_row_map: dict = {}

    @property
    def datas(self):
        return self._o.datas

    def _to_pl(self, df):
        """pandas → polars 惰性转换（对齐 M1 backends._to_pl）。"""
        if df is None:
            import polars as pl
            return pl.DataFrame()
        if isinstance(df, (type(None),)):
            import polars as pl
            return pl.DataFrame()
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df
        from ...utils.df_convert import pandas_to_polars
        return pandas_to_polars(df)

    # ---- 步骤方法 ----

    def build_unconstrained_plan(self):
        import polars as pl

        if 'M4_MaterialLocationLineCfg' not in self.datas:
            return pl.DataFrame(schema={
                c: pl.Utf8 for c in ['material', 'location', 'line', 'planned_date', 'simulation_date']
            }).with_columns([
                pl.lit(0).cast(pl.Int64).alias('uncon_planned_qty'),
                pl.lit(0).cast(pl.Int64).alias('original_quantity'),
            ])

        mat_loc_line_config = self._to_pl(self.datas['M4_MaterialLocationLineCfg'].copy())

        # 1) review day 标记 + 筛选（向量化 expr）
        days_since_start = (
            pd.Timestamp(self._o.simulation_date) - pd.Timestamp(self._o.simulation_start_date)
        ).days
        first_review_day = pl.col("day").cast(pl.Int64) - 1
        is_on_cycle = ((days_since_start - first_review_day) % pl.col("lsk").cast(pl.Int64)) == 0
        is_after_first = days_since_start >= first_review_day
        mat_loc_line_config = mat_loc_line_config.filter(
            is_on_cycle & is_after_first
        )

        if mat_loc_line_config.is_empty():
            return pl.DataFrame(schema={
                c: pl.Utf8 for c in ['material', 'location', 'line', 'planned_date', 'simulation_date']
            }).with_columns([
                pl.lit(0).cast(pl.Int64).alias('uncon_planned_qty'),
                pl.lit(0).cast(pl.Int64).alias('original_quantity'),
            ])

        # 2) 关联 net_demand
        net_demand = self._o.load_net_demand()
        if net_demand is None or net_demand.empty:
            return pl.DataFrame(schema={
                c: pl.Utf8 for c in ['material', 'location', 'line', 'planned_date', 'simulation_date']
            }).with_columns([
                pl.lit(0).cast(pl.Int64).alias('uncon_planned_qty'),
                pl.lit(0).cast(pl.Int64).alias('original_quantity'),
            ])
        net_demand_pl = self._to_pl(net_demand)
        net_demand_pl = net_demand_pl.with_columns(
            pl.col("material").cast(pl.Utf8)
        )

        net_demand_pl = net_demand_pl.group_by(["material", "location", "requirement_date"]).agg([
            pl.col("quantity").sum()
        ])

        df = mat_loc_line_config.join(
            net_demand_pl, on=["material", "location"], how="inner"
        )
        if df.is_empty() or "requirement_date" not in df.columns:
            return pl.DataFrame(schema={
                c: pl.Utf8 for c in ['material', 'location', 'line', 'planned_date', 'simulation_date']
            }).with_columns([
                pl.lit(0).cast(pl.Int64).alias('uncon_planned_qty'),
                pl.lit(0).cast(pl.Int64).alias('original_quantity'),
            ])

        # 3) 按 simulation_date 过滤
        sim_date = pd.Timestamp(self._o.simulation_date).normalize()
        df = df.with_columns(
            pl.col("requirement_date").cast(pl.Date).alias("requirement_date"),
        )
        df = df.filter(pl.col("requirement_date") == sim_date.date())
        if df.is_empty():
            return pl.DataFrame(schema={
                c: pl.Utf8 for c in ['material', 'location', 'line', 'planned_date', 'simulation_date']
            }).with_columns([
                pl.lit(0).cast(pl.Int64).alias('uncon_planned_qty'),
                pl.lit(0).cast(pl.Int64).alias('original_quantity'),
            ])

        # 4) 按 material/location/delegate_line 汇总，与 pandas / 原始端一致。
        grouped = df.group_by(["material", "location", "delegate_line"]).agg([
            pl.col("quantity").sum().alias("original_quantity"),
            pl.col("min_batch").first().alias("min_batch"),
            pl.col("rv").first().alias("rv"),
        ])

        # 向上取整（使用 map_elements 保 parity 与 pandas 逐行 apply 一致）
        def _round_up(row):
            return _round_up_to_batch_legacy(
                row['original_quantity'], row['min_batch'], row['rv'],
            )

        grouped = grouped.with_columns(
            pl.struct(["original_quantity", "min_batch", "rv"])
            .map_elements(_round_up, return_dtype=pl.Int64)
            .alias("uncon_planned_qty")
        )

        grouped = grouped.rename({"delegate_line": "line"})
        grouped = grouped.with_columns([
            pl.lit(sim_date.date()).alias("planned_date"),
            pl.lit(sim_date.date()).alias("simulation_date"),
        ])

        # 标准化 material
        grouped = grouped.with_columns(
            pl.col("material").map_elements(
                lambda m: normalize_material(str(m)),
                return_dtype=pl.Utf8,
            )
        )

        return grouped.select(UNCONSTRAINED_PLAN_COLUMNS)

    # ---- 分配器输入预处理 + 跨天状态加载 ----

    def _prepare_allocator_inputs(self):
        mlcfg = self.datas['M4_MaterialLocationLineCfg'].copy()
        self.mlcfg = mlcfg  # 保留 pandas（_alloc_batch 用 .iloc 查）

        # ChangeoverDefinition
        co_def_df = self.datas['M4_ChangeoverDefinition'].copy()
        co_def_df['changeover_id'] = co_def_df['changeover_id'].astype(str)
        self.co_def_df = co_def_df

        # co_def dict
        self.co_def = co_def_df.set_index(
            ['changeover_id', 'line']
        )['time'].to_dict()

        # co_mat dict（first-wins，保证 parity 与 pandas sort_index + iloc[0] 一致）
        co_mat_df = self.datas['M4_ChangeoverMatrix'].copy()
        co_mat_df['from_material'] = co_mat_df['from_material'].astype(str)
        co_mat_df['to_material'] = co_mat_df['to_material'].astype(str)
        co_mat_df['changeover_id'] = co_mat_df['changeover_id'].astype(str)
        # 按 (from_material, to_material) 排序后逐行写入 dict（首次出现胜出）
        co_mat_df = co_mat_df.sort_values(['from_material', 'to_material'])
        self.co_mat_dict = {}
        for from_m, to_m, coid in zip(
            co_mat_df['from_material'], co_mat_df['to_material'],
            co_mat_df['changeover_id']
        ):
            key = (from_m, to_m)
            if key not in self.co_mat_dict:
                self.co_mat_dict[key] = coid

        # LineCapacity
        cap_df = self.datas['M4_LineCapacity'].copy()
        cap_df['date'] = pd.to_datetime(cap_df['date'])
        self.cap_df = cap_df
        self.has_location = 'location' in cap_df.columns
        self.cap_map = self._build_capacity_map(cap_df)

        # rate_map dict（first-wins 对齐 pandas sort_index）
        mlcfg_sorted = mlcfg.sort_values(['material', 'location', 'delegate_line'])
        self.rate_map_dict = {}
        for mat, loc, line, rate in zip(
            mlcfg_sorted['material'].astype(str),
            mlcfg_sorted['location'].astype(str),
            mlcfg_sorted['delegate_line'].astype(str),
            mlcfg_sorted['prd_rate']
        ):
            key = (mat, loc, str(line))
            if key not in self.rate_map_dict:
                self.rate_map_dict[key] = float(rate)

        # 2-key fallback dict (material, line) → rate
        # Note: pandas _lookup_rate tries (material,location,line) then (material,line)
        # dict already supports this via .get()

        # mct_map dict（列名小写，对齐 ConfigReader 投影后的 db 列名）
        self.mct_map = mlcfg.set_index(['material', 'location'])['mct'].to_dict()

        # mlcfg row lookup dict（_alloc_batch 用，避免 pandas iloc）
        self._mlcfg_row_map = {}
        for mat, loc, row_data in zip(
            mlcfg['material'].astype(str),
            mlcfg['location'].astype(str),
            mlcfg.itertuples(index=False),
        ):
            key = (mat, loc)
            if key not in self._mlcfg_row_map:
                self._mlcfg_row_map[key] = row_data

    def _build_capacity_map(self, cap_df: pd.DataFrame) -> dict:
        cap_df = cap_df.copy()
        cap_df['capacity'] = cap_df['capacity'].astype(float)
        if 'location' in cap_df.columns:
            return cap_df.set_index(['location', 'line', 'date'])['capacity'].to_dict()
        return cap_df.set_index(['line', 'date'])['capacity'].to_dict()

    def _reset_capacity_map(self):
        """每日 run() 开头重置 cap_map：allocate_capacity 会就地扣减 cap_map，
        一次性 prepare 后必须每日从原始 cap_df 重建，避免跨天产能污染。"""
        self.cap_map = self._build_capacity_map(self.cap_df)

    def _load_previous_states(self):
        self.previous_line_states = self._o.previous_line_states_override or {}
        self.previously_allocated = self._o.allocated_capacity_override or {}

    # ---- 产能分配（polars 边界 + dict 点查 + Python 循环核心） ----

    def allocate_capacity(self, uncon):
        import polars as pl

        plans_log = []
        exceed_log = []

        if uncon is None or (isinstance(uncon, pl.DataFrame) and uncon.is_empty()) or \
           (isinstance(uncon, pd.DataFrame) and uncon.empty):
            return pl.DataFrame(), pl.DataFrame()

        # polars 模式：uncon 是 polars DataFrame
        uncon_pl = self._to_pl(uncon) if isinstance(uncon, pd.DataFrame) else uncon

        uncon_pl = uncon_pl.sort(
            ["line", "simulation_date", "planned_date", "material"]
        )

        # 按 (line, simulation_date) 分组
        for sub_df in uncon_pl.partition_by(["line", "simulation_date"], as_dict=False):
            line = sub_df["line"][0]
            sim_date_val = sub_df["simulation_date"][0]

            batch_list = sub_df.to_dicts()

            if len(batch_list) > 1:
                sequence = optimal_changeover_sequence(
                    batch_list, self.co_mat_dict, self.co_def, str(line)
                )
                batch_list = [batch_list[i] for i in sequence]

            state = self._alloc_init_line_state(str(line))
            plans = []
            exceeds = []

            for idx, batch in enumerate(batch_list):
                batch_plans, batch_exceed = self._alloc_batch(
                    str(line), sim_date_val, batch, idx, state
                )
                plans.extend(batch_plans)
                if batch_exceed:
                    exceeds.append(batch_exceed)
                state['prev_mat'] = batch['material']

            plans_log.extend(plans)
            exceed_log.extend(exceeds)

        return pl.DataFrame(plans_log) if plans_log else pl.DataFrame(), \
               pl.DataFrame(exceed_log) if exceed_log else pl.DataFrame()

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
        material = str(batch['material'])
        location = str(batch['location'])

        config_row = self._mlcfg_row_map.get((material, location))
        if config_row is None:
            # fallback: pandas 查询
            config_row = self.mlcfg[
                (self.mlcfg['material'] == material) &
                (self.mlcfg['location'] == location)
            ].iloc[0]

        lsk = int(config_row.lsk)
        ptf = int(config_row.ptf)

        # simulation_date 可能是 datetime.date 或 pd.Timestamp
        sim_date_ts = pd.Timestamp(batch['simulation_date']) if not isinstance(batch['simulation_date'], pd.Timestamp) else batch['simulation_date']
        window_start, window_end = compute_planning_window(sim_date_ts, ptf, lsk)

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
        """dict 版 changeover lookup（对齐 pandas 版逻辑，first-wins dict）。"""
        try:
            from_str = str(from_mat)
            to_str = str(to_mat)

            key = (from_str, to_str)
            if key not in self.co_mat_dict:
                coid = f"MISSING_CO_{from_str}_to_{to_str}"
                self._o.issues.append({
                    'sheet': 'M4_ChangeoverMatrix',
                    'row': '',
                    'issue': (
                        f"缺失物料切换定义: {from_str} → {to_str}，"
                        f"产线 {line}。已使用默认时间。"
                    )
                })
                return coid, DEFAULT_CHANGEOVER_TIME

            coid = str(self.co_mat_dict[key])

            if (coid, line) not in self.co_def:
                self._o.issues.append({
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

        # 用 Python 日期循环替代 pd.date_range
        from datetime import date, timedelta as _td
        if isinstance(window_start, pd.Timestamp):
            ws = window_start.date()
            we = window_end.date()
        elif hasattr(window_start, 'date'):
            ws = window_start.date()
            we = window_end.date()
        else:
            ws = window_start
            we = window_end

        day_dt = ws
        while day_dt <= we:
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
            day_dt += _td(days=1)

        exceed = None
        if prod_remain > 0:
            exceed = {
                'material': batch['material'],
                'location': location,
                'line': line,
                'simulation_date': sim_date,
                'production_plan_date': we,
                'unmet_uncon_planned_qty': prod_remain,
            }

        return plans, exceed

    def _alloc_day(
        self, line, sim_date, batch, location, material,
        day_dt, prod_remain, co_remain, coid_to_log, is_first_co_day
    ):
        """与 pandas 版逻辑完全一致，仅 cap_key 用 datetime.date 而非 pd.Timestamp。"""
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

        rate = self.rate_map_dict.get((material, location, line),
                    self.rate_map_dict.get((material, line), 1.0))
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
        # day_dt 是 datetime.date，cap_map 的 key 也是 datetime.date（从 _build_capacity_map）
        # 但 _build_capacity_map 用 pandas df → key 含 pd.Timestamp
        # 需确保 key 类型一致：把 day_dt 转为 pd.Timestamp
        day_ts = pd.Timestamp(day_dt)
        if self.has_location:
            return (location, line, day_ts)
        return (line, day_ts)

    def _alloc_adjust_prev(self, current_cap, location, line, day_dt):
        if not self.previously_allocated:
            return current_cap
        day_ts = pd.Timestamp(day_dt)
        key = f"{location}|{line}|{day_ts.strftime('%Y-%m-%d')}"
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

    # ---- 生产可靠性仿真 ----

    def simulate_reliability(self, plan_log):
        import polars as pl

        pr_cfg = self.datas.get('M4_ProductionReliability')

        if (
            plan_log is None or
            (isinstance(plan_log, pl.DataFrame) and plan_log.is_empty()) or
            'con_planned_qty' not in plan_log.columns or
            pr_cfg is None or pr_cfg.empty
        ):
            if isinstance(plan_log, pl.DataFrame):
                # 空表：仅补 produced_qty 列并保持 0 行。
                # ⚠️ 不能用 pl.lit(None) —— 它是标量，在 0 行表上会被广播成 1 行，
                # 导致 is_empty() 失效（空产日 plan_log 为空），进而 extract_line_states
                # 误判非空并进入 _analyze_end_of_day_changeover，在 join(material/... )
                # 时抛 ColumnNotFoundError。pandas 版 `df['produced_qty']=[]` 不会复活空表，
                # 此处显式构造 0 行 schema 以对齐该行为。
                if plan_log.height == 0:
                    return pl.DataFrame(
                        schema={**plan_log.schema, "produced_qty": pl.Int64}
                    )
                return plan_log.with_columns(pl.lit(None).cast(pl.Int64).alias("produced_qty"))
            return plan_log

        seed = self._o.config.get('RandomSeed', M6_RANDOM_SEED)
        rng = np.random.RandomState(seed)

        pr_lookup_pl = self._to_pl(
            pr_cfg[['location', 'line', 'pr']]
            .drop_duplicates(['location', 'line'], keep='last')
        )
        merged = plan_log.join(pr_lookup_pl, on=["location", "line"], how="left")
        p = merged["pr"].fill_null(1.0).to_numpy()
        n = merged["con_planned_qty"].cast(pl.Int64).to_numpy()

        # 逐行抽样（不排序，保 RNG 序列与 pandas 版一致）
        produced = [rng.binomial(int(ni), float(pi)) for ni, pi in zip(n, p)]
        merged = merged.with_columns(pl.Series("produced_qty", produced))
        return merged

    # ---- 换产指标 ----

    def calc_changeover_metrics(self, plan_log):
        import polars as pl

        cols = CHANGEOVER_LOG_COLUMNS

        if (
            plan_log is None or
            (isinstance(plan_log, pl.DataFrame) and plan_log.is_empty()) or
            self.co_def_df is None or self.co_def_df.empty
        ):
            return pl.DataFrame(schema={c: pl.Utf8 for c in cols})

        if "changeover_id" in plan_log.columns:
            filt = plan_log.filter(pl.col("changeover_id").is_not_null())
        else:
            return pl.DataFrame(schema={c: pl.Utf8 for c in cols})

        if filt.is_empty():
            return pl.DataFrame(schema={c: pl.Utf8 for c in cols})

        summary = filt.group_by(
            ["production_plan_date", "location", "line", "changeover_id"]
        ).agg(
            pl.len().alias("count")
        )

        co_def_pl = self._to_pl(
            self.co_def_df.drop_duplicates(["changeover_id", "line"], keep="first")
        )
        m = summary.join(co_def_pl, on=["changeover_id", "line"], how="left")

        for c in ["time", "cost", "mu_loss"]:
            m = m.with_columns(
                (pl.col(c).fill_null(0) * pl.col("count")).alias(c)
            )

        m = m.rename({
            "production_plan_date": "date",
            "changeover_id": "changeover_type",
        })
        return m.select(cols)

    # ---- 已分配产能提取 ----

    def extract_allocated_capacity(self, plan_log):
        import polars as pl

        allocated = {}

        if plan_log is None or (isinstance(plan_log, pl.DataFrame) and plan_log.is_empty()):
            return allocated

        # 速率查找：rate_map 是 dict → 转 polars DataFrame 做 join
        rate_rows = [
            {"material": k[0], "location": k[1], "line": k[2], "rate": v}
            for k, v in self.rate_map_dict.items()
        ]
        rate_pl = pl.DataFrame(rate_rows)

        g = plan_log.join(rate_pl, on=["material", "location", "line"], how="left")
        g = g.with_columns(pl.col("rate").fill_null(1.0))

        # 换产时间
        co_def_pl = self._to_pl(self.co_def_df)
        g = g.join(co_def_pl, on=["changeover_id", "line"], how="left")
        g = g.with_columns(pl.col("time").fill_null(0).alias("co_time"))
        g = g.with_columns(
            (pl.col("con_planned_qty").cast(pl.Float64) / pl.col("rate") + pl.col("co_time")).alias("alloc")
        )

        agg = g.group_by(["location", "line", "production_plan_date"]).agg(
            pl.col("alloc").sum()
        )

        for row in agg.iter_rows(named=True):
            loc = row["location"]
            line = row["line"]
            prod_date = row["production_plan_date"]
            key = f"{loc}|{line}|{pd.Timestamp(prod_date).strftime('%Y-%m-%d')}"
            allocated[key] = safe_float_conversion(row["alloc"])

        return allocated

    # ---- 产线状态提取 ----

    def extract_line_states(self, plan_log):
        import polars as pl

        if plan_log is None or (isinstance(plan_log, pl.DataFrame) and plan_log.is_empty()):
            return {}

        changeover_states = self._analyze_end_of_day_changeover(plan_log)

        last_rows = plan_log.sort("production_plan_date").group_by(
            ["line", "simulation_date"]
        ).agg(pl.all().last())

        line_states = {}
        for row in last_rows.iter_rows(named=True):
            co_state = changeover_states.get(str(row["line"]))
            line_states[str(row["line"])] = self._build_line_state(row, co_state)

        return line_states

    def _build_line_state(self, last_prod, co_state):
        base_state = {
            'last_material': str(last_prod.get('material', '')),
            'last_location': str(last_prod.get('location', '')),
            'last_production_date': pd.Timestamp(
                last_prod.get('production_plan_date', '')
            ).strftime('%Y-%m-%d'),
        }

        if co_state and co_state.get('last_activity') == 'changeover':
            base_state['last_activity'] = 'changeover'
            base_state['changeover_info'] = co_state['changeover_info']
        else:
            base_state['last_activity'] = 'production'
            base_state['changeover_info'] = None

        return base_state

    def _analyze_end_of_day_changeover(self, plan_log):
        import polars as pl

        states = {}

        if self.cap_df is None or self.cap_df.empty:
            return states
        if not self.rate_map_dict:
            return states

        # 速率 + 换产时间
        rate_rows = [
            {"material": k[0], "location": k[1], "line": k[2], "rate": v}
            for k, v in self.rate_map_dict.items()
        ]
        rate_pl = pl.DataFrame(rate_rows)

        g = plan_log.join(rate_pl, on=["material", "location", "line"], how="left")
        g = g.with_columns(pl.col("rate").fill_null(1.0))

        co_def_pl = self._to_pl(self.co_def_df)
        g = g.join(co_def_pl, on=["changeover_id", "line"], how="left")
        g = g.with_columns(
            pl.col("time").fill_null(0).alias("co_time")
        )
        g = g.with_columns(
            (pl.col("con_planned_qty").cast(pl.Float64) / pl.col("rate") + pl.col("co_time")).alias("alloc")
        )

        # 仅当日仿真
        sim_ts = pd.Timestamp(self._o.simulation_date)
        sg = g.filter(
            pl.col("simulation_date").cast(pl.Date) == sim_ts.date()
        )
        if sg.is_empty():
            return states

        agg = sg.group_by(["line", "production_plan_date"]).agg([
            pl.col("alloc").sum().alias("allocated"),
            pl.col("material").last().alias("last_material"),
        ])

        # 产线当日产能（跨 location 求和）
        cap_pl = self._to_pl(self.cap_df)
        cap_by_line = cap_pl.group_by(["line", "date"]).agg(
            pl.col("capacity").sum().alias("line_cap")
        ).rename({"date": "production_plan_date"})
        # plan_log 的 production_plan_date 为 Date（来自 datetime.date），
        # cap_df 的 date 经 _to_pl 为 Datetime —— 统一为 Date 再 join。
        cap_by_line = cap_by_line.with_columns(
            pl.col("production_plan_date").cast(pl.Date)
        )
        agg = agg.join(cap_by_line, on=["line", "production_plan_date"], how="left")
        agg = agg.with_columns(pl.col("line_cap").fill_null(0))

        agg = agg.with_columns(
            (pl.col("line_cap") - pl.col("allocated")).alias("remaining")
        )
        triggered = agg.filter(
            (pl.col("remaining") > 0.1)
            & ((pl.col("remaining") - 1.0).abs() < 0.1)
            & pl.col("last_material").is_not_null()
        ).sort("production_plan_date")

        for row in triggered.iter_rows(named=True):
            states[str(row["line"])] = {
                'last_activity': 'changeover',
                'changeover_info': {
                    'changeover_id': 'INFERRED_INCOMPLETE',
                    'from_material': str(row["last_material"]),
                    'to_material': 'UNKNOWN_NEXT',
                    'total_time': 1.0,
                    'completed_time': float(row["remaining"]),
                    'remaining_time': 1.0 - float(row["remaining"]),
                },
            }

        return states

    # ---- validate_data ----

    def validate_data(self, df, name='data', numeric_columns=None,
                      required_columns=None, strict=False):
        """Polars 版数值校验（对齐 M1 _PolarsBackend.validate_data）。"""
        import polars as pl

        if df.is_empty():
            return df

        if required_columns:
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                raise ValueError(f"validate_data({name}): 缺失必选列 {missing}")

        if numeric_columns is not None:
            cols_to_check = [c for c in numeric_columns if c in df.columns]
        else:
            cols_to_check = [c for c, t in zip(df.columns, df.dtypes) if t.is_numeric()]

        if not cols_to_check:
            return df

        issues = {}
        for col in cols_to_check:
            series = df[col]
            null_count = series.null_count()
            non_null = series.drop_nulls()
            if non_null.dtype.is_float():
                inf_count = int((non_null.is_infinite() | non_null.is_nan()).sum())
            else:
                inf_count = 0
            bad_count = null_count + inf_count
            if bad_count > 0:
                if strict:
                    raise ValueError(
                        f"validate_data({name}): 列 '{col}' 含 {bad_count} 个异常值"
                    )
                df = df.with_columns(
                    pl.col(col).fill_nan(0).fill_null(0),
                )
                issues[col] = bad_count

        if issues:
            logger.warning(
                "validate_data(%s): %d 列存在异常值 %s", name, len(issues), issues,
            )

        return df
