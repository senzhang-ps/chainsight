"""
产能分配模块

负责集中式产能分配、换产处理和生产仿真。
"""

from datetime import timedelta
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np

from .constants import DEFAULT_CHANGEOVER_TIME
from .utils import compute_planning_window, safe_float_conversion
from .plan_builder import optimal_changeover_sequence

# 尝试导入 DuckDB 优化实现
try:
    from .duckdb_batch_calculator import (
        simulate_production_batch_duckdb,
        is_duckdb_available,
    )
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


def centralized_capacity_allocation_with_changeover(
    uncon: pd.DataFrame,
    cap_df: pd.DataFrame,
    rate_map: pd.Series,
    co_mat: pd.Series,
    co_def: Dict[tuple, float],
    mlcfg: pd.DataFrame,
    previous_line_states: Optional[Dict[str, Any]] = None,
    simulation_date: Optional[pd.Timestamp] = None,
    previously_allocated_capacity: Optional[Dict[str, float]] = None,
    issues: Optional[List[Dict[str, Any]]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """集中式产能分配（含跨天换产连续性）。

    参数：
        uncon: 无约束计划DataFrame
        cap_df: 产能DataFrame
        rate_map: 产率映射
        co_mat: 换产矩阵
        co_def: 换产定义
        mlcfg: 物料地点产线配置
        previous_line_states: 前一日产线状态
        simulation_date: 当前仿真日期
        previously_allocated_capacity: 历史已分配产能
        issues: 问题列表

    返回：
        Tuple[pd.DataFrame, pd.DataFrame]: (计划日志, 超额日志)
    """
    if issues is None:
        issues = []

    allocator = CapacityAllocator(
        cap_df=cap_df,
        rate_map=rate_map,
        co_mat=co_mat,
        co_def=co_def,
        mlcfg=mlcfg,
        previous_line_states=previous_line_states,
        previously_allocated_capacity=previously_allocated_capacity,
        issues=issues
    )

    return allocator.allocate(uncon)


class CapacityAllocator:
    """产能分配器类。

    负责处理产能分配的核心逻辑，包括换产处理和跨天连续性。
    """

    def __init__(
        self,
        cap_df: pd.DataFrame,
        rate_map: pd.Series,
        co_mat: pd.Series,
        co_def: Dict[tuple, float],
        mlcfg: pd.DataFrame,
        previous_line_states: Optional[Dict[str, Any]] = None,
        previously_allocated_capacity: Optional[Dict[str, float]] = None,
        issues: Optional[List[Dict[str, Any]]] = None
    ):
        """初始化分配器。

        参数：
            cap_df: 产能DataFrame
            rate_map: 产率映射
            co_mat: 换产矩阵
            co_def: 换产定义
            mlcfg: 物料地点产线配置
            previous_line_states: 前一日产线状态
            previously_allocated_capacity: 历史已分配产能
            issues: 问题列表
        """
        self.rate_map = rate_map
        self.co_mat = co_mat
        self.co_def = co_def
        self.mlcfg = mlcfg
        self.previous_line_states = previous_line_states or {}
        self.previously_allocated = previously_allocated_capacity or {}
        self.issues = issues if issues is not None else []

        self.mct_map = mlcfg.set_index(
            ['material', 'location']
        )['MCT'].to_dict()

        self.cap_map = self._build_capacity_map(cap_df)
        self.has_location = 'location' in cap_df.columns

    def _build_capacity_map(self, cap_df: pd.DataFrame) -> Dict:
        """构建产能映射。

        参数：
            cap_df: 产能DataFrame

        返回：
            Dict: 产能映射
        """
        cap_df = cap_df.copy()
        cap_df['capacity'] = cap_df['capacity'].astype(float)

        if 'location' in cap_df.columns:
            return cap_df.set_index(
                ['location', 'line', 'date']
            )['capacity'].to_dict()

        return cap_df.set_index(['line', 'date'])['capacity'].to_dict()

    def allocate(
        self,
        uncon: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """执行产能分配。

        参数：
            uncon: 无约束计划

        返回：
            Tuple[pd.DataFrame, pd.DataFrame]: (计划日志, 超额日志)
        """
        plans_log = []
        exceed_log = []

        if uncon.empty:
            return pd.DataFrame(plans_log), pd.DataFrame(exceed_log)

        uncon = uncon.sort_values(
            ['line', 'simulation_date', 'planned_date', 'material']
        ).reset_index(drop=True)

        for (line, sim_date), group in uncon.groupby(
            ['line', 'simulation_date']
        ):
            line_plans, line_exceed = self._allocate_line_group(
                line, sim_date, group
            )
            plans_log.extend(line_plans)
            exceed_log.extend(line_exceed)

        return pd.DataFrame(plans_log), pd.DataFrame(exceed_log)

    def _allocate_line_group(
        self,
        line: str,
        sim_date: pd.Timestamp,
        group: pd.DataFrame
    ) -> Tuple[List[Dict], List[Dict]]:
        """为单个产线分组分配产能。

        参数：
            line: 产线
            sim_date: 仿真日期
            group: 计划分组

        返回：
            Tuple[List[Dict], List[Dict]]: (计划列表, 超额列表)
        """
        batch_list = group.to_dict(orient='records')

        if len(batch_list) > 1:
            sequence = optimal_changeover_sequence(
                batch_list, self.co_mat, self.co_def, line
            )
            batch_list = [batch_list[i] for i in sequence]

        state = self._init_line_state(line)
        plans = []
        exceeds = []

        for idx, batch in enumerate(batch_list):
            batch_plans, batch_exceed = self._allocate_batch(
                line, sim_date, batch, idx, state
            )
            plans.extend(batch_plans)
            if batch_exceed:
                exceeds.append(batch_exceed)
            state['prev_mat'] = batch['material']

        return plans, exceeds

    def _init_line_state(self, line: str) -> Dict[str, Any]:
        """初始化产线状态。

        参数：
            line: 产线

        返回：
            Dict[str, Any]: 状态字典
        """
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
                    'to_material',
                    state['prev_mat']
                )
            else:
                state['prev_mat'] = changeover_info.get(
                    'to_material',
                    state['prev_mat']
                )

        return state

    def _allocate_batch(
        self,
        line: str,
        sim_date: pd.Timestamp,
        batch: Dict[str, Any],
        batch_idx: int,
        state: Dict[str, Any]
    ) -> Tuple[List[Dict], Optional[Dict]]:
        """为单个批次分配产能。

        参数：
            line: 产线
            sim_date: 仿真日期
            batch: 批次信息
            batch_idx: 批次索引
            state: 产线状态

        返回：
            Tuple[List[Dict], Optional[Dict]]: (计划列表, 超额记录)
        """
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

        changeover = self._calculate_changeover(
            batch_idx, material, state, line
        )

        return self._allocate_to_horizon(
            line, sim_date, batch, location, material,
            window_start, window_end, changeover
        )

    def _calculate_changeover(
        self,
        batch_idx: int,
        material: str,
        state: Dict[str, Any],
        line: str
    ) -> Dict[str, Any]:
        """计算换产信息。

        参数：
            batch_idx: 批次索引
            material: 物料
            state: 产线状态
            line: 产线

        返回：
            Dict[str, Any]: 换产信息
        """
        if batch_idx == 0 and state['has_incomplete_changeover']:
            return {
                'remain': state['initial_co_remain'],
                'coid': state['initial_coid'],
                'is_first': True,
            }

        prev_mat = state['prev_mat']
        if prev_mat is not None and prev_mat != material:
            coid, co_time = self._lookup_changeover(
                prev_mat, material, line
            )
            return {
                'remain': co_time,
                'coid': coid,
                'is_first': True,
            }

        return {'remain': 0, 'coid': None, 'is_first': False}

    def _lookup_changeover(
        self,
        from_mat: str,
        to_mat: str,
        line: str
    ) -> Tuple[Optional[str], float]:
        """查找换产定义。

        参数：
            from_mat: 源物料
            to_mat: 目标物料
            line: 产线

        返回：
            Tuple[Optional[str], float]: (换产ID, 换产时间)
        """
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

        except Exception as e:
            print(f"换产查找异常: {e}")
            return None, 0

    def _allocate_to_horizon(
        self,
        line: str,
        sim_date: pd.Timestamp,
        batch: Dict[str, Any],
        location: str,
        material: str,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        changeover: Dict[str, Any]
    ) -> Tuple[List[Dict], Optional[Dict]]:
        """在计划窗口内分配产能。

        参数：
            line: 产线
            sim_date: 仿真日期
            batch: 批次信息
            location: 地点
            material: 物料
            window_start: 窗口起始
            window_end: 窗口结束
            changeover: 换产信息

        返回：
            Tuple[List[Dict], Optional[Dict]]: (计划列表, 超额记录)
        """
        plans = []
        prod_remain = batch['uncon_planned_qty']
        co_remain = changeover['remain']
        coid_to_log = changeover['coid']
        is_first_co_day = changeover['is_first']

        horizon_days = pd.date_range(window_start, window_end)

        for day_dt in horizon_days:
            result = self._allocate_day(
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

    def _allocate_day(
        self,
        line: str,
        sim_date: pd.Timestamp,
        batch: Dict[str, Any],
        location: str,
        material: str,
        day_dt: pd.Timestamp,
        prod_remain: int,
        co_remain: float,
        coid_to_log: Optional[str],
        is_first_co_day: bool
    ) -> Dict[str, Any]:
        """在单日分配产能。

        参数：
            line: 产线
            sim_date: 仿真日期
            batch: 批次信息
            location: 地点
            material: 物料
            day_dt: 日期
            prod_remain: 剩余生产量
            co_remain: 剩余换产时间
            coid_to_log: 换产ID
            is_first_co_day: 是否首个换产日

        返回：
            Dict[str, Any]: 分配结果
        """
        cap_key = self._get_capacity_key(location, line, day_dt)
        current_cap = self.cap_map.get(cap_key, 0)
        today_cap = self._adjust_for_previous_allocation(
            current_cap, location, line, day_dt
        )

        co_used, co_remain, today_cap, changeover_done = (
            self._consume_changeover(co_remain, today_cap)
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

        rate = float(self.rate_map.get((material, line), 1))
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

    def _get_capacity_key(
        self,
        location: str,
        line: str,
        day_dt: pd.Timestamp
    ) -> tuple:
        """获取产能映射键。

        参数：
            location: 地点
            line: 产线
            day_dt: 日期

        返回：
            tuple: 产能键
        """
        if self.has_location:
            return (location, line, day_dt)
        return (line, day_dt)

    def _adjust_for_previous_allocation(
        self,
        current_cap: float,
        location: str,
        line: str,
        day_dt: pd.Timestamp
    ) -> float:
        """调整已分配产能。

        参数：
            current_cap: 当前产能
            location: 地点
            line: 产线
            day_dt: 日期

        返回：
            float: 调整后的产能
        """
        if not self.previously_allocated:
            return current_cap

        key = f"{location}|{line}|{day_dt.strftime('%Y-%m-%d')}"
        prev_used = self.previously_allocated.get(key, 0)

        return max(0, current_cap - prev_used)

    def _consume_changeover(
        self,
        co_remain: float,
        today_cap: float
    ) -> Tuple[float, float, float, bool]:
        """消耗换产时间。

        参数：
            co_remain: 剩余换产时间
            today_cap: 今日产能

        返回：
            Tuple[float, float, float, bool]:
                (使用的换产时间, 剩余换产, 剩余产能, 是否完成)
        """
        if co_remain <= 0:
            return 0, 0, today_cap, False

        co_used = min(today_cap, co_remain)
        co_remain -= co_used
        today_cap -= co_used
        completed = (co_remain == 0)

        return co_used, co_remain, today_cap, completed


def extract_allocated_capacity_from_plan(
    plan_df: pd.DataFrame,
    rate_map: Dict[tuple, float],
    changeover_def: Optional[Dict[tuple, float]] = None
) -> Dict[str, float]:
    """从生产计划提取已分配产能信息。

    参数：
        plan_df: 生产计划DataFrame
        rate_map: 产率映射
        changeover_def: 换产定义

    返回：
        Dict[str, float]: 已分配产能字典
    """
    allocated = {}

    if plan_df.empty:
        return allocated

    grouped = plan_df.groupby(
        ['location', 'line', 'production_plan_date']
    )

    for (loc, line, prod_date), group in grouped:
        total_hours = _calculate_group_hours(
            group, rate_map, changeover_def, line
        )
        total_hours = safe_float_conversion(total_hours)

        key = f"{loc}|{line}|{prod_date.strftime('%Y-%m-%d')}"
        allocated[key] = total_hours

    return allocated


def _calculate_group_hours(
    group: pd.DataFrame,
    rate_map: Dict[tuple, float],
    changeover_def: Optional[Dict[tuple, float]],
    line: str
) -> float:
    """计算分组的总小时数。

    参数：
        group: 分组DataFrame
        rate_map: 产率映射
        changeover_def: 换产定义
        line: 产线

    返回：
        float: 总小时数
    """
    total = 0

    # 优化：使用 itertuples() 替代 iterrows()，速度快 10-100 倍
    for row in group.itertuples(index=False):
        material = row.material
        quantity = row.con_planned_qty
        changeover_id = getattr(row, 'changeover_id', None)

        rate = rate_map.get((material, line), 1)
        total += quantity / rate if rate else 0

        if changeover_id and changeover_def and pd.notna(changeover_id):
            total += changeover_def.get((changeover_id, line), 0)

    return total


def validate_capacity_allocation(
    plan_log: pd.DataFrame,
    previously_allocated: Dict[str, float],
    simulation_date: pd.Timestamp,
    rate_map: Dict[tuple, float],
    changeover_def: Optional[Dict[tuple, float]] = None
) -> List[Dict[str, Any]]:
    """校验产能分配。

    参数：
        plan_log: 当前生产计划
        previously_allocated: 历史已分配产能
        simulation_date: 当前仿真日期
        rate_map: 产率映射
        changeover_def: 换产定义

    返回：
        List[Dict[str, Any]]: 校验记录列表
    """
    issues = []

    if plan_log.empty or not previously_allocated:
        return issues

    grouped = plan_log.groupby(
        ['location', 'line', 'production_plan_date']
    )

    for (loc, line, prod_date), group in grouped:
        current_hours = _calculate_group_hours(
            group, rate_map, changeover_def, line
        )

        key = f"{loc}|{line}|{prod_date.strftime('%Y-%m-%d')}"
        prev_hours = previously_allocated.get(key, 0)

        if prev_hours > 0:
            issues.append({
                'type': 'capacity_validation',
                'location': loc,
                'line': line,
                'production_plan_date': prod_date.strftime('%Y-%m-%d'),
                'simulation_date': simulation_date.strftime('%Y-%m-%d'),
                'previously_allocated_hours': prev_hours,
                'currently_allocated_hours': current_hours,
                'total_allocated_hours': prev_hours + current_hours,
                'message': (
                    f"产能校验: {loc}/{line} 于 "
                    f"{prod_date.strftime('%Y-%m-%d')} - "
                    f"历史: {prev_hours:.2f}h, "
                    f"当前: {current_hours:.2f}h, "
                    f"总计: {prev_hours + current_hours:.2f}h"
                )
            })

    return issues


def extract_line_states_from_plan(
    plan_df: pd.DataFrame,
    cap_df: Optional[pd.DataFrame] = None,
    co_def: Optional[Dict[tuple, float]] = None,
    simulation_date: Optional[pd.Timestamp] = None,
    rate_map: Optional[Dict[tuple, float]] = None
) -> Dict[str, Any]:
    """从生产计划提取产线状态。

    参数：
        plan_df: 生产计划
        cap_df: 产能数据
        co_def: 换产定义
        simulation_date: 仿真日期
        rate_map: 产率映射

    返回：
        Dict[str, Any]: 产线状态字典
    """
    if plan_df.empty:
        return {}

    changeover_states = {}
    if all(v is not None for v in [cap_df, co_def, simulation_date, rate_map]):
        changeover_states = _analyze_end_of_day_changeover(
            plan_df, cap_df, co_def, simulation_date, rate_map
        )

    line_states = {}
    grouped = plan_df.groupby(['line', 'simulation_date'])

    for (line, sim_date), group in grouped:
        last_prod = group.sort_values('production_plan_date').iloc[-1]
        co_state = changeover_states.get(line)

        line_states[line] = _build_line_state(last_prod, co_state)

    return line_states


def _build_line_state(
    last_prod: pd.Series,
    co_state: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """构建产线状态。

    参数：
        last_prod: 最后生产记录
        co_state: 换产状态

    返回：
        Dict[str, Any]: 产线状态
    """
    base_state = {
        'last_material': str(last_prod['material']),
        'last_location': str(last_prod['location']),
        'last_production_date': last_prod[
            'production_plan_date'
        ].strftime('%Y-%m-%d'),
    }

    if co_state and co_state.get('last_activity') == 'changeover':
        base_state['last_activity'] = 'changeover'
        base_state['changeover_info'] = co_state['changeover_info']
    else:
        base_state['last_activity'] = 'production'
        base_state['changeover_info'] = None

    return base_state


def _analyze_end_of_day_changeover(
    plan_df: pd.DataFrame,
    cap_df: pd.DataFrame,
    co_def: Dict[tuple, float],
    simulation_date: pd.Timestamp,
    rate_map: Dict[tuple, float]
) -> Dict[str, Any]:
    """分析日末换产状态。

    参数：
        plan_df: 生产计划
        cap_df: 产能数据
        co_def: 换产定义
        simulation_date: 仿真日期
        rate_map: 产率映射

    返回：
        Dict[str, Any]: 换产状态字典
    """
    states = {}
    typical_co_time = 1.0

    for line, group in plan_df.groupby('line'):
        sim_group = group[group['simulation_date'] == simulation_date]
        if sim_group.empty:
            continue

        for prod_date, prod_group in sim_group.groupby('production_plan_date'):
            state = _check_line_changeover(
                line, prod_date, prod_group, cap_df,
                co_def, rate_map, typical_co_time
            )
            if state:
                states[line] = state

    return states


def _check_line_changeover(
    line: str,
    prod_date: pd.Timestamp,
    prod_group: pd.DataFrame,
    cap_df: pd.DataFrame,
    co_def: Dict[tuple, float],
    rate_map: Dict[tuple, float],
    typical_co_time: float
) -> Optional[Dict[str, Any]]:
    """检查产线换产状态。

    参数：
        line: 产线
        prod_date: 生产日期
        prod_group: 生产分组
        cap_df: 产能数据
        co_def: 换产定义
        rate_map: 产率映射
        typical_co_time: 典型换产时间

    返回：
        Optional[Dict[str, Any]]: 换产状态或None
    """
    day_cap = cap_df[cap_df['date'] == prod_date]
    if day_cap.empty:
        return None

    line_cap = day_cap[day_cap['line'] == line]['capacity'].sum()
    if line_cap <= 0:
        return None

    total_allocated = 0
    last_material = None

    # 优化：使用 itertuples() 替代 iterrows()
    sorted_group = prod_group.sort_values('production_plan_date')
    for row in sorted_group.itertuples(index=False):
        material = row.material
        quantity = row.con_planned_qty
        changeover_id = getattr(row, 'changeover_id', None)

        if changeover_id and pd.notna(changeover_id):
            total_allocated += co_def.get((changeover_id, line), 0)

        rate = rate_map.get((material, line), 1)
        if rate and quantity > 0:
            total_allocated += quantity / rate

        last_material = material

    remaining = line_cap - total_allocated

    if remaining > 0.1 and last_material:
        if abs(remaining - typical_co_time) < 0.1:
            return {
                'last_activity': 'changeover',
                'changeover_info': {
                    'changeover_id': 'INFERRED_INCOMPLETE',
                    'from_material': last_material,
                    'to_material': 'UNKNOWN_NEXT',
                    'total_time': typical_co_time,
                    'completed_time': remaining,
                    'remaining_time': typical_co_time - remaining,
                }
            }

    return None


def calculate_changeover_metrics(
    production_plan: pd.DataFrame,
    changeover_def: pd.DataFrame
) -> pd.DataFrame:
    """计算换产指标。

    参数：
        production_plan: 生产计划
        changeover_def: 换产定义

    返回：
        pd.DataFrame: 换产日志
    """
    if production_plan.empty or changeover_def.empty:
        return pd.DataFrame(columns=[
            'date', 'location', 'line', 'changeover_type',
            'count', 'time', 'cost', 'mu_loss'
        ])

    changeover_log = []

    summary = _group_changeovers(production_plan)
    def_indexed = _prepare_changeover_def(changeover_def)

    # 优化：使用 itertuples() 替代 iterrows()
    for row in summary.itertuples(index=False):
        record = _create_changeover_record(row, def_indexed)
        if record:
            changeover_log.append(record)

    return pd.DataFrame(changeover_log)


def _group_changeovers(plan: pd.DataFrame) -> pd.DataFrame:
    """分组换产记录。

    参数：
        plan: 生产计划

    返回：
        pd.DataFrame: 分组统计
    """
    filtered = plan[plan['changeover_id'].notna()]
    return filtered.groupby([
        'production_plan_date', 'location', 'line', 'changeover_id'
    ]).size().reset_index(name='count')


def _prepare_changeover_def(
    changeover_def: pd.DataFrame
) -> pd.DataFrame:
    """准备换产定义索引。

    参数：
        changeover_def: 换产定义

    返回：
        pd.DataFrame: 索引后的定义
    """
    clean = changeover_def.drop_duplicates(
        subset=['changeover_id', 'line'],
        keep='first'
    )
    return clean.set_index(['changeover_id', 'line'])


def _create_changeover_record(
    row,
    def_indexed: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """创建换产记录。

    参数：
        row: 汇总行（namedtuple 或 Series）
        def_indexed: 索引后的定义

    返回：
        Optional[Dict[str, Any]]: 换产记录
    """
    # 兼容 itertuples() 返回的 namedtuple
    if hasattr(row, '_fields'):
        changeover_id = row.changeover_id
        line = row.line
        count = row.count
        prod_date = row.production_plan_date
        location = row.location
    else:
        changeover_id = row['changeover_id']
        line = row['line']
        count = row['count']
        prod_date = row['production_plan_date']
        location = row['location']

    try:
        definition = def_indexed.loc[(changeover_id, line)]

        if isinstance(definition, pd.Series):
            time_per = float(definition.get('time', 0))
            cost_per = float(definition.get('cost', 0))
            mu_loss_per = float(definition.get('mu_loss', 0))
        else:
            time_per = float(definition.iloc[0].get('time', 0))
            cost_per = float(definition.iloc[0].get('cost', 0))
            mu_loss_per = float(definition.iloc[0].get('mu_loss', 0))

    except KeyError:
        print(
            f"警告: 未找到换产定义 "
            f"changeover_id={changeover_id}, line={line}"
        )
        time_per = cost_per = mu_loss_per = 0

    return {
        'date': prod_date,
        'location': location,
        'line': line,
        'changeover_type': changeover_id,
        'count': count,
        'time': count * time_per,
        'cost': count * cost_per,
        'mu_loss': count * mu_loss_per,
    }


def simulate_production(
    plan: pd.DataFrame,
    pr_cfg: pd.DataFrame,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """仿真生产可靠性。

    参数：
        plan: 生产计划
        pr_cfg: 生产可靠性配置
        seed: 随机种子

    返回：
        pd.DataFrame: 增加produced_qty的计划表
    """
    # 优先尝试 DuckDB 优化路径
    if DUCKDB_AVAILABLE and is_duckdb_available():
        try:
            return simulate_production_batch_duckdb(plan, pr_cfg, seed)
        except Exception as e:
            print(f"[M4] DuckDB optimization failed, using pandas: {e}")
    
    # 原始 Pandas 实现
    if plan.empty or 'con_planned_qty' not in plan.columns:
        plan['produced_qty'] = []
        return plan

    # 注意：不要对plan进行排序！源码ChainSight_Dev/module4.py的simulate_production
    # 直接按原始顺序处理，排序会导致随机数分配顺序不同，产生不同的produced_qty结果

    rng = np.random.RandomState(seed)
    pr_map = pr_cfg.set_index(['location', 'line'])['pr'].to_dict()

    def simulate_row(row: pd.Series) -> int:
        pr = pr_map.get((row['location'], row['line']), 1)
        return rng.binomial(int(row['con_planned_qty']), pr)

    plan['produced_qty'] = plan.apply(simulate_row, axis=1)
    return plan
