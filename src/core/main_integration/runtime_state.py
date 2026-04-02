"""
runtime_state.py

数据库模式下的内存运行时状态，替代每日仿真循环中模块间数据传递所依赖的临时文件中转。

该状态：
- 在单次仿真运行的各天之间持续携带
- 序列化到 checkpoint JSON 以支持断点续跑
- 仅在文件（本地）模式下不使用，对 simulation_file.py 零影响
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

import pandas as pd


class DbRuntimeState:
    """保存数据库模式下原本依赖临时文件中转的跨天状态数据。

    属性：
        m4_line_states: 按天索引的产线状态字典，格式为 {日期字符串: {产线名: 状态字典}}。
            由 module4 每天写入，供下一天的 module4 读取。
        m4_allocated_capacity: 按天索引的已分配产能字典，格式为 {日期字符串: {键: 小时数}}。
            由 module4 每天写入，累计合并用于下一天。
        previous_m3_result: 前一天 module3 的结果字典
            （含 'net_demand_df'），供 module4 加载日度净需求。
        cleanup_audit_df: 当天 cleanup_past_due_open_deployments() 的返回值；
            为瞬态数据，由当日状态刷写/批量写入数据库流程消费（不序列化到 checkpoint）。
    """

    def __init__(self) -> None:
        # M4 跨天产线状态：{日期字符串: {产线名: 状态字典}}
        self.m4_line_states: Dict[str, dict] = {}
        # M4 跨天已分配产能：{日期字符串: {键: 小时数（浮点）}}
        self.m4_allocated_capacity: Dict[str, dict] = {}
        # 前一天的 M3 结果（供 M4 次日加载净需求）
        self.previous_m3_result: Optional[Dict[str, Any]] = None
        # 当天的清理审计数据（瞬态，不序列化到 checkpoint）
        self.cleanup_audit_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # M4 产线状态辅助方法（替代基于文件的 load_line_state / save_line_state）
    # ------------------------------------------------------------------

    def save_line_state(self, sim_date: pd.Timestamp, line_states: dict) -> None:
        """存储当天的产线状态，供下一天跨天续转使用。"""
        date_key = sim_date.strftime('%Y%m%d')
        self.m4_line_states[date_key] = line_states

    def load_line_state(self, sim_date: pd.Timestamp) -> dict:
        """加载前一天的产线状态（与 module4.load_line_state 逻辑相同）。"""
        prev_date = sim_date - pd.Timedelta(days=1)
        date_key = prev_date.strftime('%Y%m%d')
        return self.m4_line_states.get(date_key, {})

    # ------------------------------------------------------------------
    # M4 已分配产能辅助方法（替代基于文件的 load_all_previous_capacity）
    # ------------------------------------------------------------------

    def save_allocated_capacity(self, sim_date: pd.Timestamp, capacity: dict) -> None:
        """存储当天的已分配产能。"""
        date_key = sim_date.strftime('%Y%m%d')
        self.m4_allocated_capacity[date_key] = capacity

    def load_all_previous_capacity(self, sim_date: pd.Timestamp) -> dict:
        """合并 sim_date 之前（不含）所有历史已分配产能。

        复现 module4.load_all_previous_capacity 的累加合并逻辑：
        对每个历史天的数据执行 consolidated[key] += value。
        """
        consolidated: Dict[str, float] = {}
        sim_date_key = sim_date.strftime('%Y%m%d')
        for date_key in sorted(self.m4_allocated_capacity.keys()):
            if date_key >= sim_date_key:
                break
            for key, value in self.m4_allocated_capacity[date_key].items():
                consolidated[key] = consolidated.get(key, 0.0) + value
        return consolidated

    # ------------------------------------------------------------------
    # 序列化 / 反序列化（用于 checkpoint 存储）
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """序列化为 JSON 兼容的字典，用于 checkpoint 存储。"""

        def _serialize_m3(result: Optional[Dict[str, Any]]) -> Optional[dict]:
            """将 M3 结果序列化为可 JSON 化的字典。"""
            if result is None:
                return None
            out = {}
            for k, v in result.items():
                if isinstance(v, pd.DataFrame):
                    out[k] = v.to_dict('records')
                elif isinstance(v, pd.Timestamp):
                    out[k] = v.isoformat()
                else:
                    out[k] = v
            return out

        return {
            'm4_line_states': self.m4_line_states,
            'm4_allocated_capacity': self.m4_allocated_capacity,
            'previous_m3_result': _serialize_m3(self.previous_m3_result),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DbRuntimeState':
        """从 checkpoint 字典反序列化。"""
        state = cls()
        if not data:
            return state
        state.m4_line_states = data.get('m4_line_states', {})
        state.m4_allocated_capacity = data.get('m4_allocated_capacity', {})

        m3_raw = data.get('previous_m3_result')
        if m3_raw is not None:
            restored = {}
            for k, v in m3_raw.items():
                if k == 'net_demand_df' and isinstance(v, list):
                    df = pd.DataFrame(v)
                    # 还原日期列为 datetime 类型
                    for col in ('requirement_date', 'simulation_date'):
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    restored[k] = df
                elif k == 'simulation_date' and isinstance(v, str):
                    restored[k] = pd.to_datetime(v)
                else:
                    restored[k] = v
            state.previous_m3_result = restored

        return state
