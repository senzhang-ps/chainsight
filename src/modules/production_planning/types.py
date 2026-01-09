"""
类型定义模块

定义Module4中使用的数据类和类型别名。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class ChangeoverInfo:
    """换产信息数据类。

    Attributes:
        changeover_id: 换产标识符
        from_material: 源物料
        to_material: 目标物料
        total_time: 总换产时间（小时）
        completed_time: 已完成时间（小时）
        remaining_time: 剩余时间（小时）
    """

    changeover_id: str
    from_material: str
    to_material: str
    total_time: float
    completed_time: float = 0.0
    remaining_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。

        Returns:
            Dict[str, Any]: 换产信息字典
        """
        return {
            'changeover_id': self.changeover_id,
            'from_material': self.from_material,
            'to_material': self.to_material,
            'total_time': self.total_time,
            'completed_time': self.completed_time,
            'remaining_time': self.remaining_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChangeoverInfo':
        """从字典创建实例。

        Args:
            data: 换产信息字典

        Returns:
            ChangeoverInfo: 换产信息实例
        """
        return cls(
            changeover_id=data.get('changeover_id', ''),
            from_material=data.get('from_material', ''),
            to_material=data.get('to_material', ''),
            total_time=float(data.get('total_time', 0)),
            completed_time=float(data.get('completed_time', 0)),
            remaining_time=float(data.get('remaining_time', 0)),
        )


@dataclass
class LineState:
    """产线状态数据类。

    Attributes:
        last_material: 最后生产的物料
        last_location: 最后的地点
        last_production_date: 最后生产日期
        last_activity: 最后活动类型（production/changeover）
        changeover_info: 换产信息（如有未完成换产）
    """

    last_material: str
    last_location: str
    last_production_date: str
    last_activity: str = 'production'
    changeover_info: Optional[ChangeoverInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。

        Returns:
            Dict[str, Any]: 产线状态字典
        """
        result = {
            'last_material': self.last_material,
            'last_location': self.last_location,
            'last_production_date': self.last_production_date,
            'last_activity': self.last_activity,
        }
        if self.changeover_info:
            result['changeover_info'] = self.changeover_info.to_dict()
        else:
            result['changeover_info'] = None
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LineState':
        """从字典创建实例。

        Args:
            data: 产线状态字典

        Returns:
            LineState: 产线状态实例
        """
        changeover_info = None
        if data.get('changeover_info'):
            changeover_info = ChangeoverInfo.from_dict(data['changeover_info'])

        return cls(
            last_material=str(data.get('last_material', '')),
            last_location=str(data.get('last_location', '')),
            last_production_date=str(data.get('last_production_date', '')),
            last_activity=data.get('last_activity', 'production'),
            changeover_info=changeover_info,
        )


@dataclass
class PlanRecord:
    """生产计划记录数据类。

    Attributes:
        material: 物料编码
        location: 地点
        line: 产线
        simulation_date: 仿真日期
        production_plan_date: 生产计划日期
        available_date: 可用日期
        uncon_planned_qty: 无约束计划量
        con_planned_qty: 约束计划量
        changeover_id: 换产标识符
    """

    material: str
    location: str
    line: str
    simulation_date: datetime
    production_plan_date: datetime
    available_date: datetime
    uncon_planned_qty: int
    con_planned_qty: int
    changeover_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。

        Returns:
            Dict[str, Any]: 计划记录字典
        """
        return {
            'material': self.material,
            'location': self.location,
            'line': self.line,
            'simulation_date': self.simulation_date,
            'production_plan_date': self.production_plan_date,
            'available_date': self.available_date,
            'uncon_planned_qty': self.uncon_planned_qty,
            'con_planned_qty': self.con_planned_qty,
            'changeover_id': self.changeover_id,
        }


@dataclass
class ExceedRecord:
    """产能超额记录数据类。

    Attributes:
        material: 物料编码
        location: 地点
        line: 产线
        simulation_date: 仿真日期
        production_plan_date: 生产计划日期
        unmet_qty: 未满足数量
    """

    material: str
    location: str
    line: str
    simulation_date: datetime
    production_plan_date: datetime
    unmet_qty: int

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。

        Returns:
            Dict[str, Any]: 超额记录字典
        """
        return {
            'material': self.material,
            'location': self.location,
            'line': self.line,
            'simulation_date': self.simulation_date,
            'production_plan_date': self.production_plan_date,
            'unmet_uncon_planned_qty': self.unmet_qty,
        }


@dataclass
class ValidationIssue:
    """校验问题数据类。

    Attributes:
        issue_type: 问题类型
        sheet: 相关工作表
        row: 相关行
        issue: 问题描述
        location: 地点（可选）
        line: 产线（可选）
        message: 详细消息（可选）
    """

    issue_type: str
    sheet: str
    row: str
    issue: str
    location: Optional[str] = None
    line: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。

        Returns:
            Dict[str, Any]: 校验问题字典
        """
        return {
            'type': self.issue_type,
            'sheet': self.sheet,
            'row': self.row,
            'issue': self.issue,
            'location': self.location,
            'line': self.line,
            'message': self.message,
        }
