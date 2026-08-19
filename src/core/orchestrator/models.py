# models.py
# 编排器数据模型定义
#
# 定义编排器使用的数据类和结构

from dataclasses import dataclass

import pandas as pd


# 完整集成链路的业务执行顺序。M4 严格消费前一自然日 M3 的结果，
# 因此 M3 必须在当日链路末尾执行。
MODULE_EXECUTION_ORDER = ("module1", "module4", "module5", "module6", "module3")

# 各模块对集成调度器承诺返回的 DataFrame 字段。模块内部可附加其他字段，
# 但这些字段是状态写回、持久化与回归对比的最小公共契约。
MODULE_RESULT_DATAFRAMES = {
    "module1": (
        "orders_df", "shipment_df", "cut_df", "supply_demand_df", "summary_df",
    ),
    "module3": ("net_demand_df",),
    "module4": (
        "production_df", "exceed_log", "issues_df", "changeover_log",
        "unconstrained_plan",
    ),
    "module5": (
        "deployment_plan", "unfulfilled_log", "stock_on_hand_log", "validation_log",
    ),
    "module6": (
        "delivery_plan", "vehicle_log", "truck_usage", "unsatisfied_log",
        "validation_log", "bypass_log",
    ),
}

# 集成回归对比需要的日末 StateContext 视图。值为对应的 StateContext getter 名称。
INTEGRATION_CONTEXT_VIEW_GETTERS = {
    "unrestricted_inventory": "get_unrestricted_inventory_view",
    "open_deployment": "get_open_deployment_view",
    "planning_intransit": "get_planning_intransit_view",
    "space_quota": "get_space_quota_view",
    "delivery_gr": "get_delivery_gr_view",
    "production_gr": "get_production_gr_view",
    "production_plan_backlog": "get_production_plan_backlog_view",
    "shipment_log": "get_shipment_log_view",
    "delivery_shipment_log": "get_delivery_shipment_log_view",
    "inventory_change_log": "generate_inventory_change_log",
}


def validate_module_result(module_id: str, result: dict, date_str: str) -> None:
    """校验模块集成输出是否满足统一结果契约。"""
    if module_id not in MODULE_RESULT_DATAFRAMES:
        raise ValueError(f"未知模块结果: {module_id}")
    if not isinstance(result, dict):
        raise TypeError(
            f"{date_str} {module_id} 输出必须为 dict，实际为 {type(result).__name__}"
        )

    missing = [key for key in MODULE_RESULT_DATAFRAMES[module_id] if key not in result]
    if missing:
        raise ValueError(f"{date_str} {module_id} 输出缺少结果字段: {missing}")

    invalid = [
        key for key in MODULE_RESULT_DATAFRAMES[module_id]
        if not isinstance(result[key], pd.DataFrame)
    ]
    if invalid:
        raise TypeError(f"{date_str} {module_id} 结果字段不是 DataFrame: {invalid}")


@dataclass
class DeploymentUID:
    """用于部署跟踪的唯一标识符"""

    material: str
    sending: str
    receiving: str
    planned_deploy_date: str  # YYYY-MM-DD format
    demand_element: str
    sequence: int  # Auto-incrementing sequence for uniqueness

    def to_string(self) -> str:
        """转换为字符串表示以便跟踪"""
        mat = self.material
        snd = self.sending
        rcv = self.receiving
        pdd = self.planned_deploy_date
        de = self.demand_element
        seq = self.sequence
        return (
            f"{mat}|{snd}|{rcv}|{pdd}|{de}|{seq:06d}"
        )

    @classmethod
    def from_string(cls, uid_str: str) -> 'DeploymentUID':
        """从字符串表示解析"""
        parts = uid_str.split('|')
        return cls(
            material=parts[0],
            sending=parts[1],
            receiving=parts[2],
            planned_deploy_date=parts[3],
            demand_element=parts[4],
            sequence=int(parts[5]),
        )
