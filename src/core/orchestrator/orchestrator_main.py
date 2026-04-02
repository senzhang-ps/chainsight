"""编排器核心类定义。

职责：
- 作为供应链仿真的统一状态中枢，承接 M1、M4、M5、M6、M3 之间的状态流转。
- 集中维护库存、开放调拨、在途、GR、空间容量与审计日志。
- 对外暴露初始化、状态更新与持久化相关的主入口方法。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .daily_ops import OrchestratorDailyOpsMixin
from .inventory_log import OrchestratorInventoryLogMixin
from .normalize import _normalize_identifiers
from .persistence import OrchestratorPersistenceMixin
from .processors import OrchestratorProcessorsMixin
from .views import OrchestratorViewsMixin


class Orchestrator(
    OrchestratorViewsMixin,
    OrchestratorProcessorsMixin,
    OrchestratorDailyOpsMixin,
    OrchestratorPersistenceMixin,
    OrchestratorInventoryLogMixin,
):
    """供应链计划的中心状态管理与协调枢纽"""

    def __init__(
        self,
        start_date: str,
        output_dir: str = "./orchestrator_output",
    ):
        """初始化编排器

        Args:
            start_date: 仿真开始日期（YYYY-MM-DD）
            output_dir: 持久化存储目录
        """
        self.start_date = (
            pd.to_datetime(start_date).normalize()
        )
        self.current_date = self.start_date
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 核心状态管理
        self.unrestricted_inventory: Dict[
            Tuple[str, str], int
        ] = {}  # (material, location) -> 数量
        self.open_deployment: Dict[str, Dict] = (
            {}
        )  # uid -> 开放调拨记录
        self.in_transit: Dict[str, Dict] = (
            {}
        )  # uid -> 在途记录
        self.production_gr: List[Dict] = (
            []
        )  # 每日生产收货记录
        self.delivery_gr: List[Dict] = (
            []
        )  # 每日交付收货记录
        self.shipment_log: List[Dict] = (
            []
        )  # 每日客户发货记录
        self.production_plan_backlog: List[Dict] = (
            []
        )  # 存所有已确认生产(含未来)，供 M3 查询
        # 空间容量配置
        self.space_capacity: pd.DataFrame = (
            pd.DataFrame()
        )

        # 按日期索引实现 O(1) 查询
        self.production_gr_by_date: Dict[
            str, List[Dict]
        ] = {}  # date_str -> 记录列表
        self.delivery_gr_by_date: Dict[
            str, List[Dict]
        ] = {}  # date_str -> 记录列表
        self.shipment_log_by_date: Dict[
            str, List[Dict]
        ] = {}  # date_str -> 记录列表
        self.delivery_shipment_log_by_date: Dict[
            str, List[Dict]
        ] = {}  # date_str -> 记录列表

        # UID 序列计数器
        self.uid_sequence = 0
        # 过期清理的全局宽限天数
        self.cleanup_grace_days: int = 100

        # 用于审计的每日日志
        self.daily_logs: List[Dict] = []

        # 期初和期末库存存储
        self.daily_beginning_inventory: Dict[
            str, Dict[Tuple[str, str], int]
        ] = (
            {}
        )  # date -> {(material, location): quantity}
        self.daily_ending_inventory: Dict[
            str, Dict[Tuple[str, str], int]
        ] = (
            {}
        )  # date -> {(material, location): quantity}

        # 初始库存配置存储
        self.initial_inventory: Dict[
            Tuple[str, str], int
        ] = (
            {}
        )  # (material, location) -> 数量

        # 发运出库日志
        self.delivery_shipment_log: List[Dict] = (
            []
        )  # M6 产生的每日调拨发运记录

        # 记录当天是否已完成过一次清理
        self._last_cleanup_date: Optional[
            pd.Timestamp
        ] = None

        msg = (
            f"✅ Orchestrator initialized for "
            f"simulation starting {start_date}"
        )
        print(msg)

    def initialize_inventory(
        self, initial_inventory_df: pd.DataFrame
    ):
        """从 M1_InitialInventory 配置初始化实物库存。

        Args:
            initial_inventory_df: 含列 [material,
                                   location, quantity]
        """
        self.unrestricted_inventory.clear()
        self.initial_inventory.clear()

        # 确保标识符字段为字符串格式
        normalized_df = _normalize_identifiers(
            initial_inventory_df
        )

        # 使用 itertuples 替代 iterrows
        for row in normalized_df.itertuples():
            key = (row.material, row.location)
            quantity = int(row.quantity)
            self.unrestricted_inventory[key] = quantity
            self.initial_inventory[key] = quantity

        msg = f"Initialized {len(normalized_df)} records"
        self._log_event("INIT_INVENTORY", msg)

    def set_space_capacity(
        self, space_capacity_df: pd.DataFrame
    ):
        """从 Global_SpaceCapacity 配置设置空间容量。

        Args:
            space_capacity_df: 含列 [location,
                               eff_from, eff_to,
                               capacity]
        """
        # 确保标识符字段为字符串格式
        self.space_capacity = _normalize_identifiers(
            space_capacity_df.copy()
        )
        self.space_capacity["eff_from"] = (
            pd.to_datetime(
                self.space_capacity["eff_from"].astype(
                    str
                ),
                format="%Y-%m-%d",
                errors="coerce",
            )
        )
        self.space_capacity["eff_to"] = (
            pd.to_datetime(
                self.space_capacity["eff_to"].astype(str),
                format="%Y-%m-%d",
                errors="coerce",
            )
        )

        msg = (
            f"Configured {len(space_capacity_df)} "
            f"space capacity records"
        )
        self._log_event("SET_SPACE_CAPACITY", msg)

    def _safe_convert_to_int(self, value):
        """安全转换 pandas Series 或标量为整数"""
        try:
            # 如果是 pandas Series，取第一个值
            if hasattr(value, 'iloc') and len(value) > 0:
                value = value.iloc[0]
            elif hasattr(value, 'item'):
                value = value.item()
            elif isinstance(value, pd.Series):
                # 处理特殊情况的Series
                if len(value) == 1:
                    value = value.iloc[0]
                elif len(value) > 1:
                    msg = (
                        f"    ⚠️  Series有多个值，"
                        f"取第一个: {value.iloc[0]}"
                    )
                    print(msg)
                    value = value.iloc[0]
                else:
                    # 空Series
                    return 0

            # 处理None或NaN
            if value is None or pd.isna(value):
                return 0

            # 转换为int
            return int(float(value))

        except (
            ValueError,
            TypeError,
            IndexError,
            AttributeError,
        ) as e:
            msg = (
                f"    ⚠️  数值转换错误: {value} "
                f"(类型: {type(value)}) -> {e}"
            )
            print(msg)
            return 0


def create_orchestrator(
    start_date: str, output_dir: str = "./orchestrator_output"
) -> Orchestrator:
    """创建并初始化编排器实例

    Args:
        start_date: 仿真开始日期（YYYY-MM-DD）
        output_dir: 持久化存储输出目录

    Returns:
        Orchestrator 实例
    """
    return Orchestrator(start_date, output_dir)
