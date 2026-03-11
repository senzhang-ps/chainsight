"""编排器状态持久化集合。

职责：
- 将每日状态视图写出为标准 CSV 文件。
- 记录审计日志并维护期初、期末库存快照。
- 为续跑恢复、数据库写入与交接核对提供稳定落盘结果。
"""

from datetime import datetime

import pandas as pd

from .normalize import (
    _normalize_identifiers,
    _normalize_location,
    _normalize_material,
)


class OrchestratorPersistenceMixin:
    """状态持久化方法集合（Mixin）。"""

    def save_daily_state(self, date: str):
        """将每日状态保存到持久化存储。

        Args:
            date: 日期（YYYY-MM-DD）
        """
        date_str = (
            pd.to_datetime(date).strftime('%Y%m%d')
        )

        # 保存非限制库存视图
        unrestricted_df = (
            self.get_unrestricted_inventory_view(date)
        )
        out_path = self.output_dir / (
            f"unrestricted_inventory_{date_str}.csv"
        )
        _normalize_identifiers(
            unrestricted_df
        ).to_csv(out_path, index=False)

        # 保存开放调拨视图
        open_deployment_df = (
            self.get_open_deployment_view(date)
        )
        out_path = self.output_dir / (
            f"open_deployment_{date_str}.csv"
        )
        _normalize_identifiers(
            open_deployment_df
        ).to_csv(out_path, index=False)

        # 保存在途视图
        intransit_df = (
            self.get_planning_intransit_view(date)
        )
        out_path = self.output_dir / (
            f"planning_intransit_{date_str}.csv"
        )
        _normalize_identifiers(intransit_df).to_csv(
            out_path, index=False
        )

        # 保存空间额度视图
        space_quota_df = self.get_space_quota_view(
            date
        )
        out_path = self.output_dir / (
            f"space_quota_{date_str}.csv"
        )
        _normalize_identifiers(
            space_quota_df
        ).to_csv(out_path, index=False)

        # 保存生产计划 backlog
        production_backlog_df = (
            self.get_production_plan_backlog_view(
                date
            )
        )
        out_path = self.output_dir / (
            f"production_plan_backlog_{date_str}.csv"
        )
        _normalize_identifiers(
            production_backlog_df
        ).to_csv(out_path, index=False)

        # 保存每日交付 GR
        delivery_gr_df = self.get_delivery_gr_view(
            date
        )
        out_path = self.output_dir / (
            f"delivery_gr_{date_str}.csv"
        )
        _normalize_identifiers(
            delivery_gr_df
        ).to_csv(out_path, index=False)

        # 保存每日生产 GR
        production_gr_df = (
            self.get_production_gr_view(date)
        )
        out_path = self.output_dir / (
            f"production_gr_{date_str}.csv"
        )
        _normalize_identifiers(
            production_gr_df
        ).to_csv(out_path, index=False)

        # 保存每日发货日志
        date_key = pd.to_datetime(date).strftime(
            '%Y-%m-%d'
        )
        daily_shipments = (
            self.shipment_log_by_date.get(
                date_key, []
            )
        )
        shipment_df = pd.DataFrame(daily_shipments)
        if shipment_df.empty:
            shipment_df = pd.DataFrame(columns=[
                'date',
                'material',
                'location',
                'quantity',
            ])
        out_path = self.output_dir / (
            f"shipment_log_{date_str}.csv"
        )
        _normalize_identifiers(shipment_df).to_csv(
            out_path, index=False
        )

        # 保存发运出库日志
        daily_delivery_shipments = (
            self.delivery_shipment_log_by_date.get(
                date_key, []
            )
        )
        delivery_shipment_df = pd.DataFrame(
            daily_delivery_shipments
        )
        if delivery_shipment_df.empty:
            delivery_shipment_df = pd.DataFrame(
                columns=[
                    'date',
                    'material',
                    'sending',
                    'receiving',
                    'quantity',
                    'ori_deployment_uid',
                    'actual_ship_date',
                    'actual_delivery_date',
                    'type',
                ]
            )
        out_path = self.output_dir / (
            f"delivery_shipment_log_{date_str}.csv"
        )
        _normalize_identifiers(
            delivery_shipment_df
        ).to_csv(out_path, index=False)

        # 生成库存变动日志
        inventory_change_df = (
            self.generate_inventory_change_log(date)
        )
        out_path = self.output_dir / (
            f"inventory_change_log_{date_str}.csv"
        )
        _normalize_identifiers(
            inventory_change_df
        ).to_csv(out_path, index=False)

        # 保存每日日志
        logs_file = self.output_dir / (
            f"daily_logs_{date_str}.csv"
        )
        if self.daily_logs:
            logs_df = pd.DataFrame(self.daily_logs)
        else:
            logs_df = pd.DataFrame(columns=[
                'timestamp',
                'date',
                'event_type',
                'message',
            ])
        logs_df.to_csv(logs_file, index=False)

    def save_beginning_inventory(self, date: str):
        """保存指定日期的期初库存状态。

        在任何库存变动之前调用。

        Args:
            date: 日期字符串（YYYY-MM-DD）
        """
        self.daily_beginning_inventory[date] = (
            self.unrestricted_inventory.copy()
        )
        msg = (
            f"💾已保存 {date} 期初库存: "
            f"{len(self.unrestricted_inventory)} 项"
        )
        print(msg)

    def save_ending_inventory(self, date: str):
        """保存指定日期的期末库存状态。

        在所有模块运行完成后调用。

        Args:
            date: 日期字符串（YYYY-MM-DD）
        """
        self.daily_ending_inventory[date] = (
            self.unrestricted_inventory.copy()
        )
        msg = (
            f"💾已保存 {date} 期末库存: "
            f"{len(self.unrestricted_inventory)} 项"
        )
        print(msg)

    def _log_event(
        self, event_type: str, message: str
    ):
        """记录编排器事件用于审计追踪。

        Args:
            event_type: 事件类型
            message: 事件消息
        """
        self.daily_logs.append({
            'timestamp': datetime.now().isoformat(),
            'date': self.current_date.strftime(
                '%Y-%m-%d'
            ),
            'event_type': event_type,
            'message': message,
        })

    def output_daily_inventory_summary(
        self, date: str
    ):
        """输出指定日期的详细库存变动记录。

        用于与库存平衡检查对照。

        Args:
            date: 日期字符串（YYYY-MM-DD）
        """
        # 获取期初期末库存
        beginning_inv = (
            self.daily_beginning_inventory.get(
                date, {}
            )
        )
        ending_inv = self.daily_ending_inventory.get(
            date, {}
        )

        date_str = pd.to_datetime(date).strftime(
            '%Y-%m-%d'
        )

        # 获取当日各项变动
        production_gr = (
            self.production_gr_by_date.get(
                date_str, []
            )
        )
        delivery_gr = self.delivery_gr_by_date.get(
            date_str, []
        )
        shipments = self.shipment_log_by_date.get(
            date_str, []
        )

        # M6 发运
        m6_ship_df = (
            self.get_delivery_shipment_log_view(date)
        )
        m6_ship_count = len(m6_ship_df)
        if not m6_ship_df.empty:
            m6_ship_qty_total = int(
                m6_ship_df['quantity'].sum()
            )
        else:
            m6_ship_qty_total = 0

        # 统计汇总
        print(f"期初库存条目: {len(beginning_inv)}")
        print(f"生产入库条目: {len(production_gr)}")
        print(f"交付入库条目: {len(delivery_gr)}")
        print(f"发货出库条目: {len(shipments)}")
        msg = (
            f"发运出库条目(M6): {m6_ship_count}，"
            f"数量合计: {m6_ship_qty_total}"
        )
        print(msg)
        print(f"期末库存条目: {len(ending_inv)}")
