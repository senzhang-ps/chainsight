"""编排器每日运行操作集合。

职责：
- 组织单日状态推进的执行顺序。
- 处理到货入库、开放调拨清理与日终状态保存。
- 为主编排器提供日粒度的流程化能力。
"""

from typing import Optional

import pandas as pd

from ...utils.normalization import (
    normalize_identifiers,
    normalize_material,
    normalize_receiving,
    normalize_sending,
)


class OrchestratorDailyOpsMixin:
    """每日运行操作方法集合（Mixin）。"""

    def run_daily_processing(
        self,
        date: str,
        shipment_df: Optional[pd.DataFrame] = None,
        production_df: Optional[pd.DataFrame] = None,
        deployment_df: Optional[pd.DataFrame] = None,
        delivery_df: Optional[pd.DataFrame] = None,
        grace_days: Optional[int] = None,
    ):
        """执行每日处理。

        顺序: M1 → M4 → M5 → M6

        Args:
            date: 仿真日期（YYYY-MM-DD）
            shipment_df: Module1 发货数据
            production_df: Module4 生产数据
            deployment_df: Module5 部署数据
            delivery_df: Module6 交付数据
            grace_days: 过期清理宽限天数
        """
        self.current_date = (
            pd.to_datetime(date).normalize()
        )

        # 仅在每日跑批开头清理一次
        normalized_date_str = pd.to_datetime(
            date
        ).strftime('%Y-%m-%d')
        g = (
            self.cleanup_grace_days
            if grace_days is None
            else int(grace_days)
        )
        if self._last_cleanup_date != (
            normalized_date_str
        ):
            self.cleanup_past_due_open_deployments(
                date, grace_days=g, write_audit=True
            )
            self._last_cleanup_date = (
                normalized_date_str
            )

        # 在每日开始时检查到货
        self._process_delivery_arrivals(date)

        # M1：处理发货
        if (
            shipment_df is not None
            and not shipment_df.empty
        ):
            self.process_module1_shipments(
                shipment_df, date
            )

        # M4：处理生产
        if (
            production_df is not None
            and not production_df.empty
        ):
            self.process_module4_production(
                production_df, date
            )

        # M5：处理部署
        if (
            deployment_df is not None
            and not deployment_df.empty
        ):
            self.process_module5_deployment(
                deployment_df, date
            )

        # M6：处理交付
        if (
            delivery_df is not None
            and not delivery_df.empty
        ):
            self.process_module6_delivery(
                delivery_df, date
            )

        # 保存每日状态
        self.save_daily_state(date)


    def _process_delivery_arrivals(self, date: str):
        """处理当天到达的在途交付。"""
        date_obj = pd.to_datetime(date).normalize()

        completed_transits = []
        for transit_uid, transit_record in (
            self.in_transit.items()
        ):
            actual_dd = pd.to_datetime(
                transit_record['actual_delivery_date']
            ).normalize()
            if actual_dd == date_obj:
                # 增加收货地库存
                receiving_key = (
                    transit_record['material'],
                    transit_record['receiving'],
                )
                old_inv = (
                    self.unrestricted_inventory.get(
                        receiving_key, 0
                    )
                )
                self.unrestricted_inventory[
                    receiving_key
                ] = (
                    old_inv
                    + transit_record['quantity']
                )

                # 记录 delivery GR
                gr_record = {
                    'date': date_obj,
                    'material': normalize_material(
                        transit_record['material']
                    ),
                    'receiving': normalize_receiving(
                        transit_record['receiving']
                    ),
                    'quantity': transit_record[
                        'quantity'
                    ],
                    'ori_deployment_uid': (
                        transit_record[
                            'ori_deployment_uid'
                        ]
                    ),
                    'vehicle_uid': transit_record[
                        'vehicle_uid'
                    ],
                    'actual_ship_date': (
                        transit_record[
                            'actual_ship_date'
                        ]
                    ),
                }

                # 改进的重复检查
                existing_key = (
                    date_obj,
                    transit_record['material'],
                    transit_record['receiving'],
                    transit_record[
                        'ori_deployment_uid'
                    ],
                    transit_record['vehicle_uid'],
                )
                date_str = date_obj.strftime(
                    '%Y-%m-%d'
                )
                if not any(
                    (
                        record['date'],
                        record['material'],
                        record['receiving'],
                        record['ori_deployment_uid'],
                        record['vehicle_uid'],
                    )
                    == existing_key
                    for record in self.delivery_gr_by_date.get(
                        date_str, []
                    )
                ):
                    self.delivery_gr.append(gr_record)
                    # 索引以便 O(1) 查询
                    if date_str not in (
                        self.delivery_gr_by_date
                    ):
                        self.delivery_gr_by_date[
                            date_str
                        ] = []
                    self.delivery_gr_by_date[
                        date_str
                    ].append(gr_record)

                completed_transits.append(
                    transit_uid
                )

        for transit_uid in completed_transits:
            del self.in_transit[transit_uid]

        if completed_transits:
            msg = (
                f"Processed "
                f"{len(completed_transits)} "
                f"delivery arrivals"
            )
            self._log_event("DELIVERY_ARRIVALS", msg)

    def cleanup_past_due_open_deployments(
        self,
        date: str,
        grace_days: int = 0,
        write_audit: bool = True,
    ) -> pd.DataFrame:
        """清理过期的 open deployment。

        规则：planned_deployment_date <
        (date - grace_days) 的记录会被清理

        Args:
            date: 当前仿真日期（YYYY-MM-DD）
            grace_days: 宽限天数
            write_audit: 是否写入审计CSV

        Returns:
            被清理掉的记录明细 DataFrame
        """
        cleanup_date = (
            pd.to_datetime(date).normalize()
        )
        threshold_date = cleanup_date - (
            pd.Timedelta(days=int(grace_days))
        )

        removed = []
        to_delete = []

        for uid, rec in self.open_deployment.items():
            pdd = pd.to_datetime(
                rec.get('planned_deployment_date')
            ).normalize()
            remaining_qty = int(
                rec.get('deployed_qty', 0)
            )
            # 只清理：计划日早于阈值
            if pdd < threshold_date:
                to_delete.append(uid)
                removed.append({
                    'cleanup_date': cleanup_date,
                    'grace_days': int(grace_days),
                    'ori_deployment_uid': uid,
                    'material': normalize_material(
                        rec.get('material')
                    ),
                    'sending': normalize_sending(
                        rec.get('sending')
                    ),
                    'receiving': normalize_receiving(
                        rec.get('receiving')
                    ),
                    'planned_deployment_date': pdd,
                    'remaining_qty': remaining_qty,
                    'demand_element': rec.get(
                        'demand_element', ''
                    ),
                    'creation_date': rec.get(
                        'creation_date', ''
                    ),
                    'reason': (
                        f"past_due>"
                        f"{int(grace_days)}d"
                    ),
                })

        # 真正删除
        for uid in to_delete:
            del self.open_deployment[uid]

        # 生成审计DF
        cleanup_df = pd.DataFrame(removed)
        if cleanup_df.empty:
            cleanup_df = pd.DataFrame(columns=[
                'cleanup_date',
                'grace_days',
                'ori_deployment_uid',
                'material',
                'sending',
                'receiving',
                'planned_deployment_date',
                'remaining_qty',
                'demand_element',
                'creation_date',
                'reason',
            ])

        # 写审计CSV
        if write_audit:
            date_str = cleanup_date.strftime(
                '%Y%m%d'
            )
            out_path = self.output_dir / (
                f"open_deployment_pastdue_"
                f"cleanup_{date_str}.csv"
            )
            normalize_identifiers(cleanup_df).to_csv(
                out_path, index=False
            )

        # 记录日志
        msg = (
            f"Removed {len(to_delete)} past-due "
            f"open deployments "
            f"(grace_days={grace_days})"
        )
        self._log_event(
            "OPEN_DEPLOYMENT_CLEANUP", msg
        )

        return cleanup_df

    def set_past_due_cleanup_grace_days(
        self, days: int
    ):
        """设置过期清理的全局宽限天数。

        Args:
            days: 宽限天数（默认0）
        """
        try:
            self.cleanup_grace_days = max(
                0, int(days)
            )
        except Exception:
            self.cleanup_grace_days = 0
