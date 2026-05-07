# views.py
# Orchestrator 只读查询方法 (Mixin)
#
# 提供库存视图、在途视图、调拨视图、生产视图等
# 所有 get_* 方法均为只读，不修改编排器状态。

from typing import Dict, Tuple

import pandas as pd

from ...utils.normalization import (
    normalize_location,
    normalize_material,
    normalize_receiving,
    normalize_sending,
)


class OrchestratorViewsMixin:
    """Orchestrator 只读查询方法集合（Mixin）。"""

    # ------ 库存视图 ------

    def get_unrestricted_inventory_view(
        self, date: str
    ) -> pd.DataFrame:
        """获取指定日期的非限制库存视图。

        Args:
            date: 日期（YYYY-MM-DD）

        Returns:
            含列 [date, material, location, quantity]
        """
        date_obj = pd.to_datetime(date).normalize()

        records = []
        for (mat, loc), qty in (
            self.unrestricted_inventory.items()
        ):
            records.append({
                'date': date_obj,
                'material': normalize_material(mat),
                'location': normalize_location(loc),
                'quantity': qty,
            })

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(
                columns=[
                    'date', 'material',
                    'location', 'quantity',
                ]
            )
        return df

    def get_current_unrestricted_inventory(
        self,
    ) -> Dict[Tuple[str, str], int]:
        """获取当前非限制库存字典（键已规范化）。

        Returns:
            {(material, location): quantity}
        """
        result: Dict[Tuple[str, str], int] = {}
        for (mat, loc), qty in (
            self.unrestricted_inventory.items()
        ):
            key = (
                normalize_material(mat),
                normalize_location(loc),
            )
            result[key] = qty
        return result

    # ------ 在途视图 ------

    def get_planning_intransit_view(
        self, date: str
    ) -> pd.DataFrame:
        """获取指定日期的在途库存视图。

        Args:
            date: 日期（YYYY-MM-DD）

        Returns:
            含列 [transit_uid, date, material, sending,
                  receiving, actual_ship_date,
                  actual_delivery_date, quantity,
                  ori_deployment_uid, vehicle_uid]
        """
        date_obj = pd.to_datetime(date).normalize()
        records = []
        for uid, rec in self.in_transit.items():
            records.append({
                'transit_uid': uid,
                'date': date_obj,
                'material': normalize_material(
                    rec['material']
                ),
                'sending': rec.get('sending', ''),
                'receiving': rec['receiving'],
                'actual_ship_date': rec.get(
                    'actual_ship_date', ''
                ),
                'actual_delivery_date': rec[
                    'actual_delivery_date'
                ],
                'quantity': rec['quantity'],
                'ori_deployment_uid': rec.get(
                    'ori_deployment_uid', ''
                ),
                'vehicle_uid': rec.get(
                    'vehicle_uid', ''
                ),
            })

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'transit_uid', 'date', 'material',
                'sending', 'receiving',
                'actual_ship_date',
                'actual_delivery_date', 'quantity',
                'ori_deployment_uid', 'vehicle_uid',
            ])
        return df

    # ------ 调拨视图 ------

    def get_open_deployment(
        self, current_date: pd.Timestamp
    ) -> pd.DataFrame:
        """获取开放调拨视图（Module6 接口）。

        Args:
            current_date: pandas Timestamp

        Returns:
            同 get_open_deployment_view
        """
        return self.get_open_deployment_view(
            current_date.strftime('%Y-%m-%d')
        )

    def get_open_deployment_view(
        self, date: str
    ) -> pd.DataFrame:
        """获取指定日期的开放调拨视图。

        不触发过期清理；清理只在 run_daily_processing
        开头执行一次。

        Args:
            date: 日期（YYYY-MM-DD）

        Returns:
            含列 [material, sending, receiving,
                  planned_deployment_date,
                  deployed_qty, demand_element,
                  ori_deployment_uid]
        """
        records = []
        for uid, rec in self.open_deployment.items():
            records.append({
                'material': normalize_material(
                    rec['material']
                ),
                'sending': normalize_sending(
                    rec['sending']
                ),
                'receiving': normalize_receiving(
                    rec['receiving']
                ),
                'planned_deployment_date': pd.to_datetime(
                    rec['planned_deployment_date']
                ),
                'deployed_qty': rec['deployed_qty'],
                'demand_element': rec['demand_element'],
                'ori_deployment_uid': uid,
            })
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'material', 'sending', 'receiving',
                'planned_deployment_date',
                'deployed_qty', 'demand_element',
                'ori_deployment_uid',
            ])
        return df

    # ------ 空间额度视图 ------

    def get_space_quota_view(
        self, date: str
    ) -> pd.DataFrame:
        """计算指定日期的可用空间额度。

        公式：capacity - unrestricted_inventory

        Args:
            date: 日期（YYYY-MM-DD）

        Returns:
            含列 [receiving, date, max_qty]
        """
        date_obj = pd.to_datetime(date).normalize()

        if (
            self.space_capacity.empty
            or 'eff_from' not in self.space_capacity.columns
        ):
            return pd.DataFrame(
                columns=['receiving', 'date', 'max_qty']
            )

        mask = (
            (self.space_capacity['eff_from'] <= date_obj)
            & (self.space_capacity['eff_to'] >= date_obj)
        )
        effective = self.space_capacity[mask]

        records = []
        for row in effective.itertuples():
            loc = row.location
            cap = row.capacity
            loc_inv = sum(
                qty
                for (_, l), qty
                in self.unrestricted_inventory.items()
                if l == loc
            )
            records.append({
                'receiving': loc,
                'date': date_obj,
                'max_qty': max(0, cap - loc_inv),
            })

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(
                columns=['receiving', 'date', 'max_qty']
            )
        return df

    # ------ 生产视图 ------

    def get_production_plan_backlog_view(
        self, date: str
    ) -> pd.DataFrame:
        """获取生产计划 backlog（含未来生产）。

        Args:
            date: 参考日期（YYYY-MM-DD）

        Returns:
            含列 [material, location,
                  available_date, quantity]
        """
        cols = [
            'material', 'location',
            'available_date', 'quantity',
        ]
        if not self.production_plan_backlog:
            return pd.DataFrame(columns=cols)

        backlog_df = pd.DataFrame(
            self.production_plan_backlog
        )
        if backlog_df.empty:
            return pd.DataFrame(columns=cols)

        for col in cols:
            if col not in backlog_df.columns:
                backlog_df[col] = ''

        return backlog_df[cols]

    def get_all_production_view(
        self, date: str
    ) -> pd.DataFrame:
        """获取当日 GR + 未来生产计划的合并视图。

        Args:
            date: 日期（YYYY-MM-DD）

        Returns:
            含列 [material, location,
                  available_date, quantity]
        """
        cols = [
            'material', 'location',
            'available_date', 'quantity',
        ]
        date_obj = pd.to_datetime(date).normalize()

        today_gr = self.get_production_gr_view(date)
        if not today_gr.empty:
            today_gr = today_gr.rename(
                columns={'date': 'available_date'}
            )[cols]
        else:
            today_gr = pd.DataFrame(columns=cols)

        future = pd.DataFrame(
            self.production_plan_backlog
        )
        if not future.empty:
            future['available_date'] = (
                pd.to_datetime(future['available_date'])
                .dt.normalize()
            )
            future = future[
                future['available_date'] >= date_obj
            ][cols]
        else:
            future = pd.DataFrame(columns=cols)

        dfs = [
            df for df in [today_gr, future]
            if not df.empty
        ]
        if not dfs:
            return pd.DataFrame(columns=cols)

        out = pd.concat(dfs, ignore_index=True)
        if out.empty:
            return out

        out = out.groupby(
            ['material', 'location', 'available_date'],
            as_index=False,
        ).agg({'quantity': 'sum'})
        out['quantity'] = out['quantity'].astype(int)
        return out

    def get_production_gr_view(
        self, date: str
    ) -> pd.DataFrame:
        """获取指定日期的生产 GR 记录。

        Args:
            date: 日期（YYYY-MM-DD）

        Returns:
            含列 [date, material, location, quantity]
        """
        date_str = (
            pd.to_datetime(date).strftime('%Y-%m-%d')
        )
        records = self.production_gr_by_date.get(
            date_str, []
        )
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'date', 'material',
                'location', 'quantity',
            ])
        return df

    def get_delivery_gr_view(
        self, date: str
    ) -> pd.DataFrame:
        """获取指定日期的交付 GR 记录。

        Args:
            date: 日期（YYYY-MM-DD）

        Returns:
            含列 [date, material, receiving, quantity,
                  ori_deployment_uid, vehicle_uid]
        """
        date_str = (
            pd.to_datetime(date).strftime('%Y-%m-%d')
        )
        records = self.delivery_gr_by_date.get(
            date_str, []
        )
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'date', 'material', 'receiving',
                'quantity', 'ori_deployment_uid',
                'vehicle_uid',
            ])
        return df

    # ------ 发货 / 发运日志视图 ------

    def get_shipment_log_view(
        self, date: str
    ) -> pd.DataFrame:
        """获取指定日期的发货日志记录。

        Args:
            date: 日期（YYYY-MM-DD）

        Returns:
            含列 [date, material, location, quantity]
        """
        date_str = (
            pd.to_datetime(date).strftime('%Y-%m-%d')
        )
        records = self.shipment_log_by_date.get(
            date_str, []
        )
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'date', 'material',
                'location', 'quantity',
            ])
        return df

    def get_delivery_shipment_log_view(
        self, date: str
    ) -> pd.DataFrame:
        """获取指定日期的发运出库日志。

        Args:
            date: 日期（YYYY-MM-DD）

        Returns:
            含列 [date, material, sending, receiving,
                  quantity, ori_deployment_uid,
                  actual_ship_date,
                  actual_delivery_date, type]
        """
        date_str = (
            pd.to_datetime(date).strftime('%Y-%m-%d')
        )
        rows = self.delivery_shipment_log_by_date.get(
            date_str, []
        )
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=[
                'date', 'material', 'sending',
                'receiving', 'quantity',
                'ori_deployment_uid',
                'actual_ship_date',
                'actual_delivery_date', 'type',
            ])
        return df

    # ------ 期初库存视图 ------

    def get_beginning_inventory_view(
        self, date: str
    ) -> pd.DataFrame:
        """获取指定日期的期初库存视图。

        Args:
            date: 日期（YYYY-MM-DD）

        Returns:
            含列 [date, material, location, quantity]
        """
        date_obj = pd.to_datetime(date).normalize()

        if date in self.daily_beginning_inventory:
            inv = self.daily_beginning_inventory[date]
        else:
            inv = self.unrestricted_inventory

        records = []
        for (mat, loc), qty in inv.items():
            records.append({
                'date': date_obj,
                'material': normalize_material(mat),
                'location': normalize_location(loc),
                'quantity': qty,
            })

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'date', 'material',
                'location', 'quantity',
            ])
        return df

    # ------ 汇总统计 ------

    def get_summary_statistics(
        self, date: str
    ) -> dict:
        """获取指定日期的汇总统计。

        Args:
            date: 日期（YYYY-MM-DD）

        Returns:
            汇总统计字典
        """
        date_str = (
            pd.to_datetime(date).strftime('%Y-%m-%d')
        )
        return {
            'date': date,
            'total_inventory_items': len(
                self.unrestricted_inventory
            ),
            'total_inventory_quantity': sum(
                self.unrestricted_inventory.values()
            ),
            'open_deployment_count': len(
                self.open_deployment
            ),
            'in_transit_count': len(self.in_transit),
            'production_gr_count': len(
                self.production_gr_by_date.get(
                    date_str, []
                )
            ),
            'delivery_gr_count': len(
                self.delivery_gr_by_date.get(
                    date_str, []
                )
            ),
            'shipment_count': len(
                self.shipment_log_by_date.get(
                    date_str, []
                )
            ),
        }
