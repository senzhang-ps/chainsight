# inventory_log.py
# 库存变动日志生成 (Mixin)
#
# 生成每日库存变动明细，用于平衡检查和审计

import pandas as pd


class OrchestratorInventoryLogMixin:
    """库存变动日志生成方法集合（Mixin）。"""

    @staticmethod
    def _stable_sort_inventory_change_log(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        sort_cols = [
            c for c in [
                'date', 'material', 'location', 'beginning_inventory',
                'production_gr', 'delivery_gr', 'shipment',
                'delivery_ship', 'ending_inventory',
                'calculated_ending', 'balance_diff',
            ] if c in df.columns
        ]
        if not sort_cols:
            return df.reset_index(drop=True)

        return df.sort_values(
            by=sort_cols,
            kind='mergesort',
        ).reset_index(drop=True)

    def generate_inventory_change_log(
        self, date: str
    ) -> pd.DataFrame:
        """生成指定日期的库存变动日志。

        记录每个物料-地点的完整库存变动：
        期初、入库、出库、期末

        Args:
            date: 日期字符串（YYYY-MM-DD）

        Returns:
            库存变动日志 DataFrame
        """
        date_obj = pd.to_datetime(date).normalize()

        # 获取所有涉及的物料-地点组合
        all_keys = set()

        # 从期初和期末库存获取
        if date in self.daily_beginning_inventory:
            all_keys.update(
                self.daily_beginning_inventory[
                    date
                ].keys()
            )
        if date in self.daily_ending_inventory:
            all_keys.update(
                self.daily_ending_inventory[
                    date
                ].keys()
            )

        # 从各种变动记录获取
        date_str = date_obj.strftime('%Y-%m-%d')

        for record in (
            self.production_gr_by_date.get(
                date_str, []
            )
        ):
            all_keys.add(
                (record['material'], record['location'])
            )

        for record in (
            self.delivery_gr_by_date.get(date_str, [])
        ):
            all_keys.add(
                (
                    record['material'],
                    record['receiving'],
                )
            )

        for record in (
            self.shipment_log_by_date.get(
                date_str, []
            )
        ):
            all_keys.add(
                (record['material'], record['location'])
            )

        # 从内存获取发运出库数据
        delivery_ship_data = {}
        for record in (
            self.delivery_shipment_log_by_date.get(
                date_str, []
            )
        ):
            material = record['material']
            sending = record['sending']
            quantity = float(record['quantity'])

            key = (material, sending)
            old_val = delivery_ship_data.get(key, 0)
            delivery_ship_data[key] = (
                old_val + quantity
            )
            all_keys.add(key)

        msg = (
            f"  📊 从内存获取发运出库 [{date}]: "
            f"{len(delivery_ship_data)} 项"
        )
        print(msg)

        change_log = []

        for material, location in all_keys:
            # 期初库存
            beginning_qty = 0
            if date in self.daily_beginning_inventory:
                beginning_qty = (
                    self.daily_beginning_inventory[
                        date
                    ].get((material, location), 0)
                )

            # 生产入库
            production_qty = sum(
                record['quantity']
                for record in (
                    self.production_gr_by_date.get(
                        date_str, []
                    )
                )
                if record['material'] == material
                and record['location'] == location
            )

            # 交付入库
            delivery_qty = sum(
                record['quantity']
                for record in (
                    self.delivery_gr_by_date.get(
                        date_str, []
                    )
                )
                if record['material'] == material
                and record['receiving'] == location
            )

            # 发货出库
            shipment_qty = sum(
                record['quantity']
                for record in (
                    self.shipment_log_by_date.get(
                        date_str, []
                    )
                )
                if record['material'] == material
                and record['location'] == location
            )

            # 发运出库（从内存获取）
            delivery_ship_qty = (
                delivery_ship_data.get(
                    (material, location), 0
                )
            )

            # 期末库存
            ending_qty = 0
            if date in self.daily_ending_inventory:
                ending_qty = (
                    self.daily_ending_inventory[
                        date
                    ].get((material, location), 0)
                )

            # 只记录有变动的记录
            if (
                beginning_qty != 0
                or production_qty != 0
                or delivery_qty != 0
                or shipment_qty != 0
                or delivery_ship_qty != 0
                or ending_qty != 0
            ):

                # 应用负库存重置逻辑
                calculated_ending = (
                    beginning_qty
                    + production_qty
                    + delivery_qty
                    - shipment_qty
                    - delivery_ship_qty
                )
                if calculated_ending < 0:
                    calculated_ending = 0

                change_log.append({
                    'date': date_obj,
                    'material': material,
                    'location': location,
                    'beginning_inventory': (
                        beginning_qty
                    ),
                    'production_gr': production_qty,
                    'delivery_gr': delivery_qty,
                    'shipment': shipment_qty,
                    'delivery_ship': delivery_ship_qty,
                    'ending_inventory': ending_qty,
                    'calculated_ending': (
                        calculated_ending
                    ),
                    'balance_diff': (
                        ending_qty - calculated_ending
                    ),
                })

        df = pd.DataFrame(change_log)
        if df.empty:
            df = pd.DataFrame(columns=[
                'date',
                'material',
                'location',
                'beginning_inventory',
                'production_gr',
                'delivery_gr',
                'shipment',
                'delivery_ship',
                'ending_inventory',
                'calculated_ending',
                'balance_diff',
            ])

        return self._stable_sort_inventory_change_log(df)
