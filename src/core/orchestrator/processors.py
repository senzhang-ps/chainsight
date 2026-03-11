"""编排器模块结果处理集合。

职责：
- 接收各业务模块产出的当日结果。
- 将结果转换为统一状态更新，回写库存、开放调拨、在途与日志。
- 保证模块输出在 Orchestrator 中的落账口径保持一致。
"""

import pandas as pd

from .models import DeploymentUID
from .normalize import (
    _normalize_location,
    _normalize_material,
    _normalize_receiving,
    _normalize_sending,
)


class OrchestratorProcessorsMixin:
    """模块处理方法集合（Mixin）。"""

    def process_module1_shipments(
        self, shipment_df: pd.DataFrame, date: str
    ):
        """处理指定日期的 Module1 发货数据。

        Args:
            shipment_df: 含列 [date, material,
                        location, quantity]
            date: 仿真日期（YYYY-MM-DD）
        """
        date_obj = pd.to_datetime(date).normalize()

        # 筛选当日发货记录
        mask = (
            pd.to_datetime(shipment_df['date'])
            .dt.normalize()
            == date_obj
        )
        daily_shipments = shipment_df[mask]

        # 更新非限制库存
        for row in daily_shipments.itertuples():
            key = (
                _normalize_material(row.material),
                _normalize_location(row.location),
            )
            if key in self.unrestricted_inventory:
                old_qty = self.unrestricted_inventory[key]
                new_qty = max(
                    0, old_qty - int(row.quantity)
                )
                self.unrestricted_inventory[key] = new_qty

            # 记录发货日志
            record = {
                'date': date_obj,
                'material': _normalize_material(
                    row.material
                ),
                'location': _normalize_location(
                    row.location
                ),
                'quantity': int(row.quantity),
                'type': 'customer_shipment',
            }
            self.shipment_log.append(record)
            # 索引以便 O(1) 查询
            date_str = date_obj.strftime('%Y-%m-%d')
            if date_str not in (
                self.shipment_log_by_date
            ):
                self.shipment_log_by_date[date_str] = []
            self.shipment_log_by_date[date_str].append(
                record
            )

        if len(daily_shipments) > 0:
            msg = (
                f"Processed {len(daily_shipments)} "
                f"shipments"
            )
            print(f"✅ {msg} for {date}")
            self._log_event("M1_SHIPMENTS", msg)

    def process_module4_production(
        self, production_df: pd.DataFrame, date: str
    ):
        """处理指定日期的 Module4 生产数据。

        Args:
            production_df: 含列 [available_date,
                           material, location,
                           produced_qty]
            date: 仿真日期（YYYY-MM-DD）
        """
        date_obj = pd.to_datetime(date).normalize()
        # === A) 缓存当日GR的生产计划到 backlog 中
        if (
            production_df is not None
            and not production_df.empty
        ):
            tmp = production_df.copy()
            # 标准列名
            if 'available_date' in tmp.columns:
                tmp['available_date'] = (
                    pd.to_datetime(
                        tmp['available_date']
                    ).dt.normalize()
                )
            if (
                'quantity' not in tmp.columns
                and 'produced_qty' in tmp.columns
            ):
                tmp = tmp.rename(
                    columns={'produced_qty': 'quantity'}
                )
            keep = [
                'material',
                'location',
                'available_date',
                'quantity',
            ]
            tmp = tmp[keep].copy()
            tmp['material'] = tmp['material'].astype(str)
            tmp['location'] = tmp['location'].apply(
                _normalize_location
            )
            tmp['quantity'] = (
                tmp['quantity'].fillna(0).astype(int)
            )

            # 新增：用于精确去重的维度
            tmp['simulation_date'] = date_obj
            if 'production_plan_date' in (
                production_df.columns
            ):
                tmp['production_plan_date'] = (
                    pd.to_datetime(
                        production_df[
                            'production_plan_date'
                        ]
                    ).dt.normalize()
                )
            else:
                tmp['production_plan_date'] = tmp[
                    'available_date'
                ]
            # 追加到 backlog
            if self.production_plan_backlog:
                existing_df = pd.DataFrame(
                    self.production_plan_backlog
                )
            else:
                existing_df = pd.DataFrame()
            # 确保旧记录具备新字段
            for col in [
                'simulation_date',
                'production_plan_date',
            ]:
                if col not in existing_df.columns:
                    existing_df[col] = pd.NaT

            combined = pd.concat(
                [existing_df, tmp], ignore_index=True
            )

            # 第一阶段：按 5 维去重
            combined = combined.drop_duplicates(
                subset=[
                    'material',
                    'location',
                    'simulation_date',
                    'production_plan_date',
                    'available_date',
                ],
                keep='first',
            )

            # 第二阶段：按 3 维汇总数量
            aggregated = combined.groupby(
                ['material', 'location', 'available_date'],
                as_index=False,
            ).agg({'quantity': 'sum'})
            aggregated['quantity'] = (
                aggregated['quantity']
                .fillna(0)
                .astype(int)
            )

            self.production_plan_backlog = (
                aggregated.to_dict('records')
            )

        # === B) 只对"今天到货"的进行 GR 入库
        mask = (
            pd.to_datetime(
                production_df['available_date']
            ).dt.normalize()
            == date_obj
        )
        daily_production = production_df[mask]

        # 更新非限制库存并记录生产 GR
        for row in daily_production.itertuples():
            key = (
                _normalize_material(row.material),
                _normalize_location(row.location),
            )
            quantity = int(row.produced_qty)

            old = self.unrestricted_inventory.get(
                key, 0
            )
            self.unrestricted_inventory[key] = (
                old + quantity
            )

            # 记录生产 GR
            record = {
                'date': date_obj,
                'material': _normalize_material(
                    row.material
                ),
                'location': _normalize_location(
                    row.location
                ),
                'quantity': quantity,
            }
            self.production_gr.append(record)
            # 索引以便 O(1) 查询
            date_str = date_obj.strftime('%Y-%m-%d')
            if date_str not in (
                self.production_gr_by_date
            ):
                self.production_gr_by_date[date_str] = []
            self.production_gr_by_date[date_str].append(
                record
            )

        if len(daily_production) > 0:
            msg = (
                f"Processed {len(daily_production)} "
                f"production receipts"
            )
            print(f"✅ {msg} for {date}")
            self._log_event("M4_PRODUCTION", msg)

    def process_module5_deployment(
        self, deployment_df: pd.DataFrame, date: str
    ):
        """处理 Module5 部署计划并更新开放调拨。

        Args:
            deployment_df: 含列 [material, sending,
                                receiving,
                                planned_deployment_date,
                                deployed_qty,
                                demand_element]
            date: 仿真日期（YYYY-MM-DD）
        """
        date_obj = pd.to_datetime(date).normalize()

        # 为保证可复现，在生成 UID 之前稳定排序
        sort_cols = [
            col
            for col in [
                'material',
                'sending',
                'receiving',
                'planned_deployment_date',
                'demand_element',
                'deployed_qty',
            ]
            if col in deployment_df.columns
        ]
        if sort_cols:
            deployment_df = deployment_df.sort_values(
                by=sort_cols, kind='mergesort'
            )
        # 将新的部署计划加入开放调拨
        for row in deployment_df.itertuples():
            # 生成唯一 UID
            self.uid_sequence += 1
            pdd = pd.to_datetime(
                row.planned_deployment_date
            ).strftime('%Y-%m-%d')
            uid_obj = DeploymentUID(
                material=str(row.material),
                sending=str(row.sending),
                receiving=str(row.receiving),
                planned_deploy_date=pdd,
                demand_element=str(row.demand_element),
                sequence=self.uid_sequence,
            )
            uid = uid_obj.to_string()

            converted_qty = self._safe_convert_to_int(
                row.deployed_qty
            )

            self.open_deployment[uid] = {
                'material': _normalize_material(
                    row.material
                ),
                'sending': _normalize_sending(
                    row.sending
                ),
                'receiving': _normalize_receiving(
                    row.receiving
                ),
                'planned_deployment_date': pdd,
                'deployed_qty': converted_qty,
                'demand_element': str(
                    row.demand_element
                ),
                'creation_date': date_obj.strftime(
                    '%Y-%m-%d'
                ),
            }

        if len(deployment_df) > 0:
            msg = (
                f"Added {len(deployment_df)} "
                f"deployment plans"
            )
            print(f"✅ {msg} for {date}")
            self._log_event("M5_DEPLOYMENT", msg)

    def process_module6_delivery(
        self, delivery_df: pd.DataFrame, date: str
    ):
        """处理 Module6 交付计划并更新状态。

        Args:
            delivery_df: 含列 [ori_deployment_uid,
                              material, sending,
                              receiving,
                              actual_ship_date,
                              actual_delivery_date,
                              delivery_qty]
            date: 仿真日期（YYYY-MM-DD）
        """
        date_obj = pd.to_datetime(date).normalize()
        msg = (
            f"[M6->Orch] incoming rows: "
            f"{len(delivery_df)}; date={date}"
        )
        print(msg)

        # 处理每条交付记录
        for row in delivery_df.itertuples():
            uid = str(row.ori_deployment_uid)
            vehicle_uid = str(row.vehicle_uid)
            material = str(row.material)
            sending = str(row.sending)
            receiving = str(row.receiving)

            norm_mat = _normalize_material(material)
            norm_rcv = _normalize_receiving(receiving)
            ship_date = pd.to_datetime(
                row.actual_ship_date
            )
            delivery_date = pd.to_datetime(
                row.actual_delivery_date
            )
            quantity = self._safe_convert_to_int(
                row.delivery_qty
            )

            # 只处理当天发运
            if ship_date.normalize() != date_obj:
                continue

            # 减少开放调拨数量
            if uid in self.open_deployment:
                old_qty = self.open_deployment[uid][
                    'deployed_qty'
                ]
                self.open_deployment[uid][
                    'deployed_qty'
                ] = (old_qty - quantity)
                if (
                    self.open_deployment[uid][
                        'deployed_qty'
                    ]
                    <= 0
                ):
                    del self.open_deployment[uid]

            # 减少发货地非限制库存
            sending_key = (
                _normalize_material(material),
                _normalize_location(sending),
            )
            if sending_key in (
                self.unrestricted_inventory
            ):
                old_inv = self.unrestricted_inventory[
                    sending_key
                ]
                self.unrestricted_inventory[
                    sending_key
                ] = max(0, old_inv - quantity)

            # 记录发运出库日志
            shipment_record = {
                'date': date_obj,
                'material': norm_mat,
                'sending': _normalize_sending(sending),
                'receiving': norm_rcv,
                'quantity': quantity,
                'ori_deployment_uid': uid,
                'actual_ship_date': ship_date.strftime(
                    '%Y-%m-%d'
                ),
                'actual_delivery_date': (
                    delivery_date.strftime('%Y-%m-%d')
                ),
                'type': 'delivery_shipment',
            }
            self.delivery_shipment_log.append(
                shipment_record
            )
            # 索引以便 O(1) 查询
            date_str = date_obj.strftime('%Y-%m-%d')
            if date_str not in (
                self.delivery_shipment_log_by_date
            ):
                self.delivery_shipment_log_by_date[
                    date_str
                ] = []
            self.delivery_shipment_log_by_date[
                date_str
            ].append(shipment_record)

            # 判断处理逻辑
            if delivery_date.normalize() > date_obj:
                # 为未来交付创建在途记录
                transit_uid = (
                    f"{uid}_transit_{vehicle_uid}"
                )
                self.in_transit[transit_uid] = {
                    'material': norm_mat,
                    'sending': _normalize_sending(
                        sending
                    ),
                    'receiving': norm_rcv,
                    'actual_ship_date': (
                        ship_date.strftime('%Y-%m-%d')
                    ),
                    'actual_delivery_date': (
                        delivery_date.strftime(
                            '%Y-%m-%d'
                        )
                    ),
                    'quantity': quantity,
                    'ori_deployment_uid': uid,
                    'vehicle_uid': vehicle_uid,
                }
            elif delivery_date.normalize() == date_obj:
                # 当天交付：创建 delivery GR
                receiving_key = (material, receiving)
                old_inv = (
                    self.unrestricted_inventory.get(
                        receiving_key, 0
                    )
                )
                self.unrestricted_inventory[
                    receiving_key
                ] = (old_inv + quantity)

                # 记录 delivery GR（含去重检查）
                gr_record = {
                    'date': date_obj,
                    'material': norm_mat,
                    'receiving': norm_rcv,
                    'quantity': quantity,
                    'ori_deployment_uid': uid,
                    'vehicle_uid': vehicle_uid,
                }

                # 基于关键字段检查重复
                existing_key = (
                    date_obj,
                    material,
                    receiving,
                    uid,
                    vehicle_uid,
                )
                is_duplicate = any(
                    (
                        record['date'],
                        record['material'],
                        record['receiving'],
                        record['ori_deployment_uid'],
                        record['vehicle_uid'],
                    )
                    == existing_key
                    for record in self.delivery_gr
                )

                if not is_duplicate:
                    self.delivery_gr.append(gr_record)
                    # 索引以便 O(1) 查询
                    date_str = date_obj.strftime(
                        '%Y-%m-%d'
                    )
                    if date_str not in (
                        self.delivery_gr_by_date
                    ):
                        self.delivery_gr_by_date[
                            date_str
                        ] = []
                    self.delivery_gr_by_date[
                        date_str
                    ].append(gr_record)

        if len(delivery_df) > 0:
            msg = (
                f"Processed {len(delivery_df)} "
                f"delivery plans"
            )
            print(f"✅ {msg} for {date}")
            self._log_event("M6_DELIVERY", msg)

    def process_delivery_plan(
        self,
        delivery_plan_df: pd.DataFrame,
        simulation_date: pd.Timestamp,
    ):
        """处理交付计划（Module6 接口）。

        Args:
            delivery_plan_df: 交付计划 DataFrame
            simulation_date: 当前仿真日期
        """
        self.process_module6_delivery(
            delivery_plan_df,
            simulation_date.strftime('%Y-%m-%d'),
        )
