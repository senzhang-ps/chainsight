"""Summary SQL pushdown 回归测试（集成测试，需可达的 PostgreSQL）。

验证改造后的 summary 生成逻辑在以下维度与预期手算结果一致：
- order 去重（DISTINCT 在 quantity/demand_type 维度）
- config_name 提取 simulation_date 常量并全量覆盖
- 三表 outer join（缺失方补 0）
- end_date 过滤（date <= end_date）
- 空数据返回 0
- 透传型 full summary 表的 date 过滤（NULL 保留、超期剔除）
- 透传型 full summary 表在 DB 内 INSERT SELECT，不再经 pandas chunk 搬运

无可达 DB 时整体 skip，不阻塞 CI。所有写入使用唯一 run_id，结束后按 run_id 清理，
不污染既有数据，也不动 module1..6 的真实业务行。
"""
from __future__ import annotations

import unittest
import uuid

import pandas as pd
from psycopg import sql

from pgsql_db.db_connection import DatabaseConnection
from pgsql_db.module_data_writer import ModuleDataWriter


# 测试涉及的表，tearDown 时按 run_id 清理
_SOURCE_TABLES = [
    "module1_output_orderlog",
    "module1_output_shipmentlog",
    "module1_output_cutlog",
    "module4_output_changeoverlog",
    "module4_output_capacityexceed",
    "module4_output_productionplan",
    "module5_output_deploymentplan",
    "module6_output_deliveryplan",
    "module6_output_truckusagelog",
]
_SUMMARY_TABLES = [
    "summary_output_ordershipmentcutsummary",
    "summary_output_fullchangeoverlog",
    "summary_output_fullcapacityexceed",
    "summary_output_fullproductionplan",
    "summary_output_fulldeploymentplan",
    "summary_output_fulldeliveryplan",
    "summary_output_fulltruckusage",
]


class SummarySqlPushdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DatabaseConnection()
        try:
            cls.db.connect()
            cls.db.test_connection()
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"PostgreSQL 不可达，跳过集成测试: {e}")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.db.close()
        except Exception:  # noqa: BLE001
            pass

    def setUp(self):
        self._run_ids: list[str] = []

    def tearDown(self):
        for run_id in self._run_ids:
            for tbl in _SOURCE_TABLES + _SUMMARY_TABLES:
                try:
                    with self.db.get_cursor() as cur:
                        cur.execute(
                            sql.SQL("DELETE FROM {} WHERE run_id = %s").format(
                                sql.Identifier(tbl)
                            ),
                            (run_id,),
                        )
                except Exception:  # noqa: BLE001 表/列不存在等
                    pass

    # ---------- 辅助 ----------
    def _new_run_id(self) -> str:
        run_id = f"__pytest_summary_{uuid.uuid4().hex[:12]}"
        self._run_ids.append(run_id)
        return run_id

    def _insert(self, table: str, df: pd.DataFrame, run_id: str, config_name: str):
        df = df.copy()
        df["run_id"] = run_id
        self.db.create_table_from_df(
            df, table, if_exists="append", config_name=config_name
        )

    def _read_summary(self, table: str, run_id: str) -> pd.DataFrame:
        q = sql.SQL("SELECT * FROM {} WHERE run_id = %s").format(sql.Identifier(table))
        return self.db.execute_query_df(q, (run_id,))

    # ---------- 用例 ----------
    def test_order_shipment_cut_full_scenario(self):
        """去重 + sim_date 常量 + outer join + end_date 过滤 综合。"""
        run_id = self._new_run_id()
        config_name = "TEST_20251224"  # → simulation_date 2025-12-24

        orders = pd.DataFrame({
            "date": ["2025-10-01", "2025-10-01", "2025-10-01", "2025-12-31"],
            "material": ["M1", "M1", "M1", "M1"],
            "location": ["L1", "L1", "L1", "L1"],
            "quantity": [10, 10, 5, 99],  # 行1==行2 → 去重；行4 超 end_date
            "demand_type": ["D1", "D1", "D2", "D1"],
        })
        shipments = pd.DataFrame({
            "date": ["2025-10-01", "2025-10-05"],
            "material": ["M1", "M2"],   # M2 不在 orders → outer join order_qty=0
            "location": ["L1", "L1"],
            "quantity": [8, 3],
        })
        cuts = pd.DataFrame({
            "date": ["2025-10-01"],
            "material": ["M1"],
            "location": ["L1"],
            "quantity": [2],
        })
        self._insert("module1_output_orderlog", orders, run_id, config_name)
        self._insert("module1_output_shipmentlog", shipments, run_id, config_name)
        self._insert("module1_output_cutlog", cuts, run_id, config_name)

        writer = ModuleDataWriter(db=self.db, config_name=config_name)
        n = writer._generate_order_shipment_cut_summary(
            run_id, pd.to_datetime("2025-12-30"), if_exists="replace"
        )
        self.assertEqual(n, 2, "end_date 过滤后应剩 2 行")

        out = self._read_summary("summary_output_ordershipmentcutsummary", run_id)
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values(["date", "material"]).reset_index(drop=True)

        # 行1: 2025-10-01 / M1 / L1 → order 15(=10+5), ship 8
        # cut_qty keeps the original CutLog value (2), not max(order - shipment, 0).
        # （CutLog 里写的 2 被刻意忽略——验证统一口径以本地公式为准）
        self.assertEqual(int(out.loc[0, "order_qty"]), 15)
        self.assertEqual(int(out.loc[0, "shipment_qty"]), 8)
        self.assertEqual(int(out.loc[0, "cut_qty"]), 2)
        # 行2: 2025-10-05 / M2 / L1 → order 0, ship 3, cut max(0-3,0)=0
        self.assertEqual(out.loc[1, "material"], "M2")
        self.assertEqual(int(out.loc[1, "order_qty"]), 0)
        self.assertEqual(int(out.loc[1, "shipment_qty"]), 3)
        self.assertEqual(int(out.loc[1, "cut_qty"]), 0)
        # simulation_date 全量覆盖为 config_name 提取的常量
        sim = pd.to_datetime(out["simulation_date"]).dt.strftime("%Y-%m-%d").unique()
        self.assertEqual(list(sim), ["2025-12-24"])
        # qty 三列必须是整型 dtype（BIGINT），不得回退为浮点
        for col in ("order_qty", "shipment_qty", "cut_qty"):
            self.assertTrue(
                pd.api.types.is_integer_dtype(out[col]),
                f"{col} 应为整型 dtype，实际 {out[col].dtype}",
            )

    def test_empty_returns_zero(self):
        """无任何源数据时返回 0，不建空汇总行。"""
        run_id = self._new_run_id()
        writer = ModuleDataWriter(db=self.db, config_name="TEST_EMPTY")
        n = writer._generate_order_shipment_cut_summary(
            run_id, pd.to_datetime("2025-12-30"), if_exists="replace"
        )
        self.assertEqual(n, 0)

    def test_dedup_collapses_exact_duplicates(self):
        """完全相同的订单行只计一次，不同 quantity 各自保留并求和。"""
        run_id = self._new_run_id()
        config_name = "TEST_20251224"
        orders = pd.DataFrame({
            "date": ["2025-10-01", "2025-10-01", "2025-10-01"],
            "material": ["MX", "MX", "MX"],
            "location": ["LX", "LX", "LX"],
            "quantity": [10, 10, 7],   # 前两行重复 → 去重后 10 + 7 = 17
            "demand_type": ["D1", "D1", "D1"],
        })
        self._insert("module1_output_orderlog", orders, run_id, config_name)

        writer = ModuleDataWriter(db=self.db, config_name=config_name)
        n = writer._generate_order_shipment_cut_summary(
            run_id, pd.to_datetime("2025-12-30"), if_exists="replace"
        )
        self.assertEqual(n, 1)
        out = self._read_summary("summary_output_ordershipmentcutsummary", run_id)
        self.assertEqual(int(out.loc[0, "order_qty"]), 17)
        self.assertEqual(int(out.loc[0, "shipment_qty"]), 0)
        # No CutLog row means the original cut_qty semantics produce 0.
        self.assertEqual(int(out.loc[0, "cut_qty"]), 0)

    def test_order_summary_stays_inside_database_and_replace_is_idempotent(self):
        run_id = self._new_run_id()
        config_name = "TEST_20251224"
        orders = pd.DataFrame({
            "date": ["2025-10-01"],
            "material": ["MZ"],
            "location": ["LZ"],
            "quantity": [4],
        })
        self._insert("module1_output_orderlog", orders, run_id, config_name)

        writer = ModuleDataWriter(db=self.db, config_name=config_name)
        original_execute_query_df = self.db.execute_query_df

        def _fail_if_dataframe_fetch_is_used(*args, **kwargs):
            raise AssertionError("summary generation should not fetch aggregate DataFrames")

        self.db.execute_query_df = _fail_if_dataframe_fetch_is_used
        try:
            n1 = writer._generate_order_shipment_cut_summary(
                run_id, pd.to_datetime("2025-12-30"), if_exists="replace"
            )
            n2 = writer._generate_order_shipment_cut_summary(
                run_id, pd.to_datetime("2025-12-30"), if_exists="replace"
            )
        finally:
            self.db.execute_query_df = original_execute_query_df

        self.assertEqual(n1, 1)
        self.assertEqual(n2, 1)
        out = self._read_summary("summary_output_ordershipmentcutsummary", run_id)
        self.assertEqual(len(out), 1)
        self.assertEqual(int(out.loc[0, "order_qty"]), 4)

    def test_truck_usage_date_filter(self):
        """透传表 date 过滤：NULL 保留、超 end_date 剔除、行数与预期一致。"""
        run_id = self._new_run_id()
        config_name = "TEST_TRUCK"
        usage = pd.DataFrame({
            "date": pd.to_datetime(["2025-10-01", "2025-12-31", None]),
            "truck_id": ["T1", "T2", "T3"],
            "load": [100, 200, 300],
        })
        self._insert("module6_output_truckusagelog", usage, run_id, config_name)

        writer = ModuleDataWriter(db=self.db, config_name=config_name)
        n = writer._generate_truck_usage_summary(
            run_id, pd.to_datetime("2025-12-30"), if_exists="replace"
        )
        # 2025-12-31 超期剔除；NULL 与 2025-10-01 保留
        self.assertEqual(n, 2)
        out = self._read_summary("summary_output_fulltruckusage", run_id)
        kept = set(out["truck_id"].tolist())
        self.assertEqual(kept, {"T1", "T3"})
        self.assertNotIn("T2", kept)

    def test_full_summary_tables_use_insert_select_without_pandas_chunks(self):
        """6 张 full summary 表应在 DB 内透传生成，不再通过 pandas chunk 写入。"""
        run_id = self._new_run_id()
        config_name = "TEST_FULL_SQL_ONLY"

        cases = [
            (
                "module4_output_changeoverlog",
                "summary_output_fullchangeoverlog",
                "_generate_changeover_summary",
                pd.DataFrame({
                    "changeover_end_date": pd.to_datetime(["2025-10-01", "2025-12-31", None]),
                    "record_id": ["change_ok", "change_late", "change_null"],
                }),
            ),
            (
                "module4_output_capacityexceed",
                "summary_output_fullcapacityexceed",
                "_generate_capacity_exceed_summary",
                pd.DataFrame({
                    "date": pd.to_datetime(["2025-10-01", "2025-12-31", None]),
                    "record_id": ["capacity_ok", "capacity_late", "capacity_null"],
                }),
            ),
            (
                "module4_output_productionplan",
                "summary_output_fullproductionplan",
                "_generate_production_plan_summary",
                pd.DataFrame({
                    "available_date": pd.to_datetime(["2025-10-01", "2025-12-31", None]),
                    "record_id": ["production_ok", "production_late", "production_null"],
                }),
            ),
            (
                "module5_output_deploymentplan",
                "summary_output_fulldeploymentplan",
                "_generate_deployment_plan_summary",
                pd.DataFrame({
                    "deployment_date": pd.to_datetime(["2025-10-01", "2025-12-31", None]),
                    "record_id": ["deployment_ok", "deployment_late", "deployment_null"],
                }),
            ),
            (
                "module6_output_deliveryplan",
                "summary_output_fulldeliveryplan",
                "_generate_delivery_plan_summary",
                pd.DataFrame({
                    "actual_ship_date": pd.to_datetime(["2025-10-01", "2025-12-31", None]),
                    "record_id": ["delivery_ok", "delivery_late", "delivery_null"],
                }),
            ),
            (
                "module6_output_truckusagelog",
                "summary_output_fulltruckusage",
                "_generate_truck_usage_summary",
                pd.DataFrame({
                    "date": pd.to_datetime(["2025-10-01", "2025-12-31", None]),
                    "record_id": ["truck_ok", "truck_late", "truck_null"],
                }),
            ),
        ]

        for source_table, _, _, df in cases:
            self._insert(source_table, df, run_id, config_name)

        writer = ModuleDataWriter(db=self.db, config_name=config_name)
        original_iter_query_chunks = self.db.iter_query_chunks
        original_create_table_from_df = self.db.create_table_from_df

        def _fail_iter_query_chunks(*args, **kwargs):
            raise AssertionError("full summary should not read pandas chunks")

        def _fail_create_table_from_df(*args, **kwargs):
            raise AssertionError("full summary should not write pandas DataFrames")

        self.db.iter_query_chunks = _fail_iter_query_chunks
        self.db.create_table_from_df = _fail_create_table_from_df
        try:
            for _, target_table, method_name, _ in cases:
                generate = getattr(writer, method_name)
                n1 = generate(run_id, pd.to_datetime("2025-12-30"), if_exists="replace")
                n2 = generate(run_id, pd.to_datetime("2025-12-30"), if_exists="replace")
                self.assertEqual(n1, 2)
                self.assertEqual(n2, 2)

                out = self._read_summary(target_table, run_id)
                self.assertEqual(len(out), 2)
                kept = set(out["record_id"].tolist())
                self.assertTrue(any(record.endswith("_ok") for record in kept))
                self.assertTrue(any(record.endswith("_null") for record in kept))
                self.assertFalse(any(record.endswith("_late") for record in kept))
                self.assertEqual(set(out["config_name"].tolist()), {config_name})
                self.assertFalse(out["db_write_time"].isna().any())
        finally:
            self.db.iter_query_chunks = original_iter_query_chunks
            self.db.create_table_from_df = original_create_table_from_df

    def test_passthrough_overrides_config_name_and_isolates_runs(self):
        """SQL-only 透传：config_name 用当前 writer 覆盖源表旧值；replace 只删当前 run_id。"""
        run_a = self._new_run_id()
        run_b = self._new_run_id()
        usage_a = pd.DataFrame({
            "date": pd.to_datetime(["2025-10-01", "2025-10-02"]),
            "truck_id": ["A1", "A2"],
        })
        usage_b = pd.DataFrame({
            "date": pd.to_datetime(["2025-10-03"]),
            "truck_id": ["B1"],
        })
        # 源表里写入"旧"的 config_name，summary 应改用 writer 的新 config_name
        self._insert("module6_output_truckusagelog", usage_a, run_a, "SOURCE_OLD_CFG")
        self._insert("module6_output_truckusagelog", usage_b, run_b, "SOURCE_OLD_CFG")

        writer = ModuleDataWriter(db=self.db, config_name="WRITER_NEW_CFG")
        n_a = writer._generate_truck_usage_summary(
            run_a, pd.to_datetime("2025-12-30"), if_exists="replace"
        )
        n_b = writer._generate_truck_usage_summary(
            run_b, pd.to_datetime("2025-12-30"), if_exists="replace"
        )
        self.assertEqual(n_a, 2)
        self.assertEqual(n_b, 1)

        out_a = self._read_summary("summary_output_fulltruckusage", run_a)
        # config_name 覆盖为 writer 的值，而非源表的 SOURCE_OLD_CFG
        self.assertEqual(set(out_a["config_name"]), {"WRITER_NEW_CFG"})
        self.assertTrue(out_a["db_write_time"].notna().all())
        self.assertEqual(set(out_a["run_id"]), {run_a})

        # 重新生成 run_a（replace）：run_b 不被误删，run_a 不重复累加
        writer._generate_truck_usage_summary(
            run_a, pd.to_datetime("2025-12-30"), if_exists="replace"
        )
        out_b = self._read_summary("summary_output_fulltruckusage", run_b)
        self.assertEqual(len(out_b), 1)
        self.assertEqual(set(out_b["truck_id"]), {"B1"})
        out_a2 = self._read_summary("summary_output_fulltruckusage", run_a)
        self.assertEqual(len(out_a2), 2)

    def test_passthrough_returns_zero_when_no_source_rows(self):
        """源表无该 run_id 数据时返回 0，不写入垃圾行。"""
        run_id = self._new_run_id()
        writer = ModuleDataWriter(db=self.db, config_name="TEST_NO_ROWS")
        n = writer._generate_capacity_exceed_summary(
            run_id, pd.to_datetime("2025-12-30"), if_exists="replace"
        )
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
