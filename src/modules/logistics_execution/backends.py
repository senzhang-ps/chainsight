"""Module6 pandas / Polars 计算后端。

pandas 后端将既有函数式 M6 的准备、仿真与输出语义收敛到新门面。Polars
后端当前具有独立的静态/动态 DataFrame 边界和配置准备实现；装车算法属于严格
顺序敏感的业务流程，后续 parity 阶段将在该后端中逐步替换为原生 Polars 路线循环。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ...utils.df_convert import pandas_to_polars
from ...utils.normalization import normalize_identifiers
from .capacity_manager import build_capacity_map, normalize_capacity_plan
from .delivery_processor import (
    calculate_actual_delivery_date, calculate_lead_time, create_bypass_record,
    create_delivery_record, create_unsatisfied_record, sample_delivery_delay,
    should_bypass_mdq,
)
from .expression_evaluator import SafeExpressionEvaluator
from .inventory_manager import calculate_inventory_limit, update_inventory_after_load
from .output_writer import generate_outputs
from .simulation import run_simulation_loop
from .vehicle_packer import VehiclePacker, create_vehicle_log_entry, determine_trigger_cause, get_representative_context
from .validators import (
    check_and_deduplicate,
    validate_deployment_plan,
    validate_priority_mapping,
    validate_threshold_config,
    validate_truck_config,
    validate_truck_specs,
)


class _PandasBackend:
    """保持既有 M6 业务语义的 pandas 正确性基线。"""

    engine = "pandas"

    def __init__(self, owner=None):
        self.owner = owner
        self.static: dict[str, pd.DataFrame] = {}
        self.result: dict[str, Any] = {}

    @staticmethod
    def _empty_frame(columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=columns)

    def empty_result(self) -> dict[str, Any]:
        return {
            "delivery_plan": pd.DataFrame(),
            "vehicle_log": self._empty_frame([
                "date", "sending", "receiving", "truck_type", "vehicle_no",
                "vehicle_uid", "total_units", "total_weight", "total_volume",
                "WFR", "VFR", "trigger",
            ]),
            "truck_usage": self._empty_frame([
                "date", "sending", "receiving", "truck_type", "truck_used",
            ]),
            "unsatisfied_log": pd.DataFrame(),
            "validation_log": pd.DataFrame(),
            "bypass_log": pd.DataFrame(),
            "statistics": {
                "delivery_count": 0,
                "vehicle_count": 0,
                "unsatisfied_count": 0,
                "bypass_count": 0,
            },
        }

    @staticmethod
    def _normalise(frame: pd.DataFrame | None) -> pd.DataFrame:
        if frame is None:
            return pd.DataFrame()
        return normalize_identifiers(frame.copy()) if not frame.empty else frame.copy()

    def normalise_static_config(self, datas: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        mapping = {
            "TruckReleaseCon": "M6_TruckReleaseCon",
            "TruckCapacityPlan": "M6_TruckCapacityPlan",
            "TruckTypeSpecs": "M6_TruckTypeSpecs",
            "MaterialMD": "M6_MaterialMD",
            "DeliveryDelayDistribution": "M6_DeliveryDelayDistribution",
            "MDQBypassRules": "M6_MDQBypassRules",
            "DemandPriority": "Global_DemandPriority",
            "LeadTime": "Global_LeadTime",
        }
        static = {name: self._normalise(datas.get(source)) for name, source in mapping.items()}
        for name, columns in {
            "TruckReleaseCon": {"wfr": "WFR", "vfr": "VFR"},
            "LeadTime": {"pdt": "PDT", "gr": "GR", "otd": "OTD"},
        }.items():
            frame = static[name]
            static[name] = frame.rename(columns={
                source: target
                for source, target in columns.items()
                if source in frame and target not in frame
            })
        for frame_name, columns in {
            "TruckCapacityPlan": ("date", "eff_from", "eff_to"),
            "DeliveryDelayDistribution": ("date",),
        }.items():
            frame = static[frame_name]
            for column in columns:
                if column in frame:
                    frame[column] = pd.to_datetime(
                        frame[column], errors="coerce", format="mixed"
                    )
        return static

    @staticmethod
    def validate_static_config(static: dict[str, pd.DataFrame]) -> None:
        required = ("TruckReleaseCon", "TruckCapacityPlan", "TruckTypeSpecs", "MaterialMD",
                    "DeliveryDelayDistribution", "MDQBypassRules", "DemandPriority", "LeadTime")
        missing = [name for name in required if name not in static]
        if missing:
            raise ValueError(f"ModuleSix 缺少静态配置: {missing}")

    def store_static_state(self, static: dict[str, pd.DataFrame]) -> None:
        self.static = static

    def load_daily_inputs(self, day: pd.Timestamp) -> dict[str, pd.DataFrame]:
        context = self.owner.state_context
        date_text = day.strftime("%Y-%m-%d")
        return {
            "DeploymentPlan": self._normalise(context.get_open_deployment_view(date_text)),
            "Inventory": self._normalise(context.get_unrestricted_inventory_view(date_text)),
        }

    @staticmethod
    def _build_priority_map(demand_prio: pd.DataFrame) -> dict[str, int]:
        if demand_prio.empty:
            return {}
        return demand_prio.set_index("demand_element")["priority"].to_dict()

    @staticmethod
    def _build_material_map(material_md: pd.DataFrame) -> dict[str, dict[str, float]]:
        if material_md.empty:
            return {}
        return material_md.set_index("material")[
            ["demand_unit_to_weight", "demand_unit_to_volume"]
        ].to_dict("index")

    @staticmethod
    def _build_spec_map(truck_specs: pd.DataFrame) -> dict[str, dict[str, Any]]:
        if truck_specs.empty:
            return {}
        return truck_specs.set_index("truck_type").to_dict("index")

    @staticmethod
    def _log_missing_materials(missing: pd.DataFrame, validation_log: list[dict]) -> None:
        for material in missing["material"].unique():
            count = int((missing["material"] == material).sum())
            validation_log.append({
                "sheet": "M6_MaterialMD", "row": "",
                "issue": f'Missing material metadata for "{material}" (affects {count} records). '
                         "Default unit conversion factors (1.0) will be used.",
                "severity": "WARNING", "impact": f"Default Values Used - {count} records",
                "missing_material": material, "affected_records": count,
            })

    def _process_material_metadata(self, dp: pd.DataFrame, mat_map: dict, validation_log: list[dict]) -> pd.DataFrame:
        if dp.empty:
            return dp
        missing = dp.loc[~dp["material"].isin(mat_map.keys())]
        if not missing.empty:
            self._log_missing_materials(missing, validation_log)
            dp["demand_unit_to_weight"] = dp["material"].map(
                lambda value: mat_map.get(value, {}).get("demand_unit_to_weight", 1.0)
            )
            dp["demand_unit_to_volume"] = dp["material"].map(
                lambda value: mat_map.get(value, {}).get("demand_unit_to_volume", 1.0)
            )
            return dp
        metadata = pd.DataFrame([{"material": material, **values} for material, values in mat_map.items()])
        dp = dp.merge(metadata, on="material", how="left")
        dp["demand_unit_to_weight"] = dp["demand_unit_to_weight"].fillna(1.0)
        dp["demand_unit_to_volume"] = dp["demand_unit_to_volume"].fillna(1.0)
        return dp

    @staticmethod
    def _prepare_deployment_plan(dp: pd.DataFrame, priority_map: dict[str, int]) -> pd.DataFrame:
        if dp.empty:
            return dp
        dp["planned_deployment_date"] = pd.to_datetime(dp["planned_deployment_date"])
        sort_columns = [column for column in (
            "planned_deployment_date", "sending", "receiving", "material", "demand_element"
        ) if column in dp]
        if sort_columns:
            dp = dp.sort_values(sort_columns, kind="mergesort")
        dp = dp.reset_index(drop=True)
        if "ori_deployment_uid" not in dp or dp["ori_deployment_uid"].isnull().any():
            dp["ori_deployment_uid"] = [f"UID{index:06d}" for index in dp.index]
        dp["priority"] = dp["demand_element"].map(priority_map)
        dp["waiting_days"] = 0
        dp["simulation_date"] = dp["planned_deployment_date"]
        dp["route_type_debug"] = np.where(dp["sending"] == dp["receiving"], "self_loop", "cross_node")
        return dp

    @staticmethod
    def _handle_uid_duplicates(dp: pd.DataFrame, validation_log: list[dict]) -> pd.DataFrame:
        if dp.empty or not dp["ori_deployment_uid"].duplicated().any():
            return dp
        duplicate_mask = dp.duplicated(subset=["ori_deployment_uid"], keep=False)
        duplicate_count = int(duplicate_mask.sum())
        uid_count = int(dp.loc[duplicate_mask, "ori_deployment_uid"].nunique())
        validation_log.append({
            "sheet": "DeploymentPlan", "row": "",
            "issue": f"Found {uid_count} duplicate ori_deployment_uid values. Deduplicating by keeping first occurrence.",
            "severity": "ERROR", "impact": f"Data Deduplication - {duplicate_count - uid_count} removed",
            "duplicate_uids": uid_count,
        })
        return dp.drop_duplicates(subset=["ori_deployment_uid"], keep="first")

    def prepare_daily_data(self, inputs: dict[str, pd.DataFrame], day: pd.Timestamp) -> tuple[dict[str, Any], dict[str, Any]]:
        validation_log: list[dict] = []
        dp = validate_deployment_plan(inputs["DeploymentPlan"].copy(), validation_log)
        truck_con = validate_truck_config(self.static["TruckReleaseCon"].copy(), validation_log)
        demand_prio = check_and_deduplicate(
            self.static["DemandPriority"].copy(), "demand_element", "Global_DemandPriority", validation_log
        )
        material_md = check_and_deduplicate(
            self.static["MaterialMD"].copy(), "material", "M6_MaterialMD", validation_log
        )
        truck_specs = check_and_deduplicate(
            self.static["TruckTypeSpecs"].copy(), "truck_type", "M6_TruckTypeSpecs", validation_log
        )
        priority_map = self._build_priority_map(demand_prio)
        material_map = self._build_material_map(material_md)
        spec_map = self._build_spec_map(truck_specs)
        dp = validate_priority_mapping(dp, priority_map, validation_log)
        dp = self._process_material_metadata(dp, material_map, validation_log)
        validate_threshold_config(truck_con, validation_log)
        validate_truck_specs(truck_con, spec_map, validation_log)
        dp = self._prepare_deployment_plan(dp, priority_map)
        dp = self._handle_uid_duplicates(dp, validation_log)

        cap_daily = normalize_capacity_plan(self.static["TruckCapacityPlan"].copy(), day, day)
        available_inventory = {}
        inventory = inputs["Inventory"]
        if not inventory.empty and {"material", "location", "quantity"}.issubset(inventory):
            available_inventory = {
                (row.material, row.location): float(row.quantity)
                for row in inventory.loc[:, ["material", "location", "quantity"]].itertuples(index=False)
            }
        run_params = {
            "max_wait_days": self.owner.max_wait_days,
            "random_seed": self.owner.random_seed,
            "is_integrated": bool(available_inventory),
            "orchestrator": None,
            "sim_dates": pd.DatetimeIndex([day]),
            "sim_start": day,
            "sim_end": day,
            "output_file": None,
        }
        prepared_data = {
            "dp": dp,
            "dp_dict": dp.set_index("ori_deployment_uid").to_dict("index") if not dp.empty else {},
            "truck_con": truck_con,
            "lead_time": self.static["LeadTime"],
            "delay_dist": self.static["DeliveryDelayDistribution"],
            "bypass_rules": self.static["MDQBypassRules"],
            "prio_map": priority_map,
            "spec_map": spec_map,
            "cap_map": build_capacity_map(cap_daily),
            "validation_log": validation_log,
            "available_inventory": available_inventory,
        }
        return run_params, prepared_data

    def execute_daily_flow(self, run_params: dict[str, Any], prepared_data: dict[str, Any]) -> dict[str, list]:
        # ``run_simulation_loop`` 仅在 is_integrated + orchestrator 非空时自行拉取库存。
        # 此处提供轻量适配器，确保新 ModuleSix 从 Context view 而非旧 Orchestrator 读取。
        class _InventoryContext:
            def __init__(self, inventory):
                self.unrestricted_inventory = inventory

            @staticmethod
            def get_shipment_log_view(date: str) -> pd.DataFrame:
                return pd.DataFrame(columns=["material", "location", "quantity"])

        if run_params["random_seed"] is not None:
            np.random.seed(run_params["random_seed"])
        if run_params["is_integrated"]:
            run_params = {**run_params, "orchestrator": _InventoryContext(prepared_data["available_inventory"])}
        return run_simulation_loop(run_params, prepared_data)

    def finalise_result(self, run_params: dict[str, Any], results: dict[str, list], validation_log: list[dict]) -> dict[str, Any]:
        return generate_outputs(run_params, results, validation_log, skip_file_output=True)


class _PolarsBackend:
    """独立的 M6 Polars 后端。

    表驱动的规范化、映射、过滤、排序和容量索引由 Polars 完成；车辆装载属于
    天然顺序敏感的逐路线/逐车型/逐车辆状态机，故在已排序的 Polars 行记录上使用
    Python 循环及共享的 ``VehiclePacker`` 纯业务组件。不会调用 pandas backend。
    """

    engine = "polars"

    def __init__(self, owner=None):
        self.owner = owner
        self.static: dict[str, pd.DataFrame] = {}
        self.static_polars: dict[str, pl.DataFrame] = {}
        self.result: dict[str, Any] = {}

    def empty_result(self) -> dict[str, Any]:
        return {
            "delivery_plan": pd.DataFrame(),
            "vehicle_log": pd.DataFrame(columns=["date", "sending", "receiving", "truck_type", "vehicle_no", "vehicle_uid", "total_units", "total_weight", "total_volume", "WFR", "VFR", "trigger"]),
            "truck_usage": pd.DataFrame(columns=["date", "sending", "receiving", "truck_type", "truck_used"]),
            "unsatisfied_log": pd.DataFrame(), "validation_log": pd.DataFrame(), "bypass_log": pd.DataFrame(),
            "statistics": {"delivery_count": 0, "vehicle_count": 0, "unsatisfied_count": 0, "bypass_count": 0},
        }

    @staticmethod
    def _normalise(frame: pd.DataFrame | None) -> pd.DataFrame:
        if frame is None:
            return pd.DataFrame()
        return normalize_identifiers(frame.copy()) if not frame.empty else frame.copy()

    @staticmethod
    def _pl(frame: pd.DataFrame | None) -> pl.DataFrame:
        return pandas_to_polars(frame) if frame is not None and not frame.empty else pl.DataFrame()

    @staticmethod
    def _pd(frame: pl.DataFrame) -> pd.DataFrame:
        return frame.to_pandas() if frame.width else pd.DataFrame()

    def normalise_static_config(self, datas: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        mapping = {
            "TruckReleaseCon": "M6_TruckReleaseCon", "TruckCapacityPlan": "M6_TruckCapacityPlan",
            "TruckTypeSpecs": "M6_TruckTypeSpecs", "MaterialMD": "M6_MaterialMD",
            "DeliveryDelayDistribution": "M6_DeliveryDelayDistribution", "MDQBypassRules": "M6_MDQBypassRules",
            "DemandPriority": "Global_DemandPriority", "LeadTime": "Global_LeadTime",
        }
        static_pandas = {
            name: self._normalise(datas.get(source))
            for name, source in mapping.items()
        }
        for name, columns in {
            "TruckReleaseCon": {"wfr": "WFR", "vfr": "VFR"},
            "LeadTime": {"pdt": "PDT", "gr": "GR", "otd": "OTD"},
        }.items():
            frame = static_pandas[name]
            static_pandas[name] = frame.rename(columns={
                source: target
                for source, target in columns.items()
                if source in frame and target not in frame
            })
        # Excel 日期经 ConfigReader 后可能仍是混合字符串/时间戳。先在 pandas 边界
        # 统一解析，再转换为 Polars，避免 Polars 对字符串格式作猜测而中止整个调度。
        for name, columns in {
            "TruckCapacityPlan": ("date", "eff_from", "eff_to"),
            "DeliveryDelayDistribution": ("date",),
        }.items():
            frame = static_pandas[name]
            for column in columns:
                if column in frame.columns:
                    frame[column] = pd.to_datetime(frame[column], errors="coerce")

        self.static_polars = {
            name: self._pl(frame) for name, frame in static_pandas.items()
        }
        # 公开静态属性仍维持 pandas 契约；计算所用 canonical 表为 static_polars。
        self.static = {name: self._pd(frame) for name, frame in self.static_polars.items()}
        return self.static

    def validate_static_config(self, static: dict[str, pd.DataFrame]) -> None:
        required = {"TruckReleaseCon", "TruckCapacityPlan", "TruckTypeSpecs", "MaterialMD", "DeliveryDelayDistribution", "MDQBypassRules", "DemandPriority", "LeadTime"}
        if required - set(self.static_polars):
            raise ValueError(f"ModuleSix 缺少静态配置: {sorted(required - set(self.static_polars))}")

    def store_static_state(self, static: dict[str, pd.DataFrame]) -> None:
        self.static = static

    def load_daily_inputs(self, day: pd.Timestamp) -> dict[str, pd.DataFrame]:
        ctx, date_text = self.owner.state_context, day.strftime("%Y-%m-%d")
        self.daily_inputs_polars = {
            "DeploymentPlan": self._pl(ctx.get_open_deployment_view(date_text)),
            "Inventory": self._pl(ctx.get_unrestricted_inventory_view(date_text)),
        }
        return {name: self._pd(frame) for name, frame in self.daily_inputs_polars.items()}

    def prepare_daily_data(self, inputs: dict[str, pd.DataFrame], day: pd.Timestamp):
        log: list[dict] = []
        dp = self.daily_inputs_polars["DeploymentPlan"]
        if dp.is_empty():
            prepared_dp = dp
        else:
            defaults = {"material": "UNKNOWN", "sending": "UNKNOWN_SENDING", "receiving": "UNKNOWN_RECEIVING", "demand_element": "DEFAULT", "planned_deployment_date": day, "deployed_qty": 0}
            for column, default in defaults.items():
                if column not in dp.columns:
                    dp = dp.with_columns(pl.lit(default).alias(column))
            prio = self.static_polars["DemandPriority"].select("demand_element", "priority")
            dp = dp.join(prio, on="demand_element", how="inner")
            material = self.static_polars["MaterialMD"]
            needed = [c for c in ("material", "demand_unit_to_weight", "demand_unit_to_volume") if c in material.columns]
            if len(needed) == 3:
                dp = dp.join(material.select(needed), on="material", how="left")
            else:
                dp = dp.with_columns(pl.lit(None).alias("demand_unit_to_weight"), pl.lit(None).alias("demand_unit_to_volume"))
            missing = dp.filter(pl.col("demand_unit_to_weight").is_null() | pl.col("demand_unit_to_volume").is_null())
            if missing.height:
                for row in missing.group_by("material").len().iter_rows(named=True):
                    log.append({"sheet": "M6_MaterialMD", "row": "", "issue": f'Missing material metadata for "{row["material"]}" (affects {row["len"]} records). Default unit conversion factors (1.0) will be used.', "severity": "WARNING", "impact": f'Default Values Used - {row["len"]} records', "missing_material": row["material"], "affected_records": row["len"]})
            dp = dp.with_columns(
                pl.col("planned_deployment_date").cast(pl.String).str.to_datetime(strict=False),
                pl.col("demand_unit_to_weight").cast(pl.Float64, strict=False).fill_null(1.0),
                pl.col("demand_unit_to_volume").cast(pl.Float64, strict=False).fill_null(1.0),
            ).sort(["planned_deployment_date", "sending", "receiving", "material", "demand_element"], maintain_order=True)
            if "ori_deployment_uid" not in dp.columns:
                dp = dp.with_row_index("_uid_index").with_columns((pl.lit("UID") + pl.col("_uid_index").cast(pl.String).str.zfill(6)).alias("ori_deployment_uid")).drop("_uid_index")
            dp = dp.unique(subset=["ori_deployment_uid"], keep="first", maintain_order=True).with_columns(
                pl.lit(0).alias("waiting_days"), pl.col("planned_deployment_date").alias("simulation_date")
            )
            prepared_dp = dp
        truck_con = self.static_polars["TruckReleaseCon"]
        for column, default in {"sending": "UNKNOWN", "receiving": "UNKNOWN", "truck_type": "UNKNOWN", "WFR": 0.0, "VFR": 0.0}.items():
            if column not in truck_con.columns:
                truck_con = truck_con.with_columns(pl.lit(default).alias(column))
        specs = self.static_polars["TruckTypeSpecs"]
        spec_map = {str(row["truck_type"]): row for row in specs.to_dicts()} if "truck_type" in specs.columns else {}
        cap = self.static_polars["TruckCapacityPlan"]
        cap_map: dict[tuple, int] = {}
        if not cap.is_empty() and {"date", "sending", "receiving", "truck_type", "truck_number"}.issubset(cap.columns):
            for row in cap.filter(pl.col("date").dt.date() == day.date()).group_by(["date", "sending", "receiving", "truck_type"]).agg(pl.col("truck_number").sum()).iter_rows(named=True):
                cap_map[(pd.Timestamp(row["date"]), row["sending"], row["receiving"], row["truck_type"])] = int(row["truck_number"])
        inventory = self.daily_inputs_polars["Inventory"]
        available = {(row["material"], row["location"]): float(row["quantity"]) for row in inventory.to_dicts()} if {"material", "location", "quantity"}.issubset(inventory.columns) else {}
        return ({"max_wait_days": self.owner.max_wait_days, "random_seed": self.owner.random_seed, "day": day}, {
            "rows": prepared_dp.to_dicts(), "truck_con": truck_con.to_dicts(), "spec_map": spec_map,
            "cap_map": cap_map, "lead_time": self._pd(self.static_polars["LeadTime"]),
            "delay_dist": self._pd(self.static_polars["DeliveryDelayDistribution"]),
            "bypass_rules": self._pd(self.static_polars["MDQBypassRules"]), "available_inventory": available,
            "validation_log": log,
        })

    def execute_daily_flow(self, run_params: dict[str, Any], prepared_data: dict[str, Any]):
        if run_params["random_seed"] is not None:
            np.random.seed(run_params["random_seed"])
        day, max_wait = run_params["day"], run_params["max_wait_days"]
        results = {"delivery_plan": [], "vehicle_log": [], "unsat_log": [], "bypass_log": []}
        evaluator = SafeExpressionEvaluator(["waiting_days", "deployed_qty_ratio", "exception_MDQ", "sending", "receiving", "truck_type", "demand_element"])
        statuses = {row["ori_deployment_uid"]: {"qty": row["deployed_qty"], "planned": row["planned_deployment_date"]} for row in prepared_data["rows"]}
        rows = []
        for row in prepared_data["rows"]:
            planned = pd.Timestamp(row["planned_deployment_date"])
            row = dict(row)
            row["deployed_qty"] = statuses[row["ori_deployment_uid"]]["qty"]
            row["waiting_days"] = (day - planned).days + 1
            if row["deployed_qty"] > 0 and row["sending"] != row["receiving"]:
                rows.append(row)
        rows.sort(key=lambda row: (row["priority"], row["planned_deployment_date"], row["sending"], row["receiving"]))
        routes: dict[tuple, list[dict]] = {}
        for row in rows:
            routes.setdefault((row["sending"], row["receiving"]), []).append(row)
        inventory = prepared_data["available_inventory"]
        for (sending, receiving), demand in routes.items():
            configs = [row for row in prepared_data["truck_con"] if row["sending"] == sending and row["receiving"] == receiving]
            if not configs:
                continue
            optimal = [r["truck_type"] for r in configs if r.get("optimal_type") == "Y"]
            types = optimal + [r["truck_type"] for r in configs if r["truck_type"] not in optimal]
            remaining = [dict(row) for row in demand]
            route_mdq = min(float(row.get("MDQ") or 0) for row in configs)
            for truck_type in types:
                config = next(row for row in configs if row["truck_type"] == truck_type)
                spec = prepared_data["spec_map"].get(str(truck_type))
                if spec is None:
                    continue
                n_truck = int(prepared_data["cap_map"].get((day, sending, receiving, truck_type), 99))
                used = 0
                while used < n_truck and remaining:
                    packer = VehiclePacker(float(spec["capacity_qty_in_weight"]), float(spec["capacity_qty_in_volume"]))
                    for index, row in enumerate(remaining):
                        if packer.is_full(): break
                        limit = calculate_inventory_limit(inventory, row["material"], row["sending"], packer.get_material_loaded(row["material"])) if inventory else None
                        packer.add_demand(index, row, limit)
                    rep_type, rep_wait = get_representative_context(packer.load_records)
                    wfr, vfr = packer.get_load_ratios()
                    mdq = float(config.get("MDQ") or 0)
                    context = {"sending": sending, "receiving": receiving, "truck_type": truck_type, "demand_element": rep_type, "waiting_days": rep_wait, "deployed_qty_ratio": packer.current_units / mdq if mdq > 0 else 0.0, "exception_MDQ": 1 if mdq == 0 else 0}
                    bypass, rule_id = should_bypass_mdq(context, prepared_data["bypass_rules"], evaluator)
                    trigger = determine_trigger_cause(packer.has_load(), wfr, vfr, float(config["WFR"]), float(config["VFR"]), bypass, max((r["demand_row"]["waiting_days"] for r in packer.load_records), default=0), max_wait)
                    if not trigger: break
                    for index, row in enumerate(remaining):
                        if index in packer.get_loaded_indices() or packer.is_full(): continue
                        limit = calculate_inventory_limit(inventory, row["material"], row["sending"], packer.get_material_loaded(row["material"])) if inventory else None
                        packer.add_demand(index, row, limit)
                    vehicle = create_vehicle_log_entry(day, sending, receiving, truck_type, used + 1, packer, trigger)
                    results["vehicle_log"].append(vehicle)
                    for record in packer.load_records:
                        row, uid, quantity = record["demand_row"], record["demand_row"]["ori_deployment_uid"], record["load_qty"]
                        lead = calculate_lead_time(prepared_data["lead_time"], sending, receiving)
                        eta = calculate_actual_delivery_date(day, lead["OTD"], lead["GR"], sample_delivery_delay(sending, receiving, prepared_data["delay_dist"]))
                        results["delivery_plan"].append(create_delivery_record(vehicle["vehicle_uid"], uid, row, quantity, day, eta, truck_type, wfr, vfr))
                        if inventory: update_inventory_after_load(inventory, row["material"], sending, quantity)
                        statuses[uid]["qty"] = max(0, statuses[uid]["qty"] - quantity)
                        remaining[record["idx"]]["deployed_qty"] = max(0, remaining[record["idx"]]["deployed_qty"] - quantity)
                    if trigger == "bypass":
                        for record in packer.load_records:
                            results["bypass_log"].append(create_bypass_record(record["demand_row"]["ori_deployment_uid"], rule_id, day, context, vehicle["vehicle_uid"]))
                    remaining = [row for row in remaining if row["deployed_qty"] > 0]
                    used += 1
            for row in demand:
                uid, waiting = row["ori_deployment_uid"], row["waiting_days"]
                if statuses[uid]["qty"] > 0 and waiting > max_wait:
                    results["unsat_log"].append(create_unsatisfied_record(uid, row, sending, receiving, day, waiting, statuses[uid]["qty"], route_mdq))
                    statuses[uid]["qty"] = 0
        return results

    def finalise_result(self, run_params: dict[str, Any], results: dict[str, list], validation_log: list[dict]):
        return generate_outputs({"orchestrator": None, "output_file": None}, results, validation_log, skip_file_output=True)


__all__ = ["_PandasBackend", "_PolarsBackend"]
