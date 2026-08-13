"""ModuleFive 的 pandas / Polars 计算后端。

后端负责静态配置预处理及当日网络层级、残余 gap、Push、空间约束和输出计算。
``ModuleFive`` 仅保留由 Orch 分别调用的 ``prepare()`` / ``run()`` 生命周期入口。
Polars 后端以原生 Polars 表执行完整规划流程，仅在输入和 ModuleFive 输出边界
转换 pandas DataFrame。
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import date
from itertools import groupby
from time import perf_counter
from typing import Any, Optional

import numpy as np
import pandas as pd
import polars as pl

from ...utils.df_convert import pandas_to_polars


class _PandasBackend:
    """M5 的 pandas 计算后端。

    后端持有门面仅用于读取其公共运行上下文（日期、Orch 和
    ``StateContext``）；计算期间的静态配置、网络层级和结果均属于后端。
    ``allocate_*`` 保持静态方法，以兼容历史的直接测试调用。
    """

    engine = "pandas"

    def __init__(self, owner=None):
        self.owner = owner
        self.static: dict[str, pd.DataFrame] = {}
        self.static_config: dict[str, pd.DataFrame] = self.static
        self.layer_map: dict[tuple[str, str], int] = {}
        self.location_to_layer: dict[tuple[str, str], int] = self.layer_map
        self.layers: list[int] = []
        self.prepared = False
        self.result: dict = {}

    _ID_COLUMNS = ("material", "location", "sending", "receiving", "sourcing")
    _DATE_COLUMNS = ("date", "simulation_date", "available_date", "actual_delivery_date", "planned_deployment_date", "eff_from", "eff_to")

    @classmethod
    def _normalise(cls, frame: Optional[pd.DataFrame]) -> pd.DataFrame:
        if frame is None:
            return pd.DataFrame()
        result = frame.copy()
        if result.empty:
            return result
        for column in cls._ID_COLUMNS:
            if column not in result:
                continue
            value = result[column].fillna("").astype(str).str.strip()
            result[column] = value
            if column != "material":
                numeric = value.str.fullmatch(r"\d+", na=False)
                result.loc[numeric, column] = value.loc[numeric].str.zfill(4)
        if "material" in result:
            result["material"] = result["material"].str.replace(r"\.0$", "", regex=True)
        for column in cls._DATE_COLUMNS:
            if column in result:
                result[column] = pd.to_datetime(result[column], errors="coerce").dt.normalize()
        return result

    @staticmethod
    def empty(columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=columns)

    @staticmethod
    def quantity(frame: pd.DataFrame, candidates: tuple[str, ...] = ("quantity",)) -> pd.Series:
        """按优先级合并同一事实行中的候选数量列。"""
        quantity = pd.Series(np.nan, index=frame.index, dtype=float)
        for column in candidates:
            if column in frame:
                quantity = quantity.fillna(pd.to_numeric(frame[column], errors="coerce"))
        return quantity.fillna(0).clip(lower=0).astype(np.int64)

    @staticmethod
    def number(frame: pd.DataFrame, column: str, default: int) -> pd.Series:
        source = frame[column] if column in frame else pd.Series(default, index=frame.index)
        return pd.to_numeric(source, errors="coerce").fillna(default)

    @staticmethod
    def stable(frame: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
        if frame.empty:
            return frame.reset_index(drop=True)
        columns = [column for column in preferred if column in frame]
        columns += [column for column in frame if column not in columns]
        try:
            return frame.sort_values(columns, kind="mergesort").reset_index(drop=True)
        except TypeError:
            keys = pd.DataFrame({f"_k{index}": frame[column].astype(str) for index, column in enumerate(columns)})
            return frame.loc[keys.sort_values(list(keys), kind="mergesort").index].reset_index(drop=True)

    @staticmethod
    def _layers(network: pd.DataFrame) -> dict[tuple[str, str], int]:
        """Build the deterministic per-material network layer map."""
        layers: dict[tuple[str, str], int] = {}
        if network.empty:
            return layers
        for material, part in network.groupby("material", sort=True, dropna=False):
            edges = part.loc[(part["sourcing"] != "") & (part["location"] != ""), ["sourcing", "location"]]
            nodes = set(edges["sourcing"]) | set(edges["location"])
            parents: dict[str, set[str]] = defaultdict(set)
            children: dict[str, set[str]] = defaultdict(set)
            for edge in edges.itertuples(index=False):
                parents[str(edge.location)].add(str(edge.sourcing))
                children[str(edge.sourcing)].add(str(edge.location))
            queue = deque((node, 0) for node in sorted(node for node in nodes if not parents[node]))
            assigned: dict[str, int] = {}
            while queue:
                node, level = queue.popleft()
                if node in assigned and assigned[node] <= level:
                    continue
                assigned[node] = level
                queue.extend((child, level + 1) for child in sorted(children[node]))
            next_level = max(assigned.values(), default=-1) + 1
            for node in sorted(nodes - set(assigned)):
                assigned[node] = next_level
                next_level += 1
            layers.update({(str(material), node): level for node, level in assigned.items()})
        return layers

    @staticmethod
    def active_network(config: dict[str, pd.DataFrame], day: pd.Timestamp) -> pd.DataFrame:
        network = config["Network"]
        active = network.loc[network["eff_from"].le(day) & network["eff_to"].ge(day)].copy()
        duplicates = active.duplicated(["material", "location"], keep=False)
        if duplicates.any():
            duplicated = active.loc[duplicates, ["material", "location", "sourcing"]]
            raise ValueError(f"Network 同日存在多个 sourcing: {duplicated.to_dict('records')[:5]}")
        return active.rename(columns={"sourcing": "upstream", "location": "node"})

    @staticmethod
    def validate(config: dict[str, pd.DataFrame], active: pd.DataFrame) -> tuple[dict[str, int], list[dict[str, Any]]]:
        """构建 M5 DemandPriority 映射与兼容的配置验证日志。"""
        log: list[dict[str, Any]] = []
        priority = config["DemandPriority"].copy()
        if not priority.empty and {"demand_element", "priority"}.issubset(priority):
            priority["demand_element"] = priority["demand_element"].astype(str)
            priority["priority"] = pd.to_numeric(priority["priority"], errors="coerce").fillna(9).astype(int)

        network = config["Network"]
        if not network.empty:
            multiple = network.groupby(["material", "location"], dropna=False)["sourcing"].nunique()
            for (material, location), _ in multiple[multiple.gt(1)].sort_index().items():
                log.append({"No": len(log) + 1, "Issue": f"Network配置不合法: material={material}, location={location} 有多个sourcing"})
        lead = config["LeadTime"]
        lead_paths = set(lead.loc[:, ["sending", "receiving"]].astype(str).itertuples(index=False, name=None)) if {"sending", "receiving"}.issubset(lead) else set()
        for route in network.loc[:, ["material", "sourcing", "location"]].itertuples(index=False):
            if (str(route.sourcing), str(route.location)) not in lead_paths:
                log.append({"No": len(log) + 1, "Issue": f"Missing leadtime for {route.sourcing}->{route.location} ({route.material})"})

        deploy, push_pull = config["DeployConfig"], config["PushPullModel"]
        push_paths = set(push_pull.loc[:, ["material", "sending"]].astype(str).itertuples(index=False, name=None)) if {"material", "sending"}.issubset(push_pull) else set()
        if {"material", "sending"}.issubset(deploy):
            for row in deploy.loc[:, ["material", "sending"]].itertuples(index=False):
                if (str(row.material), str(row.sending)) not in push_paths:
                    log.append({"No": len(log) + 1, "Issue": f"Missing PushPullModel for {row.material}/{row.sending}"})

        source_types = set(config["SupplyDemandLog"].get("demand_element", pd.Series(dtype=str)).dropna().astype(str))
        source_types.update(config["OrderLog"].get("demand_type", pd.Series(dtype=str)).dropna().astype(str))
        configured = set(priority.get("demand_element", pd.Series(dtype=str)).astype(str))
        for element in sorted(source_types):
            if element not in configured:
                value = 1 if element == "AO" else 2 if element == "normal" else 9
                priority.loc[len(priority)] = {"demand_element": element, "priority": value}
                configured.add(element)
                log.append({"No": len(log) + 1, "Issue": f"Auto add DemandPriority for {element}={value}"})
        return dict(zip(priority["demand_element"].astype(str), priority["priority"].astype(int))), log

    @staticmethod
    def legacy_inventory_keys(config: dict[str, pd.DataFrame]) -> pd.DataFrame:
        materials: set[str] = set()
        locations: set[str] = set()
        for name in ("SupplyDemandLog", "SafetyStock", "OrderLog"):
            frame = config[name]
            if frame.empty:
                continue
            if "material" in frame:
                materials.update(frame["material"].dropna().astype(str))
            if "location" in frame:
                locations.update(frame["location"].dropna().astype(str))
        return pd.DataFrame([(material, location) for material in sorted(materials) for location in sorted(locations)], columns=["material", "node"])

    @staticmethod
    def horizon(nodes: pd.DataFrame, day: pd.Timestamp, routes: pd.DataFrame) -> pd.DataFrame:
        horizon = nodes.loc[:, ["material", "node", "upstream"]].copy()
        incoming = routes.rename(columns={"sending": "upstream", "receiving": "node"})
        horizon = horizon.merge(incoming.loc[:, ["material", "upstream", "node", "leadtime"]], on=["material", "upstream", "node"], how="left")
        root_horizon = routes.groupby(["material", "sending"], as_index=False)["leadtime"].max().rename(columns={"sending": "node", "leadtime": "root_leadtime"})
        horizon = horizon.merge(root_horizon, on=["material", "node"], how="left")
        horizon["horizon_days"] = horizon["leadtime"].fillna(horizon["root_leadtime"]).fillna(1).astype(int).clip(lower=1)
        horizon["horizon_end"] = day + pd.to_timedelta(horizon["horizon_days"], unit="D")
        return horizon.loc[:, ["material", "node", "horizon_end"]]

    @classmethod
    def direct_demand(cls, nodes: pd.DataFrame, day: pd.Timestamp, config: dict[str, pd.DataFrame], routes: pd.DataFrame) -> pd.DataFrame:
        """构建 SDL、订单与窗口末日安全库存的直接需求事实。"""
        horizon = cls.horizon(nodes, day, routes)
        columns = ["material", "node", "receiving", "demand_element", "demand_qty", "requirement_date", "orig_location"]

        def windowed(source: pd.DataFrame, element: str, element_column: Optional[str] = None) -> pd.DataFrame:
            if source.empty or not {"material", "location", "date"}.issubset(source):
                return cls.empty(columns)
            data = source.merge(horizon.rename(columns={"node": "location"}), on=["material", "location"], how="inner")
            data = data.loc[data["date"].ge(day) & data["date"].le(data["horizon_end"])].copy()
            if data.empty:
                return cls.empty(columns)
            return pd.DataFrame({
                "material": data["material"], "node": data["location"], "receiving": data["location"],
                "demand_element": data[element_column].astype(str) if element_column else element,
                "demand_qty": cls.quantity(data), "requirement_date": data["date"], "orig_location": data["location"],
            })

        sdl = windowed(config["SupplyDemandLog"], "forecast", "demand_element")
        orders = windowed(config["OrderLog"], "normal", "demand_type")
        safety = config["SafetyStock"]
        if safety.empty or not {"material", "location", "date"}.issubset(safety):
            ss = cls.empty(columns)
        else:
            ss = safety.merge(horizon.rename(columns={"node": "location"}), on=["material", "location"], how="inner")
            ss = ss.loc[ss["date"].eq(ss["horizon_end"])].copy()
            ss = pd.DataFrame({"material": ss["material"], "node": ss["location"], "receiving": ss["location"], "demand_element": "safety", "demand_qty": cls.quantity(ss, ("safety_stock_qty", "quantity")), "requirement_date": ss["date"], "orig_location": ss["location"]})
            ss = ss.loc[ss["demand_qty"].gt(0)].groupby(["material", "node", "receiving", "demand_element", "requirement_date", "orig_location"], as_index=False, sort=False)["demand_qty"].sum()
        return pd.concat([sdl, orders, ss], ignore_index=True, sort=False)

    @staticmethod
    def round_routes(demand: pd.DataFrame) -> pd.DataFrame:
        result = demand.copy()
        result["planned_qty"] = result["demand_qty"].astype(np.int64)
        cross = result["node"].ne(result["receiving"])
        for _, index in result.loc[cross].groupby(["material", "node", "receiving"], sort=False).groups.items():
            rows = result.loc[index]
            total = int(rows["demand_qty"].sum())
            moq, rv = int(rows["moq"].max()), max(1, int(rows["rv"].max()))
            adjusted = 0 if total <= 0 else moq if total < moq else int(np.ceil(total / rv) * rv)
            if total <= 0:
                result.loc[index, "planned_qty"] = 0
                continue
            exact = rows["demand_qty"].to_numpy(dtype=float) * adjusted / total
            floor = np.floor(exact).astype(np.int64)
            extra = adjusted - int(floor.sum())
            order = np.lexsort((rows["row_id"].to_numpy(), -rows["demand_qty"].to_numpy(), -(exact - floor)))
            floor[order[:extra]] += 1
            result.loc[index, "planned_qty"] = floor
        return result

    @staticmethod
    def next_gap(demand: pd.DataFrame, active: pd.DataFrame) -> pd.DataFrame:
        columns = ["material", "node", "receiving", "demand_element", "demand_qty", "requirement_date", "orig_location"]
        shortage = demand.loc[demand["residual_qty"].gt(0)].copy()
        if shortage.empty:
            return _PandasBackend.empty(columns)
        shortage = shortage.merge(active.loc[:, ["material", "node", "upstream"]], on=["material", "node"], how="left")
        shortage = shortage.loc[shortage["upstream"].notna() & shortage["upstream"].ne("")].copy()
        if shortage.empty:
            return _PandasBackend.empty(columns)
        shortage["receiving"], shortage["node"] = shortage["node"], shortage["upstream"]
        shortage["demand_element"] = "net demand for " + shortage["demand_element"].astype(str)
        shortage["demand_qty"] = shortage["residual_qty"].astype(np.int64)
        keys = ["material", "node", "receiving", "demand_element", "requirement_date", "orig_location"]
        return shortage.groupby(keys, as_index=False, sort=False)["demand_qty"].sum()

    def daily_inputs(self, day: pd.Timestamp) -> dict[str, pd.DataFrame]:
        """从 StateContext 读取当日动态视图并与静态配置合并。"""
        ctx = self.owner.state_context
        if ctx is None:
            raise RuntimeError("ModuleFive.run() 需要 StateContext")
        date_text = day.strftime("%Y-%m-%d")
        static = self.static
        profile: dict[str, float] = {}

        def load(name: str, getter) -> pd.DataFrame:
            started = perf_counter()
            raw = getter(date_text)
            profile[f"{name}.view"] = perf_counter() - started
            started = perf_counter()
            result = self._normalise(raw)
            profile[f"{name}.normalise"] = perf_counter() - started
            return result

        production = load("Production", ctx.get_deployment_production_view)
        started = perf_counter()
        raw_sdl = ctx.get_deployment_supply_demand_view(date_text)
        profile["SupplyDemandLog.view"] = perf_counter() - started
        if raw_sdl is None or raw_sdl.empty:
            sdl = static["SupplyDemandLog"]
            profile["SupplyDemandLog.normalise"] = 0.0
        else:
            started = perf_counter()
            sdl = self._normalise(raw_sdl)
            profile["SupplyDemandLog.normalise"] = perf_counter() - started
        delivery = load("DeliveryGR", ctx.get_delivery_gr_view)
        if "location" in delivery and "receiving" not in delivery:
            delivery = delivery.rename(columns={"location": "receiving"})
        inputs = {
            **static,
            "SupplyDemandLog": sdl,
            "OrderLog": load("OrderLog", ctx.get_deployment_order_log_view),
            "TodayShipment": load("TodayShipment", ctx.get_shipment_log_view),
            "Inventory": load("Inventory", ctx.get_beginning_inventory_view),
            "InTransit": load("InTransit", ctx.get_planning_intransit_view),
            "DeliveryGR": delivery,
            "OpenDeployment": load("OpenDeployment", ctx.get_open_deployment_view),
            "Production": production,
            "ReceivingSpace": load("ReceivingSpace", ctx.get_space_quota_view),
        }
        self.last_daily_input_profile = profile
        return inputs

    def normalise_static_config(self, datas: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        sources = {
            "SafetyStock": "M3_SafetyStock", "Network": "Global_Network",
            "LeadTime": "Global_LeadTime", "DemandPriority": "Global_DemandPriority",
            "PushPullModel": "M5_PushPullModel", "DeployConfig": "M5_DeployConfig",
            "SupplyDemandLog": "M5_SupplyDemandLog", "MaterialLocation": "M4_MaterialLocationLineCfg",
        }
        static = {name: self._normalise(datas.get(source, pd.DataFrame())) for name, source in sources.items()}
        return static

    @staticmethod
    def validate_static_network(static: dict[str, pd.DataFrame]) -> pd.DataFrame:
        network = static["Network"].copy()
        for column in ("material", "location", "sourcing"):
            if column not in network:
                raise ValueError(f"Global_Network 缺少 {column}")
        network["eff_from"] = network.get("eff_from", pd.Series(pd.Timestamp.min, index=network.index)).fillna(pd.Timestamp.min)
        network["eff_to"] = network.get("eff_to", pd.Series(pd.Timestamp.max, index=network.index)).fillna(pd.Timestamp.max)
        static["Network"] = network
        return network

    def build_network_layers(self, network: pd.DataFrame) -> dict[tuple[str, str], int]:
        return self._layers(network)

    def store_static_state(self, static: dict[str, pd.DataFrame], layer_map: dict[tuple[str, str], int]):
        self.static = static
        self.static_config = static
        self.layer_map = layer_map
        self.location_to_layer = self.layer_map
        self.layers = sorted(set(self.layer_map.values()), reverse=True)
        self.prepared = True

    @staticmethod
    def available_supply(available: pd.DataFrame) -> pd.DataFrame:
        return available.loc[:, ["material", "node", "qty"]]

    @staticmethod
    def projected_supply(projected: pd.DataFrame) -> pd.DataFrame:
        return projected.loc[:, ["material", "node", "qty"]]

    @staticmethod
    def append_plan(plan: pd.DataFrame, push: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([plan, push], ignore_index=True, sort=False)

    @staticmethod
    def append_unfulfilled(unfulfilled: pd.DataFrame, space_unfulfilled: pd.DataFrame) -> pd.DataFrame:
        if space_unfulfilled.empty:
            return unfulfilled
        return pd.concat([unfulfilled, space_unfulfilled], ignore_index=True, sort=False)

    def route_parameters(self, active: pd.DataFrame, config: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """构建路线提前期、MOQ 和 RV 事实表。"""
        edges = active.loc[:, ["material", "upstream", "node"]].rename(columns={"upstream": "sending", "node": "receiving"}).copy()
        edges = edges.loc[edges["sending"].ne("")]
        lead = config["LeadTime"].copy()
        for column in ("PDT", "GR", "MCT"):
            lead[column] = pd.to_numeric(lead.get(column, 0), errors="coerce").fillna(0).astype(int)
        routes = edges.merge(lead.loc[:, ["sending", "receiving", "PDT", "GR", "MCT"]], on=["sending", "receiving"], how="left")
        routes[["PDT", "GR", "MCT"]] = routes[["PDT", "GR", "MCT"]].fillna(0).astype(int)
        location_type = active.loc[:, ["material", "node"]].copy()
        location_type["location_type"] = active.get("location_type", "DC")
        routes = routes.merge(location_type.rename(columns={"node": "sending"}), on=["material", "sending"], how="left")
        type_text = routes["location_type"].fillna("").astype(str).str.strip()
        roots = routes.apply(lambda row: self.owner.layer_map.get((str(row.material), str(row.sending)), 99) == 0, axis=1)
        plant = type_text.str.casefold().eq("plant") | roots
        ptf_source = config["MaterialLocation"].copy()
        ptf_source["ptf"] = self.number(ptf_source, "ptf" if "ptf" in ptf_source else "PTF", 0).astype(int)
        ptf_source["lsk"] = self.number(ptf_source, "lsk" if "lsk" in ptf_source else "LSK", 1).astype(int)
        routes = routes.merge(ptf_source.loc[:, ["material", "location", "ptf", "lsk"]].rename(columns={"location": "sending"}), on=["material", "sending"], how="left")
        routes[["ptf", "lsk"]] = routes[["ptf", "lsk"]].fillna({"ptf": 0, "lsk": 1}).astype(int)
        routes["leadtime"] = (routes["PDT"] + routes["GR"]).clip(lower=1)
        routes.loc[plant, "leadtime"] = np.maximum(1, np.maximum(routes.loc[plant, "MCT"], routes.loc[plant, "PDT"] + routes.loc[plant, "GR"]) + routes.loc[plant, "ptf"] + routes.loc[plant, "lsk"] - 1)
        routes["push_leadtime"] = (routes["PDT"] + routes["GR"]).clip(lower=1)
        push_plant = type_text.str.casefold().eq("plant") | (type_text.eq("") & roots)
        routes.loc[push_plant, "push_leadtime"] = np.maximum(1, np.maximum(routes.loc[push_plant, "MCT"], routes.loc[push_plant, "PDT"] + routes.loc[push_plant, "GR"]) + routes.loc[push_plant, "ptf"] + routes.loc[push_plant, "lsk"] - 1)
        deploy = config["DeployConfig"].copy()
        deploy["moq"] = self.number(deploy, "moq", 1).clip(lower=0).astype(int)
        deploy["rv"] = self.number(deploy, "rv", 1).clip(lower=1).astype(int)
        exact = deploy.loc[:, [column for column in ("material", "sending", "receiving", "moq", "rv") if column in deploy]].copy()
        if "receiving" in exact:
            routes = routes.merge(exact, on=["material", "sending", "receiving"], how="left")
        else:
            routes["moq"], routes["rv"] = np.nan, np.nan
        base = deploy.loc[:, ["material", "sending", "moq", "rv"]].drop_duplicates(["material", "sending"], keep="first").rename(columns={"moq": "base_moq", "rv": "base_rv"})
        routes = routes.merge(base, on=["material", "sending"], how="left")
        routes["moq"] = routes["moq"].fillna(routes["base_moq"]).fillna(1).astype(int)
        routes["rv"] = routes["rv"].fillna(routes["base_rv"]).fillna(1).astype(int)
        return routes.loc[:, ["material", "sending", "receiving", "leadtime", "push_leadtime", "moq", "rv"]].drop_duplicates()

    @staticmethod
    def apply_space(plan: pd.DataFrame, space: pd.DataFrame, priority: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """按接收地/日期/优先级应用可用空间配额。"""
        result = plan.copy()
        result["deployed_qty"] = result["deployed_qty_invCon"].astype(np.int64)
        result["quota"] = np.inf
        cross = result["sending"].ne(result["receiving"])
        if not cross.any() or space.empty:
            return result, pd.DataFrame()
        quota = space.loc[:, [column for column in ("receiving", "date", "max_qty") if column in space]].copy()
        if not {"receiving", "date", "max_qty"}.issubset(quota):
            return result, pd.DataFrame()
        quota["max_qty"] = pd.to_numeric(quota["max_qty"], errors="coerce").fillna(np.inf)
        result.loc[cross, "priority"] = result.loc[cross, "demand_element"].map(priority).fillna(99).astype(int)
        work = result.loc[cross].copy()
        work["_plan_index"] = work.index
        work = work.merge(quota.rename(columns={"max_qty": "quota"}), on=["receiving", "date"], how="left", suffixes=("", "_new"))
        work["quota"] = work["quota_new"].fillna(np.inf)
        for _, index in work.groupby(["receiving", "date"], sort=False).groups.items():
            rows = work.loc[index].sort_values(["priority", "material", "sending", "demand_element"], kind="mergesort")
            left = float(rows["quota"].iloc[0])
            for _, block in rows.groupby("priority", sort=True):
                need = block["deployed_qty_invCon"].to_numpy(dtype=np.int64)
                total = int(need.sum())
                if left >= total:
                    allocation = need
                    left -= total
                else:
                    allocation = np.minimum(np.floor(left * need / total).astype(np.int64), need) if total else np.zeros(len(need), dtype=np.int64)
                    left -= int(allocation.sum())
                work.loc[block.index, "deployed_qty"] = allocation
                if left <= 0:
                    break
        result.loc[work["_plan_index"], "deployed_qty"] = work["deployed_qty"].to_numpy(dtype=np.int64)
        result.loc[work["_plan_index"], "quota"] = work["quota"].to_numpy()
        gaps = result.loc[cross & result["deployed_qty_invCon"].gt(result["deployed_qty"])].copy()
        unfulfilled = pd.DataFrame({"date": gaps["date"], "sending": gaps["sending"], "receiving": gaps["receiving"], "material": gaps["material"], "demand_qty": gaps["demand_qty"], "demand_element": gaps["demand_element"], "unfulfilled_qty": gaps["deployed_qty_invCon"] - gaps["deployed_qty"], "reason": "space constraint"})
        return result.drop(columns="priority", errors="ignore"), unfulfilled

    def push(self, plan: pd.DataFrame, direct: pd.DataFrame, active: pd.DataFrame, routes: pd.DataFrame, config: dict[str, pd.DataFrame], stock: pd.DataFrame, projected: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
        """计算 Push/Soft-Push 调拨；保留旧端按组选择补货水平的顺序语义。"""
        columns = list(plan.columns)
        if plan.empty or config["PushPullModel"].empty:
            return self.empty(columns)
        regular = plan.loc[~plan["demand_element"].astype(str).str.contains("push", case=False, na=False)]
        pending = regular.assign(pending=regular["deployed_qty_invCon"].lt(regular["planned_qty"])).groupby(["material", "sending"], as_index=False)["pending"].any()
        candidates = regular.loc[:, ["material", "sending"]].drop_duplicates().merge(pending.loc[~pending["pending"], ["material", "sending"]], on=["material", "sending"])
        model = config["PushPullModel"].copy()
        model["model"] = model.get("model", "").astype(str).str.casefold().str.strip()
        candidates = candidates.merge(model.loc[model["model"].isin(["push", "soft push"]), ["material", "sending", "model"]], on=["material", "sending"], how="inner")
        if candidates.empty:
            return self.empty(columns)
        allocated = regular.groupby(["material", "sending"], as_index=False)["deployed_qty_invCon"].sum()
        candidates = candidates.merge(stock.rename(columns={"node": "sending", "qty": "stock"}), on=["material", "sending"], how="left").merge(allocated, on=["material", "sending"], how="left")
        candidates[["stock", "deployed_qty_invCon"]] = candidates[["stock", "deployed_qty_invCon"]].fillna(0)
        ss_today = config["SafetyStock"].loc[config["SafetyStock"].get("date", pd.Series(pd.NaT, index=config["SafetyStock"].index)).eq(day)].copy()
        ss_today["sending_safety"] = self.quantity(ss_today, ("safety_stock_qty", "quantity"))
        ss_today = ss_today.groupby(["material", "location"], as_index=False)["sending_safety"].sum().rename(columns={"location": "sending"})
        candidates = candidates.merge(ss_today, on=["material", "sending"], how="left")
        candidates["available"] = (candidates["stock"] - candidates["deployed_qty_invCon"] - np.where(candidates["model"].eq("soft push"), candidates["sending_safety"].fillna(0), 0)).clip(lower=0)
        children = active.loc[:, ["material", "upstream", "node"]].rename(columns={"upstream": "sending", "node": "receiving"})
        work = candidates.merge(children, on=["material", "sending"], how="inner").merge(routes, on=["material", "sending", "receiving"], how="left")
        if work.empty:
            return self.empty(columns)
        work["leadtime"] = work.get("push_leadtime", work["leadtime"]).fillna(work["leadtime"]).fillna(1).astype(int)
        work["planned_delivery_date"] = day + pd.to_timedelta(work["leadtime"], unit="D")
        safety = config["SafetyStock"].copy(); safety["receiving_safety"] = self.quantity(safety, ("safety_stock_qty", "quantity"))
        work = work.merge(safety.loc[:, ["material", "location", "date", "receiving_safety"]].rename(columns={"location": "receiving", "date": "planned_delivery_date"}), on=["material", "receiving", "planned_delivery_date"], how="left")
        work = work.merge(projected.rename(columns={"node": "receiving", "qty": "projected"}), on=["material", "receiving"], how="left")
        source = direct.loc[direct["demand_element"].astype(str).str.casefold().isin(["ao", "normal", "forecast", "safety"]), ["material", "node", "requirement_date", "demand_qty", "demand_element"]].rename(columns={"node": "receiving"})
        if source.empty:
            commitment = self.empty(["material", "receiving", "planned_delivery_date", "commitment"])
        else:
            commitment = work.loc[:, ["material", "receiving", "planned_delivery_date"]].drop_duplicates().merge(source, on=["material", "receiving"], how="left")
            safety_mask = commitment["demand_element"].astype(str).str.casefold().eq("safety")
            commitment = commitment.loc[((~safety_mask) & commitment["requirement_date"].gt(day) & commitment["requirement_date"].le(commitment["planned_delivery_date"])) | (safety_mask & commitment["requirement_date"].eq(commitment["planned_delivery_date"]))].groupby(["material", "receiving", "planned_delivery_date"], as_index=False, sort=False)["demand_qty"].sum().rename(columns={"demand_qty": "commitment"})
        work = work.merge(commitment, on=["material", "receiving", "planned_delivery_date"], how="left")
        work[["receiving_safety", "projected", "commitment"]] = work[["receiving_safety", "projected", "commitment"]].fillna(0)
        work["baseline"] = (work["projected"] - work["commitment"]).clip(lower=0); work["push_level"] = 1.2
        for _, index in work.groupby(["material", "sending"], sort=False).groups.items():
            group, selected, available_soh = work.loc[index], 1.2, float(work.loc[index, "available"].iloc[0])
            for level in (1.2, 1.5, 2.0, 2.5, 3.0):
                if (level * group["receiving_safety"] - group["baseline"]).clip(lower=0).sum() <= available_soh + 1e-9: selected = level
                else: break
            work.loc[index, "push_level"] = selected
        work["need"] = (work["push_level"] * work["receiving_safety"] - work["baseline"]).clip(lower=0)
        work = work.merge(work.groupby(["material", "sending"], as_index=False)["need"].sum().rename(columns={"need": "total_need"}), on=["material", "sending"], how="left")
        work["push_qty"] = np.floor(work["available"] * work["need"] / work["total_need"].replace(0, np.nan)).fillna(0).astype(np.int64)
        work = work.loc[work["push_qty"].gt(0)].copy()
        if work.empty: return self.empty(columns)
        output = pd.DataFrame({"date": day, "material": work["material"], "sending": work["sending"], "receiving": work["receiving"], "demand_qty": 0, "demand_element": np.where(work["model"].eq("push"), "push replenishment", "soft push replenishment"), "planned_qty": work["push_qty"], "deployed_qty_invCon": work["push_qty"], "deploy_qty_with_plan_order": 0, "deploy_from_in_transit": 0, "deploy_from_open_deployment_inbound": 0, "deploy_from_future_production": 0, "planned_delivery_date": work["planned_delivery_date"], "orig_location": work["receiving"], "leadtime": work["leadtime"], "is_cross_node": True})
        return output.reindex(columns=columns, fill_value=0)

    def finalise_result(self, plan: pd.DataFrame, unfulfilled: pd.DataFrame, available: pd.DataFrame, today_transit: pd.DataFrame, shipment: pd.DataFrame, validation: list[dict[str, Any]], day: pd.Timestamp) -> dict:
        """构建 M5 对外 pandas 输出表和统计信息。"""
        deployed = plan.loc[plan["sending"].ne(plan["receiving"])].groupby(["material", "sending"], as_index=False)["deployed_qty_invCon"].sum().rename(columns={"sending": "node", "deployed_qty_invCon": "deployed_qty"}) if not plan.empty else self.empty(["material", "node", "deployed_qty"])
        soh = available.loc[:, ["material", "node", "stock", "production", "delivery"]].merge(today_transit.rename(columns={"qty": "in_transit"}), on=["material", "node"], how="outer").merge(shipment.rename(columns={"qty": "today_shipment"}), on=["material", "node"], how="outer").merge(deployed, on=["material", "node"], how="outer").fillna(0)
        soh["ending_soh"] = soh["stock"] + soh["production"] + soh["delivery"] + soh["in_transit"] - soh["today_shipment"] - soh["deployed_qty"]
        soh = soh.rename(columns={"node": "location", "stock": "beginning_soh", "delivery": "delivery_gr"}); soh.insert(0, "date", day)
        return {"deployment_plan": self.stable(plan, ["date", "material", "sending", "receiving", "planned_delivery_date", "demand_element"]), "unfulfilled_log": self.stable(unfulfilled, ["date", "sending", "receiving", "demand_element"]), "stock_on_hand_log": self.stable(soh, ["date", "material", "location"]), "validation_log": pd.DataFrame(validation), "statistics": {"deployment_count": len(plan), "unfulfilled_count": len(unfulfilled), "processed_dates": 1}}

    def supply_ledger(self, config: dict[str, pd.DataFrame], day: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Build available inventory, three future-supply pools and SOH inputs."""
        inventory = config["Inventory"].copy(); inventory["qty"] = self.quantity(inventory)
        stock = inventory.groupby(["material", "location"], as_index=False)["qty"].sum().rename(columns={"location": "node"})
        stock = pd.concat([stock, self.legacy_inventory_keys(config).assign(qty=0)], ignore_index=True, sort=False).groupby(["material", "node"], as_index=False)["qty"].sum()
        delivery = config["DeliveryGR"].copy(); delivery["qty"] = self.quantity(delivery)
        delivery = delivery.groupby(["material", "receiving"], as_index=False)["qty"].sum().rename(columns={"receiving": "node"}) if not delivery.empty else self.empty(["material", "node", "qty"])
        production = config["Production"].copy(); production["qty"] = self.quantity(production, ("produced_qty", "planned_qty", "quantity"))
        production_date = production.get("available_date", pd.Series(pd.NaT, index=production.index))
        today_production = production.loc[production_date.eq(day)].groupby(["material", "location"], as_index=False)["qty"].sum().rename(columns={"location": "node"}) if not production.empty else self.empty(["material", "node", "qty"])
        future_production = production.loc[production_date.gt(day)].groupby(["material", "location"], as_index=False)["qty"].sum().rename(columns={"location": "node", "qty": "future_production"}) if not production.empty else self.empty(["material", "node", "future_production"])
        transit = config["InTransit"].copy(); transit["qty"] = self.quantity(transit)
        transit_date = transit.get("actual_delivery_date", transit.get("available_date", pd.Series(pd.NaT, index=transit.index)))
        today_transit = transit.loc[transit_date.eq(day)].groupby(["material", "receiving"], as_index=False)["qty"].sum().rename(columns={"receiving": "node"}) if not transit.empty else self.empty(["material", "node", "qty"])
        future_transit = transit.loc[transit_date.gt(day)].groupby(["material", "receiving"], as_index=False)["qty"].sum().rename(columns={"receiving": "node", "qty": "future_intransit"}) if not transit.empty else self.empty(["material", "node", "future_intransit"])
        opened = config["OpenDeployment"].copy(); opened["qty"] = self.quantity(opened, ("deployed_qty", "quantity"))
        cross = opened.get("sending", pd.Series("", index=opened.index)).ne(opened.get("receiving", pd.Series("", index=opened.index)))
        outbound = opened.loc[cross].groupby(["material", "sending"], as_index=False)["qty"].sum().rename(columns={"sending": "node"}) if not opened.empty else self.empty(["material", "node", "qty"])
        inbound = opened.loc[cross].groupby(["material", "receiving"], as_index=False)["qty"].sum().rename(columns={"receiving": "node", "qty": "open_inbound"}) if not opened.empty else self.empty(["material", "node", "open_inbound"])
        shipment = config["TodayShipment"].copy(); shipment["qty"] = self.quantity(shipment)
        shipment = shipment.loc[shipment.get("date", pd.Series(day, index=shipment.index)).eq(day)].groupby(["material", "location"], as_index=False)["qty"].sum().rename(columns={"location": "node"}) if not shipment.empty else self.empty(["material", "node", "qty"])
        ledger = pd.concat([stock.assign(kind="stock"), delivery.assign(kind="delivery"), today_production.assign(kind="production"), outbound.assign(kind="outbound")], ignore_index=True, sort=False)
        available = ledger.pivot_table(index=["material", "node"], columns="kind", values="qty", aggfunc="sum", fill_value=0).reset_index()
        for column in ("stock", "delivery", "production", "outbound"):
            if column not in available: available[column] = 0
        available["qty"] = (available["stock"] + available["delivery"] + available["production"] - available["outbound"]).clip(lower=0).astype(np.int64)
        pools = available.loc[:, ["material", "node"]].merge(future_transit, on=["material", "node"], how="left").merge(inbound, on=["material", "node"], how="left").merge(future_production, on=["material", "node"], how="left").fillna(0)
        projected = pools.loc[:, ["material", "node", "future_production"]].merge(available.loc[:, ["material", "node", "qty"]], on=["material", "node"], how="left").merge(today_transit.rename(columns={"qty": "today_transit"}), on=["material", "node"], how="left").merge(shipment.rename(columns={"qty": "shipment"}), on=["material", "node"], how="left").fillna(0)
        projected["qty"] = projected["qty"] + projected["today_transit"] + projected["future_production"] - projected["shipment"]
        return available, pools, projected, today_transit, shipment

    def plan_layers(self, day: pd.Timestamp, config: dict[str, pd.DataFrame], active: pd.DataFrame, routes: pd.DataFrame, priority: dict[str, int], available: pd.DataFrame, pools: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Propagate residual demand from downstream layers and build deployment rows."""
        direct_parts: list[pd.DataFrame] = []; plan_parts: list[pd.DataFrame] = []; unfulfilled_parts: list[pd.DataFrame] = []
        gap = self.empty(["material", "node", "receiving", "demand_element", "demand_qty", "requirement_date", "orig_location"]); row_id = 0
        profile: list[dict[str, int]] = []
        for layer in self.layers:
            base = pd.DataFrame([(material, node) for (material, node), value in self.layer_map.items() if value == layer], columns=["material", "node"])
            nodes = base.merge(active.loc[:, ["material", "node", "upstream"]], on=["material", "node"], how="left")
            direct = self.direct_demand(nodes, day, config, routes); direct_parts.append(direct)
            layer_gap = gap.merge(nodes.loc[:, ["material", "node"]].drop_duplicates(), on=["material", "node"], how="inner")
            if not layer_gap.empty:
                layer_gap = layer_gap.merge(self.horizon(nodes, day, routes), on=["material", "node"], how="left")
                layer_gap = layer_gap.loc[layer_gap["requirement_date"].ge(day) & layer_gap["requirement_date"].le(layer_gap["horizon_end"])].drop(columns="horizon_end")
            demand = pd.concat([direct, layer_gap], ignore_index=True, sort=False)
            if demand.empty:
                gap = self.empty(gap.columns.tolist()); continue
            demand = demand.merge(routes.rename(columns={"sending": "node"}), on=["material", "node", "receiving"], how="left")
            demand["moq"] = demand["moq"].fillna(1).astype(int); demand["rv"] = demand["rv"].fillna(1).astype(int)
            demand["priority"] = demand["demand_element"].map(priority).fillna(99).astype(int)
            demand["row_id"] = np.arange(row_id, row_id + len(demand)); row_id += len(demand)
            demand = self.round_routes(demand); demand = self.allocate_priority(demand, available.loc[:, ["material", "node", "qty"]]); demand = self.allocate_pipeline(demand, pools)
            demand["residual_qty"] = (demand["planned_qty"] - demand["deployed_qty_invCon"] - demand["deploy_qty_with_plan_order"]).clip(lower=0).astype(np.int64)
            shortage = demand.loc[demand["residual_qty"].gt(0)]
            direct_by_element = direct.groupby("demand_element", sort=False).size().to_dict() if not direct.empty else {}
            profile.append({"layer": layer, "direct_rows": len(direct), "direct_by_element": direct_by_element, "gap_rows": len(layer_gap), "demand_rows": len(demand), "shortage_rows": len(shortage)})
            unfulfilled_parts.append(pd.DataFrame({"date": day, "sending": shortage["node"], "receiving": shortage["receiving"], "demand_qty": shortage["demand_qty"], "demand_element": shortage["demand_element"], "unfulfilled_qty": shortage["residual_qty"], "reason": "supply shortage"}))
            planned_delivery = demand["requirement_date"].where(demand["node"].ne(demand["receiving"]), day)
            plan_parts.append(pd.DataFrame({"date": day, "material": demand["material"], "sending": demand["node"], "receiving": demand["receiving"], "demand_qty": demand["demand_qty"], "demand_element": demand["demand_element"], "planned_qty": demand["planned_qty"], "deployed_qty_invCon": demand["deployed_qty_invCon"], "deploy_qty_with_plan_order": demand["deploy_qty_with_plan_order"], "deploy_from_in_transit": demand["deploy_from_in_transit"], "deploy_from_open_deployment_inbound": demand["deploy_from_open_deployment_inbound"], "deploy_from_future_production": demand["deploy_from_future_production"], "planned_delivery_date": planned_delivery, "orig_location": demand["orig_location"], "leadtime": np.where(demand["node"].eq(demand["receiving"]), 0, demand["leadtime"].fillna(1).astype(int)), "is_cross_node": demand["node"].ne(demand["receiving"])}))
            gap = self.next_gap(demand, active)
        plan = pd.concat(plan_parts, ignore_index=True, sort=False) if plan_parts else self.empty(["date", "material", "sending", "receiving", "demand_qty", "demand_element", "planned_qty", "deployed_qty_invCon"])
        direct_all = pd.concat(direct_parts, ignore_index=True, sort=False) if direct_parts else self.empty(["material", "node", "demand_element", "demand_qty"])
        unfulfilled = pd.concat(unfulfilled_parts, ignore_index=True, sort=False) if unfulfilled_parts else pd.DataFrame()
        self.last_plan_layer_profile = profile
        return plan, direct_all, unfulfilled

    def empty_result(self):
        self.result = {
            "deployment_plan": pd.DataFrame(), "unfulfilled_log": pd.DataFrame(),
            "stock_on_hand_log": pd.DataFrame(), "validation_log": pd.DataFrame(),
            "statistics": {"deployment_count": 0, "unfulfilled_count": 0, "processed_dates": 1},
        }
        return self.result

    @staticmethod
    def allocate_priority(demand: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
        return _allocate_priority_pandas(demand, stock)

    @staticmethod
    def allocate_pipeline(demand: pd.DataFrame, pools: pd.DataFrame) -> pd.DataFrame:
        return _allocate_pipeline_pandas(demand, pools)


def _allocate_priority_pandas(demand: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    """与原 ModuleFive 保持完全一致的 pandas 基线实现。"""
    result = demand.copy()
    result["deployed_qty_invCon"] = 0
    if result.empty or stock.empty:
        return result
    keys = ["material", "node"]
    priority_keys = keys + ["priority"]
    available = stock.loc[:, keys + ["qty"]].drop_duplicates(keys, keep="last")
    priority_need = result.groupby(priority_keys, as_index=False, sort=False)["planned_qty"].sum()
    priority_need = priority_need.merge(available, on=keys, how="left")
    priority_need["qty"] = priority_need["qty"].fillna(0).astype(np.int64)
    priority_need = priority_need.sort_values(priority_keys, kind="mergesort")
    previous_need = priority_need.groupby(keys, sort=False)["planned_qty"].cumsum() - priority_need["planned_qty"]
    priority_need["available"] = (priority_need["qty"] - previous_need).clip(lower=0)
    priority_need["allocation"] = np.minimum(priority_need["available"], priority_need["planned_qty"])
    result = result.merge(priority_need.loc[:, priority_keys + ["allocation", "planned_qty"]], on=priority_keys, how="left", suffixes=("", "_priority"), sort=False)
    priority_total = result["planned_qty_priority"].replace(0, np.nan)
    result["deployed_qty_invCon"] = np.minimum(
        np.floor(result["allocation"] * result["planned_qty"] / priority_total).fillna(0).astype(np.int64),
        result["planned_qty"].astype(np.int64),
    )
    return result.drop(columns=["allocation", "planned_qty_priority"])


def _allocate_pipeline_pandas(demand: pd.DataFrame, pools: pd.DataFrame) -> pd.DataFrame:
    """与原 ModuleFive 保持完全一致的 pandas 基线实现。"""
    result = demand.copy()
    for column in ("deploy_qty_with_plan_order", "deploy_from_in_transit", "deploy_from_open_deployment_inbound", "deploy_from_future_production"):
        result[column] = 0
    self_demand = result["node"].eq(result["receiving"])
    if not self_demand.any() or pools.empty:
        return result
    keys = ["material", "node"]
    pool_columns = ["future_intransit", "open_inbound", "future_production"]
    pool_data = pools.loc[:, keys + pool_columns].drop_duplicates(keys, keep="last")
    result = result.merge(pool_data, on=keys, how="left", sort=False)
    result[pool_columns] = result[pool_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    previous_used = pd.Series(0, index=result.index, dtype=np.int64)
    for source, output in (("future_intransit", "deploy_from_in_transit"), ("open_inbound", "deploy_from_open_deployment_inbound"), ("future_production", "deploy_from_future_production")):
        remaining = (result["planned_qty"] - result["deployed_qty_invCon"] - result[["deploy_from_in_transit", "deploy_from_open_deployment_inbound", "deploy_from_future_production"]].sum(axis=1)).clip(lower=0).astype(np.int64)
        candidate = self_demand & remaining.gt(0)
        available = (pd.to_numeric(result[source], errors="coerce").fillna(0) - previous_used).clip(lower=0)
        need_total = remaining.where(candidate, 0).groupby([result["material"], result["node"]], sort=False).transform("sum")
        ratio = (available.astype(float) * remaining.astype(float) / need_total.replace(0, np.nan)).fillna(0)
        shares = pd.Series(np.minimum(np.floor(ratio.to_numpy()).astype(np.int64), remaining.to_numpy()), index=result.index, dtype=np.int64).where(candidate, 0).astype(np.int64)
        result[output] = shares
        previous_used = shares.groupby([result["material"], result["node"]], sort=False).transform("sum")
    result["deploy_qty_with_plan_order"] = result[["deploy_from_in_transit", "deploy_from_open_deployment_inbound", "deploy_from_future_production"]].sum(axis=1)
    return result.drop(columns=pool_columns)


class _PolarsBackend:
    """原生 Polars M5 后端。

    配置和 ``StateContext`` 是既有 pandas I/O 契约，因此仅在读入时转换；本类
    的静态表、当日表和计划中间结果始终为 :class:`polars.DataFrame`。顺序敏感的
    MOQ、余量传播和空间配额保留确定性的 Python 分组循环，但循环读写的是 Polars
    行/表，绝不委托 pandas backend。
    """

    engine = "polars"
    _ID_COLUMNS = ("material", "location", "sending", "receiving", "sourcing")

    def __init__(self, owner=None):
        self.owner = owner
        self.static: dict[str, pl.DataFrame] = {}
        self.static_config = self.static
        self.layer_map: dict[tuple[str, str], int] = {}
        self.location_to_layer = self.layer_map
        self.layers: list[int] = []
        self.prepared = False
        self.result: dict = {}

    @staticmethod
    def _pl(frame: pd.DataFrame | pl.DataFrame | None) -> pl.DataFrame:
        if isinstance(frame, pl.DataFrame):
            return frame
        if frame is None:
            return pl.DataFrame()
        if frame.empty:
            return pl.DataFrame(schema={column: pl.Null for column in frame.columns})
        try:
            return pandas_to_polars(frame)
        except (TypeError, ValueError):
            return pl.DataFrame({c: [None if pd.isna(v) else v for v in frame[c].tolist()] for c in frame.columns})

    @staticmethod
    def _out(frame: pl.DataFrame) -> pd.DataFrame:
        """唯一的 Polars → pandas 输出边界。"""
        return frame.to_pandas()

    @classmethod
    def _normalise(cls, frame: pd.DataFrame | pl.DataFrame | None) -> pl.DataFrame:
        data = cls._pl(frame)
        expressions: list[pl.Expr] = []
        for column in cls._ID_COLUMNS:
            if column in data.columns:
                value = pl.col(column).cast(pl.Utf8, strict=False).fill_null("").str.strip_chars()
                if column != "material":
                    value = pl.when(value.str.contains(r"^\d+$")).then(value.str.zfill(4)).otherwise(value)
                else:
                    value = value.str.replace(r"\.0$", "")
                expressions.append(value.alias(column))
        for column in ("date", "simulation_date", "available_date", "actual_delivery_date", "planned_deployment_date", "eff_from", "eff_to"):
            if column in data.columns:
                value = pl.col(column)
                expressions.append(
                    pl.coalesce([
                        value.cast(pl.Date, strict=False),
                        value.cast(pl.Utf8, strict=False).str.to_datetime(strict=False).cast(pl.Date),
                    ]).alias(column)
                )
        return data.with_columns(expressions) if expressions else data

    @staticmethod
    def _empty(columns: list[str]) -> pl.DataFrame:
        identifiers = {"material", "node", "location", "sending", "receiving", "upstream", "orig_location", "demand_element", "reason"}
        quantities = {"demand_qty", "planned_qty", "residual_qty", "deployed_qty_invCon", "unfulfilled_qty", "row_id"}
        schema = {
            column: (pl.Utf8 if column in identifiers else pl.Int64 if column in quantities else pl.Null)
            for column in columns
        }
        return pl.DataFrame(schema=schema)

    @staticmethod
    def _qty(data: pl.DataFrame, candidates=("quantity",)) -> pl.Expr:
        expressions = [pl.col(c).cast(pl.Int64, strict=False) for c in candidates if c in data.columns]
        return pl.coalesce(expressions).fill_null(0).clip(lower_bound=0).cast(pl.Int64) if expressions else pl.lit(0, dtype=pl.Int64)

    @staticmethod
    def _layers(network: pl.DataFrame) -> dict[tuple[str, str], int]:
        layers: dict[tuple[str, str], int] = {}
        if network.is_empty():
            return layers
        for material, part in network.partition_by("material", as_dict=True, maintain_order=True).items():
            material = material[0] if isinstance(material, tuple) else material
            edges = [(str(r["sourcing"]), str(r["location"])) for r in part.select(["sourcing", "location"]).to_dicts() if r["sourcing"] and r["location"]]
            parents, children = defaultdict(set), defaultdict(set)
            for parent, child in edges:
                parents[child].add(parent); children[parent].add(child)
            nodes = set(parents) | set(children); queue = deque((n, 0) for n in sorted(n for n in nodes if not parents[n])); assigned = {}
            while queue:
                node, level = queue.popleft()
                if node in assigned and assigned[node] <= level: continue
                assigned[node] = level
                queue.extend((child, level + 1) for child in sorted(children[node]))
            next_level = max(assigned.values(), default=-1) + 1
            for node in sorted(nodes - set(assigned)):
                assigned[node] = next_level; next_level += 1
            layers.update({(str(material), node): level for node, level in assigned.items()})
        return layers

    def daily_inputs(self, day: pd.Timestamp) -> dict[str, pl.DataFrame]:
        ctx = self.owner.state_context
        if ctx is None: raise RuntimeError("ModuleFive.run() 需要 StateContext")
        text = day.strftime("%Y-%m-%d")
        profile: dict[str, float] = {}

        def load(name: str, getter) -> pl.DataFrame:
            started = perf_counter()
            raw = getter(text)
            profile[f"{name}.view"] = perf_counter() - started
            started = perf_counter()
            result = self._normalise(raw)
            profile[f"{name}.normalise"] = perf_counter() - started
            return result

        started = perf_counter()
        raw_sdl = ctx.get_deployment_supply_demand_view(text)
        profile["SupplyDemandLog.view"] = perf_counter() - started
        if raw_sdl is None or raw_sdl.empty:
            sdl = self.static["SupplyDemandLog"]
            profile["SupplyDemandLog.normalise"] = 0.0
        else:
            started = perf_counter()
            sdl = self._normalise(raw_sdl)
            profile["SupplyDemandLog.normalise"] = perf_counter() - started
        delivery = load("DeliveryGR", ctx.get_delivery_gr_view)
        if "location" in delivery.columns and "receiving" not in delivery.columns: delivery = delivery.rename({"location": "receiving"})
        inputs = {**self.static, "SupplyDemandLog": sdl, "OrderLog": load("OrderLog", ctx.get_deployment_order_log_view), "TodayShipment": load("TodayShipment", ctx.get_shipment_log_view), "Inventory": load("Inventory", ctx.get_beginning_inventory_view), "InTransit": load("InTransit", ctx.get_planning_intransit_view), "DeliveryGR": delivery, "OpenDeployment": load("OpenDeployment", ctx.get_open_deployment_view), "Production": load("Production", ctx.get_deployment_production_view), "ReceivingSpace": load("ReceivingSpace", ctx.get_space_quota_view)}
        self.last_daily_input_profile = profile
        return inputs

    def normalise_static_config(self, datas: dict[str, pd.DataFrame]) -> dict[str, pl.DataFrame]:
        sources = {"SafetyStock": "M3_SafetyStock", "Network": "Global_Network", "LeadTime": "Global_LeadTime", "DemandPriority": "Global_DemandPriority", "PushPullModel": "M5_PushPullModel", "DeployConfig": "M5_DeployConfig", "SupplyDemandLog": "M5_SupplyDemandLog", "MaterialLocation": "M4_MaterialLocationLineCfg"}
        return {name: self._normalise(datas.get(source, pd.DataFrame())) for name, source in sources.items()}

    @staticmethod
    def validate_static_network(static: dict[str, pl.DataFrame]) -> pl.DataFrame:
        network = static["Network"]
        missing = [column for column in ("material", "location", "sourcing") if column not in network.columns]
        if missing:
            raise ValueError(f"Global_Network 缺少 {missing[0]}")
        static["Network"] = network.with_columns([
            pl.col("eff_from").fill_null(pl.date(1, 1, 1)) if "eff_from" in network.columns else pl.lit(date.min).alias("eff_from"),
            pl.col("eff_to").fill_null(pl.date(9999, 12, 31)) if "eff_to" in network.columns else pl.lit(date.max).alias("eff_to"),
        ])
        return static["Network"]

    def build_network_layers(self, network: pl.DataFrame) -> dict[tuple[str, str], int]:
        return self._layers(network)

    def store_static_state(self, static: dict[str, pl.DataFrame], layer_map: dict[tuple[str, str], int]):
        self.static = static
        self.static_config = static
        self.layer_map = layer_map
        self.location_to_layer = self.layer_map
        self.layers = sorted(set(self.layer_map.values()), reverse=True)
        self.prepared = True

    @staticmethod
    def available_supply(available: pl.DataFrame) -> pl.DataFrame:
        return available.select(["material", "node", "qty"])

    @staticmethod
    def projected_supply(projected: pl.DataFrame) -> pl.DataFrame:
        return projected.select(["material", "node", "qty"])

    @staticmethod
    def append_plan(plan: pl.DataFrame, push: pl.DataFrame) -> pl.DataFrame:
        return pl.concat([plan, push], how="diagonal_relaxed")

    @staticmethod
    def append_unfulfilled(unfulfilled: pl.DataFrame, space_unfulfilled: pl.DataFrame) -> pl.DataFrame:
        if space_unfulfilled.is_empty():
            return unfulfilled
        return pl.concat([unfulfilled, space_unfulfilled], how="diagonal_relaxed")

    @staticmethod
    def active_network(config: dict[str, pl.DataFrame], day: pd.Timestamp) -> pl.DataFrame:
        active = config["Network"].filter((pl.col("eff_from") <= day.date()) & (pl.col("eff_to") >= day.date()))
        if active.group_by(["material", "location"]).len().filter(pl.col("len") > 1).height:
            raise ValueError("Network 同日存在多个 sourcing")
        return active.rename({"sourcing": "upstream", "location": "node"})

    def validate(self, config: dict[str, pl.DataFrame], active: pl.DataFrame) -> tuple[dict[str, int], list[dict[str, Any]]]:
        priority = config["DemandPriority"]
        values = {str(r["demand_element"]): int(r["priority"] or 9) for r in priority.select(["demand_element", "priority"]).to_dicts()} if {"demand_element", "priority"}.issubset(priority.columns) else {}
        log = []
        deploy, model = config["DeployConfig"], config["PushPullModel"]
        if {"material", "sending"}.issubset(deploy.columns):
            model_paths = set((str(r["material"]), str(r["sending"])) for r in model.select(["material", "sending"]).to_dicts()) if {"material", "sending"}.issubset(model.columns) else set()
            for row in deploy.select(["material", "sending"]).to_dicts():
                path = (str(row["material"]), str(row["sending"]))
                if path not in model_paths:
                    log.append({"No": len(log) + 1, "Issue": f"Missing PushPullModel for {path[0]}/{path[1]}"})
        elements: set[str] = set()
        for name, column in (("SupplyDemandLog", "demand_element"), ("OrderLog", "demand_type")):
            if column in config[name].columns: elements.update(str(x) for x in config[name].get_column(column).drop_nulls().to_list())
        for element in sorted(elements - set(values)):
            values[element] = 1 if element == "AO" else 2 if element == "normal" else 9
            log.append({"No": len(log) + 1, "Issue": f"Auto add DemandPriority for {element}={values[element]}"})
        return values, log

    @staticmethod
    def legacy_inventory_keys(config: dict[str, pl.DataFrame]) -> pl.DataFrame:
        materials: set[str] = set()
        locations: set[str] = set()
        for name in ("SupplyDemandLog", "SafetyStock", "OrderLog"):
            frame = config[name]
            if "material" in frame.columns:
                materials.update(str(value) for value in frame.get_column("material").drop_nulls().to_list())
            if "location" in frame.columns:
                locations.update(str(value) for value in frame.get_column("location").drop_nulls().to_list())
        pairs = [(material, location) for material in sorted(materials) for location in sorted(locations)]
        return pl.DataFrame(pairs, schema=["material", "node"], orient="row") if pairs else pl.DataFrame(schema={"material": pl.Utf8, "node": pl.Utf8})

    @staticmethod
    def horizon(nodes: pl.DataFrame, day: pd.Timestamp, routes: pl.DataFrame) -> pl.DataFrame:
        incoming = routes.rename({"sending": "upstream", "receiving": "node"}).select(["material", "upstream", "node", "leadtime"])
        roots = routes.group_by(["material", "sending"]).agg(pl.col("leadtime").max().alias("root_leadtime")).rename({"sending": "node"})
        return nodes.select(["material", "node", "upstream"]).join(incoming, on=["material", "upstream", "node"], how="left").join(roots, on=["material", "node"], how="left").with_columns(pl.coalesce([pl.col("leadtime"), pl.col("root_leadtime"), pl.lit(1)]).cast(pl.Int64).clip(lower_bound=1).alias("horizon_days")).with_columns((pl.lit(day.date()) + pl.duration(days=pl.col("horizon_days"))).alias("horizon_end")).select(["material", "node", "horizon_end"])

    def route_parameters(self, active: pl.DataFrame, config: dict[str, pl.DataFrame]) -> pl.DataFrame:
        edge_columns = ["material", pl.col("upstream").alias("sending"), pl.col("node").alias("receiving")]
        if "location_type" in active.columns:
            location_types = active.select(["material", pl.col("node").alias("sending"), "location_type"])
        else:
            location_types = active.select(["material", pl.col("node").alias("sending"), pl.lit("DC").alias("location_type")])
        edges = active.select(edge_columns).filter(pl.col("sending") != "")
        if not edges.is_empty():
            layer_map = getattr(self.owner, "layer_map", self.layer_map)
            root_flags = [
                layer_map.get((str(row["material"]), str(row["sending"])), 99) == 0
                for row in edges.select(["material", "sending"]).to_dicts()
            ]
            edges = edges.with_columns(pl.Series("_is_root", root_flags))
        else:
            edges = edges.with_columns(pl.lit(False).alias("_is_root"))
        lead = config["LeadTime"]
        for c in ("PDT", "GR", "MCT"):
            lead = lead.with_columns((pl.col(c).cast(pl.Int64, strict=False).fill_null(0) if c in lead.columns else pl.lit(0)).alias(c))
        routes = edges.join(location_types, on=["material", "sending"], how="left").join(
            lead.select(["sending", "receiving", "PDT", "GR", "MCT"]), on=["sending", "receiving"], how="left"
        ).with_columns([pl.col(c).fill_null(0) for c in ("PDT", "GR", "MCT")])
        ml = config["MaterialLocation"]
        for c, default in (("ptf", 0), ("lsk", 1)):
            source = c if c in ml.columns else c.upper()
            ml = ml.with_columns((pl.col(source).cast(pl.Int64, strict=False).fill_null(default) if source in ml.columns else pl.lit(default)).alias(c))
        routes = routes.join(ml.select(["material", pl.col("location").alias("sending"), "ptf", "lsk"]), on=["material", "sending"], how="left").with_columns([pl.col("ptf").fill_null(0), pl.col("lsk").fill_null(1)])
        type_text = pl.col("location_type").cast(pl.Utf8, strict=False).fill_null("").str.strip_chars().str.to_lowercase()
        plant = type_text.eq("plant") | pl.col("_is_root")
        push_plant = type_text.eq("plant") | (type_text.eq("") & pl.col("_is_root"))
        plant_leadtime = pl.max_horizontal([pl.col("MCT"), pl.col("PDT") + pl.col("GR")]) + pl.col("ptf") + pl.col("lsk") - 1
        deploy = config["DeployConfig"]
        deploy = deploy.with_columns([self._qty(deploy, ("moq",)).alias("moq"), self._qty(deploy, ("rv",)).clip(lower_bound=1).alias("rv")])
        keys = [c for c in ("material", "sending", "receiving") if c in deploy.columns]
        if len(keys) == 3: routes = routes.join(deploy.select(keys + ["moq", "rv"]), on=keys, how="left")
        else: routes = routes.with_columns([pl.lit(None).cast(pl.Int64).alias("moq"), pl.lit(None).cast(pl.Int64).alias("rv")])
        base = deploy.select(["material", "sending", pl.col("moq").alias("base_moq"), pl.col("rv").alias("base_rv")]).unique(["material", "sending"], maintain_order=True)
        return routes.join(base, on=["material", "sending"], how="left").with_columns([
            pl.coalesce([pl.col("moq"), pl.col("base_moq"), pl.lit(1)]).cast(pl.Int64).alias("moq"),
            pl.coalesce([pl.col("rv"), pl.col("base_rv"), pl.lit(1)]).cast(pl.Int64).alias("rv"),
            pl.when(plant).then(plant_leadtime).otherwise(pl.col("PDT") + pl.col("GR")).clip(lower_bound=1).cast(pl.Int64).alias("leadtime"),
            pl.when(push_plant).then(plant_leadtime).otherwise(pl.col("PDT") + pl.col("GR")).clip(lower_bound=1).cast(pl.Int64).alias("push_leadtime"),
        ]).select(["material", "sending", "receiving", "leadtime", "push_leadtime", "moq", "rv"]).unique(maintain_order=True)

    def direct_demand(self, nodes, day, config, routes):
        columns = ["material", "node", "receiving", "demand_element", "demand_qty", "requirement_date", "orig_location"]
        horizon = self.horizon(nodes, day, routes)
        diagnostics = []
        def window(source, default, element=None):
            if not {"material", "location", "date"}.issubset(source.columns):
                diagnostics.append({"source": default, "source_rows": source.height, "joined_rows": 0, "window_rows": 0, "has_required_columns": False})
                return self._empty(columns)
            joined = source.join(horizon.rename({"node": "location"}), on=["material", "location"], how="inner")
            data = joined.filter((pl.col("date") >= day.date()) & (pl.col("date") <= pl.col("horizon_end")))
            diagnostics.append({
                "source": default,
                "source_rows": source.height,
                "joined_rows": joined.height,
                "window_rows": data.height,
                "has_required_columns": True,
                "date_dtype": str(source.schema["date"]),
                "joined_date_min": str(joined.get_column("date").min()) if not joined.is_empty() else None,
                "joined_date_max": str(joined.get_column("date").max()) if not joined.is_empty() else None,
                "horizon_min": str(joined.get_column("horizon_end").min()) if not joined.is_empty() else None,
                "horizon_max": str(joined.get_column("horizon_end").max()) if not joined.is_empty() else None,
            })
            return data.select(["material", pl.col("location").alias("node"), pl.col("location").alias("receiving"), (pl.col(element).cast(pl.Utf8) if element and element in data.columns else pl.lit(default)).alias("demand_element"), self._qty(data).alias("demand_qty"), pl.col("date").alias("requirement_date"), pl.col("location").alias("orig_location")])
        sdl, orders = window(config["SupplyDemandLog"], "forecast", "demand_element"), window(config["OrderLog"], "normal", "demand_type")
        safety = config["SafetyStock"]
        ss = self._empty(columns) if not {"material", "location", "date"}.issubset(safety.columns) else safety.join(horizon.rename({"node": "location"}), on=["material", "location"], how="inner").filter(pl.col("date") == pl.col("horizon_end")).select(["material", pl.col("location").alias("node"), pl.col("location").alias("receiving"), pl.lit("safety").alias("demand_element"), self._qty(safety, ("safety_stock_qty", "quantity")).alias("demand_qty"), pl.col("date").alias("requirement_date"), pl.col("location").alias("orig_location")]).filter(pl.col("demand_qty") > 0).group_by([column for column in columns if column != "demand_qty"], maintain_order=True).agg(pl.col("demand_qty").sum())
        self.last_direct_demand_diagnostics = getattr(self, "last_direct_demand_diagnostics", []) + [{"layer_nodes": nodes.height, "sources": diagnostics}]
        return pl.concat([sdl, orders, ss], how="diagonal_relaxed")

    @classmethod
    def allocate_priority(cls, demand, stock):
        pandas_boundary = isinstance(demand, pd.DataFrame); data = cls._pl(demand).with_columns(pl.lit(0, dtype=pl.Int64).alias("deployed_qty_invCon"))
        available = cls._pl(stock)
        if data.is_empty() or available.is_empty(): return cls._out(data) if pandas_boundary else data
        need = data.group_by(["material", "node", "priority"], maintain_order=True).agg(pl.col("planned_qty").sum().alias("_need")).join(available.select(["material", "node", "qty"]), on=["material", "node"], how="left").with_columns(pl.col("qty").fill_null(0)).sort(["material", "node", "priority"]).with_columns((pl.col("_need").cum_sum().over(["material", "node"]) - pl.col("_need")).alias("_prior")).with_columns(pl.min_horizontal([(pl.col("qty") - pl.col("_prior")).clip(lower_bound=0), pl.col("_need")]).alias("_allocation"))
        data = data.join(need.select(["material", "node", "priority", "_need", "_allocation"]), on=["material", "node", "priority"], how="left").with_columns(pl.min_horizontal([((pl.col("_allocation") * pl.col("planned_qty") / pl.col("_need").replace(0, None)).floor().fill_null(0)).cast(pl.Int64), pl.col("planned_qty").cast(pl.Int64)]).alias("deployed_qty_invCon")).drop(["_need", "_allocation"])
        return cls._out(data) if pandas_boundary else data

    @classmethod
    def allocate_pipeline(cls, demand, pools):
        pandas_boundary = isinstance(demand, pd.DataFrame); data = cls._pl(demand); pool = cls._pl(pools)
        outputs = ("deploy_qty_with_plan_order", "deploy_from_in_transit", "deploy_from_open_deployment_inbound", "deploy_from_future_production")
        data = data.with_columns([pl.lit(0, dtype=pl.Int64).alias(c) for c in outputs])
        if data.is_empty() or pool.is_empty(): return cls._out(data) if pandas_boundary else data
        sources = ("future_intransit", "open_inbound", "future_production")
        data = data.join(pool.select(["material", "node", *sources]), on=["material", "node"], how="left").with_columns([pl.col(c).cast(pl.Float64, strict=False).fill_null(0) for c in sources])
        previous = None
        for source, output in zip(sources, outputs[1:]):
            remaining = (pl.col("planned_qty") - pl.col("deployed_qty_invCon") - pl.sum_horizontal([pl.col(c) for c in outputs[1:]])).clip(lower_bound=0)
            candidate = pl.col("node") == pl.col("receiving"); total = pl.when(candidate).then(remaining).otherwise(0).sum().over(["material", "node"])
            used = pl.col(previous).sum().over(["material", "node"]) if previous else pl.lit(0)
            data = data.with_columns(pl.when(candidate).then(pl.min_horizontal([((pl.col(source) - used).clip(lower_bound=0) * remaining / total.replace(0, None)).floor().fill_null(0).cast(pl.Int64), remaining.cast(pl.Int64)])).otherwise(0).alias(output)); previous = output
        data = data.with_columns(pl.sum_horizontal([pl.col(c) for c in outputs[1:]]).alias(outputs[0])).drop(list(sources))
        return cls._out(data) if pandas_boundary else data

    def supply_ledger(self, config, day):
        def grouped(name, keys, qty_cols=("quantity",), date_col=None, date_test=None, output="qty"):
            data = config[name]
            if data.is_empty() or not set(keys).issubset(data.columns): return pl.DataFrame(schema={**{k: pl.Utf8 for k in keys}, output: pl.Int64})
            if date_col and date_col in data.columns: data = data.filter(date_test(pl.col(date_col)))
            return data.group_by(keys).agg(self._qty(data, qty_cols).sum().alias(output))
        stock = grouped("Inventory", ["material", "location"]).rename({"location": "node"})
        inventory_keys = self.legacy_inventory_keys(config)
        if not inventory_keys.is_empty():
            stock = pl.concat([stock, inventory_keys.with_columns(pl.lit(0, dtype=pl.Int64).alias("qty"))], how="vertical_relaxed").group_by(["material", "node"], maintain_order=True).agg(pl.col("qty").sum())
        delivery = grouped("DeliveryGR", ["material", "receiving"]).rename({"receiving": "node"})
        production = grouped("Production", ["material", "location"], ("produced_qty", "planned_qty", "quantity"), "available_date", lambda c: c == day.date()).rename({"location": "node"})
        transit = grouped("InTransit", ["material", "receiving"], ("quantity",), "actual_delivery_date", lambda c: c == day.date()).rename({"receiving": "node"})
        shipment = grouped("TodayShipment", ["material", "location"], ("quantity",), "date", lambda c: c == day.date()).rename({"location": "node"})
        opened = config["OpenDeployment"]
        if opened.is_empty() or not {"material", "sending", "receiving"}.issubset(opened.columns):
            outbound = pl.DataFrame(schema={"material": pl.Utf8, "node": pl.Utf8, "qty": pl.Int64})
            inbound = pl.DataFrame(schema={"material": pl.Utf8, "node": pl.Utf8, "open_inbound": pl.Int64})
        else:
            cross_opened = opened.filter(pl.col("sending") != pl.col("receiving"))
            outbound = cross_opened.group_by(["material", "sending"]).agg(
                self._qty(cross_opened, ("deployed_qty", "quantity")).sum().alias("qty")
            ).rename({"sending": "node"})
            inbound = cross_opened.group_by(["material", "receiving"]).agg(
                self._qty(cross_opened, ("deployed_qty", "quantity")).sum().alias("open_inbound")
            ).rename({"receiving": "node"})
        parts = [stock.with_columns(pl.lit("stock").alias("kind")), delivery.with_columns(pl.lit("delivery").alias("kind")), production.with_columns(pl.lit("production").alias("kind")), outbound.with_columns(pl.lit("outbound").alias("kind"))]
        available = pl.concat(parts, how="diagonal_relaxed").pivot(on="kind", index=["material", "node"], values="qty", aggregate_function="sum").with_columns([pl.col(c).fill_null(0) if c in pl.concat(parts, how="diagonal_relaxed").pivot(on="kind", index=["material", "node"], values="qty", aggregate_function="sum").columns else pl.lit(0).alias(c) for c in ("stock", "delivery", "production", "outbound")]).with_columns((pl.col("stock") + pl.col("delivery") + pl.col("production") - pl.col("outbound")).clip(lower_bound=0).cast(pl.Int64).alias("qty"))
        # Future pools are intentionally constructed independently of today's ledger.
        future_t = grouped("InTransit", ["material", "receiving"], ("quantity",), "actual_delivery_date", lambda c: c > day.date(), "future_intransit").rename({"receiving": "node"})
        future_p = grouped("Production", ["material", "location"], ("produced_qty", "planned_qty", "quantity"), "available_date", lambda c: c > day.date(), "future_production").rename({"location": "node"})
        pools = available.select(["material", "node"]).join(future_t, on=["material", "node"], how="left").join(inbound, on=["material", "node"], how="left").join(future_p, on=["material", "node"], how="left").fill_null(0)
        projected = available.select(["material", "node", "qty"]).join(future_p, on=["material", "node"], how="left").join(transit.rename({"qty": "today_transit"}), on=["material", "node"], how="left").join(shipment.rename({"qty": "shipment"}), on=["material", "node"], how="left").fill_null(0).with_columns((pl.col("qty") + pl.col("future_production") + pl.col("today_transit") - pl.col("shipment")).alias("qty"))
        return available, pools, projected, transit, shipment

    def round_routes(self, demand):
        # Deterministic route-local largest-remainder rounding.
        rows = demand.to_dicts(); groups = defaultdict(list)
        for i, row in enumerate(rows):
            row["planned_qty"] = int(row["demand_qty"] or 0); groups[(row["material"], row["node"], row["receiving"])].append((i, row))
        for _, items in groups.items():
            if items[0][1]["node"] == items[0][1]["receiving"]: continue
            total = sum(int(r["demand_qty"] or 0) for _, r in items); moq = max(int(r.get("moq") or 1) for _, r in items); rv = max(1, max(int(r.get("rv") or 1) for _, r in items)); adjusted = 0 if total <= 0 else (moq if total < moq else int(np.ceil(total / rv) * rv))
            exact = [(i, r, int(r["demand_qty"] or 0) * adjusted / total if total else 0) for i, r in items]; floors = {i: int(np.floor(x)) for i, _, x in exact}
            for i, _, _ in sorted(exact, key=lambda x: (-(x[2] - floors[x[0]]), -int(x[1]["demand_qty"] or 0), int(x[1].get("row_id") or 0)))[:adjusted - sum(floors.values())]: floors[i] += 1
            for i, value in floors.items(): rows[i]["planned_qty"] = value
        schema = {**demand.schema, "planned_qty": pl.Int64}
        return pl.DataFrame(
            rows,
            schema=schema,
            strict=False,
            infer_schema_length=None,
        )

    def next_gap(self, demand, active):
        shortage = demand.filter(pl.col("residual_qty") > 0).join(active.select(["material", "node", "upstream"]), on=["material", "node"], how="left").filter(pl.col("upstream").is_not_null() & (pl.col("upstream") != ""))
        if shortage.is_empty(): return self._empty(["material", "node", "receiving", "demand_element", "demand_qty", "requirement_date", "orig_location"])
        return shortage.select(["material", pl.col("upstream").alias("node"), pl.col("node").alias("receiving"), pl.concat_str([pl.lit("net demand for "), pl.col("demand_element")]).alias("demand_element"), pl.col("residual_qty").alias("demand_qty"), "requirement_date", "orig_location"]).group_by(["material", "node", "receiving", "demand_element", "requirement_date", "orig_location"], maintain_order=True).agg(pl.col("demand_qty").sum())

    def plan_layers(self, day, config, active, routes, priority, available, pools):
        gap = self._empty(["material", "node", "receiving", "demand_element", "demand_qty", "requirement_date", "orig_location"]); plans=[]; directs=[]; unmet=[]; row_id=0; profile=[]; self.last_direct_demand_diagnostics=[]
        for layer in self.layers:
            base = pl.DataFrame([(m, n) for (m, n), value in self.layer_map.items() if value == layer], schema=["material", "node"], orient="row")
            nodes = base.join(active.select(["material", "node", "upstream"]), on=["material", "node"], how="left"); direct = self.direct_demand(nodes, day, config, routes); directs.append(direct)
            layer_gap = gap.join(nodes.select(["material", "node"]), on=["material", "node"], how="inner")
            if not layer_gap.is_empty():
                layer_gap = layer_gap.join(self.horizon(nodes, day, routes), on=["material", "node"], how="left").filter(
                    (pl.col("requirement_date") >= day.date()) & (pl.col("requirement_date") <= pl.col("horizon_end"))
                ).drop("horizon_end")
            demand = pl.concat([direct, layer_gap], how="diagonal_relaxed")
            if demand.is_empty():
                profile.append({"layer": layer, "direct_rows": direct.height, "direct_by_element": {}, "gap_rows": layer_gap.height, "demand_rows": 0, "shortage_rows": 0})
                gap = self._empty(gap.columns)
                continue
            demand = demand.join(routes.rename({"sending": "node"}), on=["material", "node", "receiving"], how="left").with_columns([pl.col("moq").fill_null(1), pl.col("rv").fill_null(1), pl.col("demand_element").replace_strict(priority, default=99).cast(pl.Int64).alias("priority"), pl.int_range(row_id, row_id + pl.len()).alias("row_id")]); row_id += demand.height
            demand = self.allocate_pipeline(self.allocate_priority(self.round_routes(demand), available.select(["material", "node", "qty"])), pools).with_columns((pl.col("planned_qty") - pl.col("deployed_qty_invCon") - pl.col("deploy_qty_with_plan_order")).clip(lower_bound=0).cast(pl.Int64).alias("residual_qty")); shortage = demand.filter(pl.col("residual_qty") > 0)
            direct_by_element = {
                str(row["demand_element"]): int(row["len"])
                for row in direct.group_by("demand_element").len().to_dicts()
            } if not direct.is_empty() else {}
            profile.append({"layer": layer, "direct_rows": direct.height, "direct_by_element": direct_by_element, "gap_rows": layer_gap.height, "demand_rows": demand.height, "shortage_rows": shortage.height})
            unmet.append(shortage.select([pl.lit(day.date()).alias("date"), pl.col("node").alias("sending"), "receiving", "demand_qty", "demand_element", pl.col("residual_qty").alias("unfulfilled_qty"), pl.lit("supply shortage").alias("reason")]))
            plans.append(demand.select([pl.lit(day.date()).alias("date"), "material", pl.col("node").alias("sending"), "receiving", "demand_qty", "demand_element", "planned_qty", "deployed_qty_invCon", "deploy_qty_with_plan_order", "deploy_from_in_transit", "deploy_from_open_deployment_inbound", "deploy_from_future_production", pl.when(pl.col("node") != pl.col("receiving")).then(pl.col("requirement_date")).otherwise(pl.lit(day.date())).alias("planned_delivery_date"), "orig_location", pl.when(pl.col("node") == pl.col("receiving")).then(0).otherwise(pl.col("leadtime").fill_null(1)).alias("leadtime"), (pl.col("node") != pl.col("receiving")).alias("is_cross_node")]))
            gap = self.next_gap(demand, active)
        self.last_plan_layer_profile = profile
        return (pl.concat(plans, how="diagonal_relaxed") if plans else self._empty(["date", "material", "sending", "receiving"]), pl.concat(directs, how="diagonal_relaxed") if directs else self._empty([]), pl.concat(unmet, how="diagonal_relaxed") if unmet else self._empty([]))

    def push(self, plan, direct, active, routes, config, stock, projected, day):
        if plan.is_empty() or config["PushPullModel"].is_empty(): return self._empty(plan.columns)
        regular = plan.filter(~pl.col("demand_element").cast(pl.Utf8).str.contains("push", literal=False))
        pending = regular.group_by(["material", "sending"]).agg((pl.col("deployed_qty_invCon") < pl.col("planned_qty")).any().alias("pending"))
        model = config["PushPullModel"].with_columns(pl.col("model").cast(pl.Utf8).str.to_lowercase().str.strip_chars()).filter(pl.col("model").is_in(["push", "soft push"]))
        candidates = regular.select(["material", "sending"]).unique(maintain_order=True).join(pending.filter(~pl.col("pending")).select(["material", "sending"]), on=["material", "sending"], how="inner").join(model.select(["material", "sending", "model"]), on=["material", "sending"], how="inner")
        if candidates.is_empty(): return self._empty(plan.columns)
        allocated = regular.group_by(["material", "sending"]).agg(pl.col("deployed_qty_invCon").sum().alias("allocated"))
        work = candidates.join(stock.rename({"node":"sending", "qty":"stock"}), on=["material", "sending"], how="left").join(allocated, on=["material", "sending"], how="left").with_columns([pl.col("stock").fill_null(0), pl.col("allocated").fill_null(0)])
        safety = config["SafetyStock"]
        if {"material", "location", "date"}.issubset(safety.columns):
            safety = safety.with_columns(self._qty(safety, ("safety_stock_qty", "quantity")).alias("safety_qty"))
            sending_ss = safety.filter(pl.col("date") == day.date()).group_by(["material", "location"]).agg(pl.col("safety_qty").sum()).rename({"location":"sending", "safety_qty":"sending_safety"})
        else: sending_ss = pl.DataFrame(schema={"material":pl.Utf8,"sending":pl.Utf8,"sending_safety":pl.Int64})
        work = work.join(sending_ss, on=["material", "sending"], how="left").with_columns(pl.when(pl.col("model") == "soft push").then(pl.col("sending_safety").fill_null(0)).otherwise(0).alias("reserved")).with_columns((pl.col("stock") - pl.col("allocated") - pl.col("reserved")).clip(lower_bound=0).alias("available"))
        children = active.select(["material", pl.col("upstream").alias("sending"), pl.col("node").alias("receiving")])
        work = work.join(children, on=["material", "sending"], how="inner").join(routes, on=["material", "sending", "receiving"], how="left").with_columns(pl.coalesce([pl.col("push_leadtime"), pl.col("leadtime"), pl.lit(1)]).cast(pl.Int64).alias("leadtime")).with_columns((pl.lit(day.date()) + pl.duration(days=pl.col("leadtime"))).alias("planned_delivery_date"))
        if work.is_empty(): return self._empty(plan.columns)
        if {"material", "location", "date"}.issubset(safety.columns): receiving_ss = safety.select(["material", pl.col("location").alias("receiving"), pl.col("date").alias("planned_delivery_date"), pl.col("safety_qty").alias("receiving_safety")])
        else: receiving_ss = pl.DataFrame(schema={"material":pl.Utf8,"receiving":pl.Utf8,"planned_delivery_date":pl.Date,"receiving_safety":pl.Int64})
        work = work.join(receiving_ss, on=["material", "receiving", "planned_delivery_date"], how="left").join(projected.rename({"node":"receiving", "qty":"projected"}), on=["material", "receiving"], how="left")
        source = direct.filter(
            pl.col("demand_element").cast(pl.Utf8, strict=False).str.to_lowercase().is_in(["ao", "normal", "forecast", "safety"])
        ).select(["material", pl.col("node").alias("receiving"), "requirement_date", "demand_qty", "demand_element"])
        if source.is_empty():
            commitment = pl.DataFrame(schema={"material": pl.Utf8, "receiving": pl.Utf8, "planned_delivery_date": pl.Date, "commitment": pl.Int64})
        else:
            commitment = work.select(["material", "receiving", "planned_delivery_date"]).unique(maintain_order=True).join(
                source, on=["material", "receiving"], how="left"
            ).filter(
                ((pl.col("demand_element").cast(pl.Utf8, strict=False).str.to_lowercase() != "safety") &
                 (pl.col("requirement_date") > day.date()) &
                 (pl.col("requirement_date") <= pl.col("planned_delivery_date"))) |
                ((pl.col("demand_element").cast(pl.Utf8, strict=False).str.to_lowercase() == "safety") &
                 (pl.col("requirement_date") == pl.col("planned_delivery_date")))
            ).group_by(["material", "receiving", "planned_delivery_date"], maintain_order=True).agg(
                pl.col("demand_qty").sum().alias("commitment")
            )
        work = work.join(commitment, on=["material", "receiving", "planned_delivery_date"], how="left").with_columns([
            pl.col("receiving_safety").fill_null(0), pl.col("projected").fill_null(0), pl.col("commitment").fill_null(0),
            (pl.col("projected").fill_null(0) - pl.col("commitment").fill_null(0)).clip(lower_bound=0).alias("baseline"),
        ]).with_row_index("_push_row")
        selected_levels: dict[int, float] = {}
        for _, group in work.partition_by(["material", "sending"], as_dict=True, maintain_order=True).items():
            rows = group.to_dicts()
            available_soh = float(rows[0]["available"] or 0)
            selected = 1.2
            for level in (1.2, 1.5, 2.0, 2.5, 3.0):
                need = sum(max(0, level * float(row["receiving_safety"] or 0) - float(row["baseline"] or 0)) for row in rows)
                if need <= available_soh + 1e-9:
                    selected = level
                else:
                    break
            selected_levels.update({int(row["_push_row"]): selected for row in rows})
        work = work.with_columns(
            pl.Series("push_level", [selected_levels[index] for index in work.get_column("_push_row").to_list()], dtype=pl.Float64)
        ).with_columns((pl.col("push_level") * pl.col("receiving_safety") - pl.col("baseline")).clip(lower_bound=0).alias("need"))
        work = work.with_columns(pl.col("need").sum().over(["material", "sending"]).alias("total_need")).with_columns((pl.col("available") * pl.col("need") / pl.col("total_need").replace(0, None)).floor().fill_null(0).cast(pl.Int64).alias("push_qty")).filter(pl.col("push_qty") > 0)
        if work.is_empty(): return self._empty(plan.columns)
        output = work.select([pl.lit(day.date()).alias("date"), "material", "sending", "receiving", pl.lit(0).alias("demand_qty"), pl.when(pl.col("model") == "push").then(pl.lit("push replenishment")).otherwise(pl.lit("soft push replenishment")).alias("demand_element"), pl.col("push_qty").alias("planned_qty"), pl.col("push_qty").alias("deployed_qty_invCon"), pl.lit(0).alias("deploy_qty_with_plan_order"), pl.lit(0).alias("deploy_from_in_transit"), pl.lit(0).alias("deploy_from_open_deployment_inbound"), pl.lit(0).alias("deploy_from_future_production"), "planned_delivery_date", pl.col("receiving").alias("orig_location"), "leadtime", pl.lit(True).alias("is_cross_node")])
        return output.select([pl.col(c) if c in output.columns else pl.lit(0).alias(c) for c in plan.columns])

    def apply_space(self, plan, space, priority):
        if plan.is_empty() or space.is_empty() or not {"receiving", "date", "max_qty"}.issubset(space.columns): return plan.with_columns([pl.col("deployed_qty_invCon").cast(pl.Int64).alias("deployed_qty"), pl.lit(float("inf")).alias("quota")]), self._empty([])
        quota = {(r["receiving"], r["date"]): float(r["max_qty"] or float("inf")) for r in space.select(["receiving", "date", "max_qty"]).to_dicts()}; rows = plan.with_columns(pl.col("deployed_qty_invCon").cast(pl.Int64).alias("deployed_qty")).to_dicts(); grouped=defaultdict(list)
        for i, row in enumerate(rows):
            row["quota"] = quota.get((row["receiving"], row["date"]), float("inf"))
            if row["sending"] != row["receiving"]: grouped[(row["receiving"], row["date"])] .append(i)
        gaps=[]
        for key, indexes in grouped.items():
            left = quota.get(key, float("inf"))
            ordered = sorted(indexes, key=lambda j: (priority.get(str(rows[j]["demand_element"]), 99), str(rows[j]["material"]), str(rows[j]["sending"]), str(rows[j]["demand_element"])))
            for _, block in groupby(ordered, key=lambda j: priority.get(str(rows[j]["demand_element"]), 99)):
                block = list(block)
                needs = [int(rows[i]["deployed_qty_invCon"] or 0) for i in block]
                total = sum(needs)
                if left >= total:
                    allocation = needs
                    left -= total
                else:
                    allocation = [min(int(np.floor(left * need / total)), need) if total else 0 for need in needs]
                    left -= sum(allocation)
                for i, value in zip(block, allocation):
                    rows[i]["deployed_qty"] = value
                if left <= 0:
                    break
            for i in indexes:
                old, new = int(rows[i]["deployed_qty_invCon"] or 0), int(rows[i]["deployed_qty"] or 0)
                if new < old: gaps.append({"date": rows[i]["date"], "sending": rows[i]["sending"], "receiving": rows[i]["receiving"], "material": rows[i]["material"], "demand_qty": rows[i]["demand_qty"], "demand_element": rows[i]["demand_element"], "unfulfilled_qty": old-new, "reason":"space constraint"})
        return pl.DataFrame(rows), pl.DataFrame(gaps) if gaps else self._empty([])

    def finalise_result(self, plan, unfulfilled, available, today_transit, shipment, validation, day):
        deployed = plan.filter(pl.col("sending") != pl.col("receiving")).group_by(["material", "sending"]).agg(pl.col("deployed_qty_invCon").sum().alias("deployed_qty")).rename({"sending":"node"}) if not plan.is_empty() else pl.DataFrame(schema={"material":pl.Utf8,"node":pl.Utf8,"deployed_qty":pl.Int64})
        soh = available.select(["material","node","stock","production","delivery"]).join(today_transit.rename({"qty":"in_transit"}),on=["material","node"],how="full",coalesce=True).join(shipment.rename({"qty":"today_shipment"}),on=["material","node"],how="full",coalesce=True).join(deployed,on=["material","node"],how="full",coalesce=True).fill_null(0).with_columns((pl.col("stock")+pl.col("production")+pl.col("delivery")+pl.col("in_transit")-pl.col("today_shipment")-pl.col("deployed_qty")).alias("ending_soh")).rename({"node":"location","stock":"beginning_soh","delivery":"delivery_gr"}).with_columns(pl.lit(day.date()).alias("date"))
        stable=lambda frame, keys: frame.sort([k for k in keys if k in frame.columns]) if not frame.is_empty() else frame
        return {"deployment_plan":self._out(stable(plan,["date","material","sending","receiving","planned_delivery_date","demand_element"])),"unfulfilled_log":self._out(stable(unfulfilled,["date","sending","receiving","demand_element"])),"stock_on_hand_log":self._out(stable(soh,["date","material","location"])),"validation_log":pd.DataFrame(validation),"statistics":{"deployment_count":plan.height,"unfulfilled_count":unfulfilled.height,"processed_dates":1}}

    def empty_result(self):
        self.result={"deployment_plan":pd.DataFrame(),"unfulfilled_log":pd.DataFrame(),"stock_on_hand_log":pd.DataFrame(),"validation_log":pd.DataFrame(),"statistics":{"deployment_count":0,"unfulfilled_count":0,"processed_dates":1}}
        return self.result


__all__ = ["_PandasBackend", "_PolarsBackend"]
