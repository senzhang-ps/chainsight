"""Module3 独立 pandas / Polars 计算后端。"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import pandas as pd
import polars as pl

from ...utils.df_convert import pandas_to_polars
from ...utils.defaults import DEFAULT_LSK, DEFAULT_PTF
from ...utils.ptf_lsk import build_ptf_lsk_cache
from .lead_time import compute_root_horizon
from .utils import apportion_largest_remainder, apply_moq_rv


_RESULT_COLUMNS = [
    "material", "location", "requirement_date", "quantity",
    "demand_element", "layer", "simulation_date", "horizon_days",
]
_FACT_VERSION = 1
_DEMAND_ELEMENT = {
    "AO": "net demand for AO",
    "FC": "net demand for forecast",
    "SS": "net demand for safety",
}


def _canonicalise_refactor_lead_time(frame: pd.DataFrame) -> pd.DataFrame:
    """仅在重构 M3 内部恢复 legacy MRP 所需的提前期列名。

    Excel 运行时配置会把 ``PDT/GR/MCT`` 规整为小写，但 legacy MRP
    提前期公式仍读取大写列。不得修改全局 ConfigReader，否则会影响 legacy
    入口；仅在 M3 refactor backend 的静态副本中补齐 canonical 列。
    """
    data = frame.copy(deep=True)
    rename = {
        source: target
        for source, target in (("pdt", "PDT"), ("gr", "GR"), ("mct", "MCT"))
        if source in data.columns and target not in data.columns
    }
    return data.rename(columns=rename)


class _PandasBackend:
    """M3 pandas 实现；网络、路线、窗口需求由 M5 的 PlanningFacts 提供。"""

    engine = "pandas"

    def __init__(self, owner=None):
        self.owner = owner
        self.static: dict[str, pd.DataFrame] = {}
        self.result: dict = {}

    @staticmethod
    def empty(columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=columns)

    def empty_result(self) -> dict:
        return {"net_demand_df": self.empty(_RESULT_COLUMNS), "net_demand_count": 0}

    @staticmethod
    def _normalise(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
        if frame is None:
            return pd.DataFrame()
        data = frame.copy(deep=True)
        for column in ("material", "location", "sending", "receiving", "node", "upstream"):
            if column in data:
                values = data[column].fillna("").astype(str).str.strip()
                if column != "material":
                    numeric = values.str.fullmatch(r"\d+", na=False)
                    values.loc[numeric] = values.loc[numeric].str.zfill(4)
                data[column] = values
        for column in ("date", "available_date", "actual_delivery_date", "planned_deployment_date"):
            if column in data:
                data[column] = pd.to_datetime(data[column], errors="coerce").dt.normalize()
        return data

    def normalise_static_config(self, datas: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        static = {
            name: self._normalise(datas.get(name, pd.DataFrame()))
            for name in self.owner.schema
        }
        static["Global_LeadTime"] = _canonicalise_refactor_lead_time(
            static.get("Global_LeadTime", pd.DataFrame())
        )
        return static

    def store_static_state(self, static: dict[str, pd.DataFrame]):
        self.static = static

    def planning_facts(self, facts: dict) -> dict:
        if int(facts.get("version", -1)) != _FACT_VERSION:
            raise RuntimeError(f"不支持的 PlanningFacts 版本: {facts.get('version')}")
        output = dict(facts)
        for name in ("active_network", "routes", "node_horizon", "direct_demand"):
            output[name] = self._normalise(output.get(name, pd.DataFrame()))
        day = pd.Timestamp(output["simulation_date"]).normalize()
        upstream_keys = {
            (str(row.material), str(row.node))
            for row in output["active_network"].itertuples(index=False)
            if pd.notna(getattr(row, "upstream", None)) and str(row.upstream).strip()
        }
        horizons = output["node_horizon"].copy()
        if not horizons.empty:
            cache = build_ptf_lsk_cache(self.static.get("M4_MaterialLocationLineCfg", pd.DataFrame()))
            for index, row in horizons.iterrows():
                key = (str(row["material"]), str(row["node"]))
                if key not in upstream_keys:
                    days = compute_root_horizon(
                        key[0], key[1], self.static.get("Global_LeadTime", pd.DataFrame()),
                        self.static.get("M4_MaterialLocationLineCfg", pd.DataFrame()), cache,
                    )
                    horizons.at[index, "horizon_end"] = day + pd.Timedelta(days=days)
        output["node_horizon"] = horizons
        # M5 部署计划按整数处理数量；M3 旧算法必须保留安全库存的小数，
        # 因此在共享的窗口事实中只重建 safety 部分，不重复网络/路线计算。
        direct = output["direct_demand"]
        non_safety = direct.loc[
            ~direct.get("demand_element", pd.Series("", index=direct.index)).astype(str).str.casefold().eq("safety")
        ].copy()
        safety = self.static.get("M3_SafetyStock", pd.DataFrame())
        columns = ["material", "node", "receiving", "demand_element", "demand_qty", "requirement_date", "orig_location"]
        if safety.empty or horizons.empty or not {"material", "location", "date"}.issubset(safety.columns):
            safety_demand = self.empty(columns)
        else:
            safety_demand = safety.merge(
                horizons.rename(columns={"node": "location"}), on=["material", "location"], how="inner",
            )
            safety_demand = safety_demand.loc[safety_demand["date"].eq(safety_demand["horizon_end"])].copy()
            quantity = "safety_stock_qty" if "safety_stock_qty" in safety_demand else "quantity"
            safety_demand = pd.DataFrame({
                "material": safety_demand["material"], "node": safety_demand["location"],
                "receiving": safety_demand["location"], "demand_element": "safety",
                "demand_qty": pd.to_numeric(safety_demand[quantity], errors="coerce").fillna(0),
                "requirement_date": safety_demand["date"], "orig_location": safety_demand["location"],
            })
            safety_demand = safety_demand.loc[safety_demand["demand_qty"].gt(0)].groupby(
                [column for column in columns if column != "demand_qty"], as_index=False, sort=False,
            )["demand_qty"].sum()
        output["direct_demand"] = pd.concat([non_safety, safety_demand], ignore_index=True, sort=False)
        return output

    def supply_views(self, ctx, day: pd.Timestamp) -> dict[str, pd.DataFrame]:
        text = day.strftime("%Y-%m-%d")
        open_deployment_getter = getattr(
            ctx, "get_m3_open_deployment_view", ctx.get_open_deployment_view,
        )
        return {
            "beginning_inventory": self._normalise(ctx.get_beginning_inventory_view(text)),
            "in_transit": self._normalise(ctx.get_planning_intransit_view(text)),
            "delivery_gr": self._normalise(ctx.get_delivery_gr_view(text)),
            "all_production": self._normalise(ctx.get_all_production_view(text)),
            "open_deployment": self._normalise(open_deployment_getter(text)),
            "shipment": self._normalise(ctx.get_shipment_log_view(text)),
            "delivery_shipment": self._normalise(ctx.get_delivery_shipment_log_view(text)),
        }

    @staticmethod
    def _mask(frame: pd.DataFrame, material: str, location_column: str, node: str) -> pd.Series:
        return (
            frame.get("material", pd.Series("", index=frame.index)).eq(material)
            & frame.get(location_column, pd.Series("", index=frame.index)).eq(node)
        )

    @staticmethod
    def _sum(frame: pd.DataFrame, mask: pd.Series, candidates=("quantity", "deployed_qty")) -> float:
        if frame.empty:
            return 0.0
        column = next((name for name in candidates if name in frame), None)
        if column is None:
            return 0.0
        return float(pd.to_numeric(frame.loc[mask, column], errors="coerce").fillna(0).sum())

    def _available_supply(self, material: str, node: str, supply: dict[str, pd.DataFrame]) -> float:
        bi, transit, gr = supply["beginning_inventory"], supply["in_transit"], supply["delivery_gr"]
        production, opened = supply["all_production"], supply["open_deployment"]
        shipment, delivery_shipment = supply["shipment"], supply["delivery_shipment"]
        total = self._sum(bi, self._mask(bi, material, "location", node))
        total += self._sum(transit, self._mask(transit, material, "receiving", node))
        total += self._sum(gr, self._mask(gr, material, "receiving", node))
        total += self._sum(production, self._mask(production, material, "location", node), ("con_planned_qty", "produced_qty", "quantity"))
        total -= self._sum(shipment, self._mask(shipment, material, "location", node))
        if not delivery_shipment.empty:
            column = "sending" if "sending" in delivery_shipment else "location"
            total -= self._sum(delivery_shipment, self._mask(delivery_shipment, material, column, node))
        if not opened.empty:
            cross = opened.get("sending", pd.Series("", index=opened.index)).ne(
                opened.get("receiving", pd.Series("", index=opened.index))
            )
            total -= self._sum(opened, cross & self._mask(opened, material, "sending", node))
            date_col = "date" if "date" in opened else "planned_deployment_date"
            inbound_date = pd.to_datetime(
                opened.get(date_col, pd.Series(pd.NaT, index=opened.index)),
                errors="coerce",
            ).dt.normalize()
            total += self._sum(
                opened,
                cross & inbound_date.gt(pd.Timestamp(self.owner.simulation_date).normalize())
                & self._mask(opened, material, "receiving", node),
            )
        return total

    @staticmethod
    def _direct_by_node(direct: pd.DataFrame, material: str, node: str) -> dict[str, float]:
        values = {"AO": 0.0, "FC": 0.0, "SS": 0.0}
        if direct.empty:
            return values
        rows = direct.loc[(direct["material"] == material) & (direct["node"] == node)]
        for row in rows.itertuples(index=False):
            element = str(getattr(row, "demand_element", "")).casefold()
            # M5 还会提供 normal 等部署需求；legacy M3 仅将 AO 订单纳入
            # AO 缺口，SDL forecast 与窗口末 safety 分别按原语义处理。
            if element not in {"ao", "forecast", "safety"}:
                continue
            key = "AO" if element == "ao" else "SS" if element == "safety" else "FC"
            values[key] += float(getattr(row, "demand_qty", 0) or 0)
        return values

    @staticmethod
    def _route_params(routes: pd.DataFrame, material: str, sending: str, receiving: str) -> tuple[int, int]:
        rows = routes.loc[
            (routes["material"] == material)
            & (routes["sending"] == sending)
            & (routes["receiving"] == receiving)
        ]
        if rows.empty:
            return 1, 1
        return max(0, int(rows.iloc[0].get("moq", 1) or 1)), max(1, int(rows.iloc[0].get("rv", 1) or 1))

    @staticmethod
    def _grouped(frame: pd.DataFrame, location_column: str,
                 candidates=("quantity", "deployed_qty")) -> dict[tuple[str, str], float]:
        """按物料/节点预聚合供给，避免在每个 MRP 节点重复筛选全表。"""
        if frame.empty or not {"material", location_column}.issubset(frame.columns):
            return {}
        quantity_column = next((name for name in candidates if name in frame.columns), None)
        if quantity_column is None:
            return {}
        grouped = frame.assign(
            _m3_qty=pd.to_numeric(frame[quantity_column], errors="coerce").fillna(0)
        ).groupby(["material", location_column], sort=False, dropna=False)["_m3_qty"].sum()
        return {
            (str(material), str(node)): float(quantity)
            for (material, node), quantity in grouped.items()
        }

    def _available_by_node(self, day: pd.Timestamp,
                           supply: dict[str, pd.DataFrame]) -> dict[tuple[str, str], float]:
        """构造一次性的节点可用供给账本，语义与 ``_available_supply`` 保持一致。"""
        available = self._grouped(supply["beginning_inventory"], "location")
        additions = (
            ("in_transit", "receiving", ("quantity", "deployed_qty")),
            ("delivery_gr", "receiving", ("quantity", "deployed_qty")),
            ("all_production", "location", ("con_planned_qty", "produced_qty", "quantity")),
        )
        for name, location_column, candidates in additions:
            for key, quantity in self._grouped(supply[name], location_column, candidates).items():
                available[key] = available.get(key, 0.0) + quantity
        for name, location_column in (("shipment", "location"), ("delivery_shipment", "sending")):
            frame = supply[name]
            if name == "delivery_shipment" and location_column not in frame.columns:
                location_column = "location"
            for key, quantity in self._grouped(frame, location_column).items():
                available[key] = available.get(key, 0.0) - quantity
        opened = supply["open_deployment"]
        if not opened.empty and {"material", "sending", "receiving"}.issubset(opened.columns):
            cross = opened.loc[opened["sending"].ne(opened["receiving"])]
            for key, quantity in self._grouped(cross, "sending").items():
                available[key] = available.get(key, 0.0) - quantity
            date_column = "date" if "date" in cross.columns else "planned_deployment_date"
            if date_column in cross.columns:
                future = cross.loc[pd.to_datetime(cross[date_column], errors="coerce").dt.normalize().gt(day)]
                for key, quantity in self._grouped(future, "receiving").items():
                    available[key] = available.get(key, 0.0) + quantity
        return available

    def calculate_layers(self, day: pd.Timestamp, facts: dict, supply: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        direct, routes = facts["direct_demand"], facts["routes"]
        layer_map: dict[tuple[str, str], int] = facts["layer_map"]
        active = facts["active_network"]
        horizon = {
            (str(row.material), str(row.node)): row.horizon_end
            for row in facts["node_horizon"].itertuples(index=False)
        }
        upstream = {
            (str(row.material), str(row.node)): str(row.upstream)
            for row in active.itertuples(index=False)
            if pd.notna(getattr(row, "upstream", None)) and str(row.upstream).strip()
        }
        direct_map = defaultdict(lambda: {"AO": 0.0, "FC": 0.0, "SS": 0.0})
        if not direct.empty:
            for row in direct.itertuples(index=False):
                element = str(getattr(row, "demand_element", "")).casefold()
                if element not in {"ao", "forecast", "safety"}:
                    continue
                kind = "AO" if element == "ao" else "SS" if element == "safety" else "FC"
                direct_map[(str(row.material), str(row.node))][kind] += float(
                    getattr(row, "demand_qty", 0) or 0
                )
        route_map = {
            (str(row.material), str(row.sending), str(row.receiving)): (
                max(0, int(getattr(row, "moq", 1) or 1)),
                max(1, int(getattr(row, "rv", 1) or 1)),
            )
            for row in routes.itertuples(index=False)
        }
        nodes_by_layer: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for node_key, layer in layer_map.items():
            nodes_by_layer[layer].append(node_key)
        available_by_node = self._available_by_node(day, supply)
        gaps = defaultdict(lambda: {"AO": 0.0, "FC": 0.0, "SS": 0.0})
        records: list[dict[str, Any]] = []
        for layer in sorted(facts["layers"], reverse=True):
            for material, node in sorted(nodes_by_layer[layer]):
                demand = dict(direct_map[(material, node)])
                for kind in demand:
                    demand[kind] += gaps[(material, node)][kind]
                available = available_by_node.get((material, node), 0.0)
                shortages: dict[str, float] = {}
                for kind in ("AO", "FC", "SS"):
                    shortages[kind] = max(demand[kind] - available, 0.0)
                    available = max(available - min(available, demand[kind]), 0.0)
                horizon_days = max(1, (pd.Timestamp(horizon.get((material, node), day)) - day).days)
                for kind, qty in shortages.items():
                    if qty > 0:
                        records.append({
                            "material": material, "location": node,
                            "requirement_date": day + pd.Timedelta(days=1),
                            "quantity": -qty,
                            "demand_element": _DEMAND_ELEMENT[kind],
                            "layer": layer, "simulation_date": day,
                            "horizon_days": horizon_days,
                        })
                parent = upstream.get((material, node))
                total = sum(shortages.values())
                if parent and total > 0:
                    moq, rv = route_map.get((material, parent, node), (1, 1))
                    split = apportion_largest_remainder(
                        [shortages["AO"], shortages["FC"], shortages["SS"]],
                        apply_moq_rv(total, moq, rv),
                    )
                    for kind, qty in zip(("AO", "FC", "SS"), split):
                        gaps[(material, parent)][kind] += float(qty)
        return records

    def finalise_result(self, records: list[dict[str, Any]]) -> dict:
        if not records:
            return self.empty_result()
        frame = pd.DataFrame(records)
        grouped = frame.groupby(
            ["material", "location", "requirement_date", "demand_element", "layer"],
            as_index=False, sort=False,
        ).agg({"quantity": "sum", "simulation_date": "first", "horizon_days": "first"})
        grouped = grouped.loc[:, _RESULT_COLUMNS].sort_values(
            ["material", "location", "requirement_date", "demand_element", "layer"],
            kind="mergesort",
        ).reset_index(drop=True)
        return {"net_demand_df": grouped, "net_demand_count": len(grouped)}


class _PolarsBackend:
    """独立 Polars M3 后端；只在 StateContext 输入/输出边界转换 pandas。"""

    engine = "polars"

    def __init__(self, owner=None):
        self.owner = owner
        self.static: dict[str, pl.DataFrame] = {}
        self.result: dict = {}

    @staticmethod
    def empty(columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=columns)

    def empty_result(self) -> dict:
        return {"net_demand_df": self.empty(_RESULT_COLUMNS), "net_demand_count": 0}

    @staticmethod
    def _pl(frame: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
        if isinstance(frame, pl.DataFrame):
            return frame
        if frame.empty:
            return pl.DataFrame(schema={column: pl.Null for column in frame.columns})
        return pandas_to_polars(frame)

    @staticmethod
    def _normalise(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
        if frame is None:
            return pd.DataFrame()
        data = frame.copy(deep=True)
        for column in ("material", "location", "sending", "receiving", "node", "upstream"):
            if column in data:
                values = data[column].fillna("").astype(str).str.strip()
                if column != "material":
                    numeric = values.str.fullmatch(r"\d+", na=False)
                    values.loc[numeric] = values.loc[numeric].str.zfill(4)
                data[column] = values
        for column in ("date", "available_date", "actual_delivery_date", "planned_deployment_date"):
            if column in data:
                data[column] = pd.to_datetime(data[column], errors="coerce").dt.normalize()
        return data

    def normalise_static_config(self, datas: dict[str, pd.DataFrame]) -> dict[str, pl.DataFrame]:
        static = {
            name: self._pl(self._normalise(datas.get(name, pd.DataFrame())))
            for name in self.owner.schema
        }
        lead_time = static.get("Global_LeadTime", pl.DataFrame())
        if "pdt" in lead_time.columns and "PDT" not in lead_time.columns:
            lead_time = lead_time.rename({"pdt": "PDT"})
        if "gr" in lead_time.columns and "GR" not in lead_time.columns:
            lead_time = lead_time.rename({"gr": "GR"})
        if "mct" in lead_time.columns and "MCT" not in lead_time.columns:
            lead_time = lead_time.rename({"mct": "MCT"})
        static["Global_LeadTime"] = lead_time
        return static

    def store_static_state(self, static: dict[str, pl.DataFrame]):
        self.static = static

    def planning_facts(self, facts: dict) -> dict:
        if int(facts.get("version", -1)) != _FACT_VERSION:
            raise RuntimeError(f"不支持的 PlanningFacts 版本: {facts.get('version')}")
        checked = dict(facts)
        for name in ("active_network", "routes", "node_horizon", "direct_demand"):
            checked[name] = self._pl(self._normalise(checked.get(name, pd.DataFrame())))
        day = pd.Timestamp(checked["simulation_date"]).normalize()
        upstream_keys = {
            (str(row["material"]), str(row["node"]))
            for row in checked["active_network"].to_dicts()
            if row.get("upstream")
        }
        lead_max: dict[str, tuple[int, int, int]] = {}
        for row in self.static.get("Global_LeadTime", pl.DataFrame()).to_dicts():
            sending = str(row.get("sending") or "")
            current = lead_max.get(sending, (0, 0, 0))
            lead_max[sending] = tuple(max(current[i], int(row.get(key) or 0)) for i, key in enumerate(("PDT", "GR", "MCT")))
        ptf_lsk: dict[tuple[str, str], tuple[int, int]] = {}
        for row in self.static.get("M4_MaterialLocationLineCfg", pl.DataFrame()).to_dicts():
            ptf_lsk[(str(row.get("material") or ""), str(row.get("location") or ""))] = (
                int(row.get("ptf", row.get("PTF", DEFAULT_PTF)) or DEFAULT_PTF),
                int(row.get("lsk", row.get("LSK", DEFAULT_LSK)) or DEFAULT_LSK),
            )
        horizon_rows = []
        for row in checked["node_horizon"].to_dicts():
            key = (str(row["material"]), str(row["node"]))
            horizon_end = row.get("horizon_end")
            if key not in upstream_keys:
                ptf, lsk = ptf_lsk.get(key, (DEFAULT_PTF, DEFAULT_LSK))
                pdt, gr, mct = lead_max.get(key[1], (0, 0, 0))
                horizon_end = day + pd.Timedelta(days=max(1, max(mct, pdt + gr) + ptf + lsk - 1))
            horizon_rows.append({"material": key[0], "node": key[1], "horizon_end": horizon_end})
        checked["node_horizon"] = pl.DataFrame(horizon_rows) if horizon_rows else checked["node_horizon"]
        direct = checked["direct_demand"]
        non_safety = direct.filter(
            pl.col("demand_element").cast(pl.Utf8, strict=False).str.to_lowercase() != "safety"
        ) if "demand_element" in direct.columns else direct
        safety = self.static.get("M3_SafetyStock", pl.DataFrame())
        columns = ["material", "node", "receiving", "demand_element", "demand_qty", "requirement_date", "orig_location"]
        if safety.is_empty() or checked["node_horizon"].is_empty() or not {"material", "location", "date"}.issubset(safety.columns):
            safety_demand = pl.DataFrame(schema={column: pl.Null for column in columns})
        else:
            quantity = "safety_stock_qty" if "safety_stock_qty" in safety.columns else "quantity"
            safety_demand = safety.join(
                checked["node_horizon"].rename({"node": "location"}), on=["material", "location"], how="inner",
            ).filter(pl.col("date") == pl.col("horizon_end")).select([
                "material", pl.col("location").alias("node"), pl.col("location").alias("receiving"),
                pl.lit("safety").alias("demand_element"), pl.col(quantity).cast(pl.Float64, strict=False).fill_null(0).alias("demand_qty"),
                pl.col("date").alias("requirement_date"), pl.col("location").alias("orig_location"),
            ]).filter(pl.col("demand_qty") > 0).group_by(
                [column for column in columns if column != "demand_qty"], maintain_order=True,
            ).agg(pl.col("demand_qty").sum())
        checked["direct_demand"] = pl.concat([non_safety, safety_demand], how="diagonal_relaxed")
        return checked

    def supply_views(self, ctx, day: pd.Timestamp) -> dict[str, pl.DataFrame]:
        text = day.strftime("%Y-%m-%d")
        open_deployment_getter = getattr(
            ctx, "get_m3_open_deployment_view", ctx.get_open_deployment_view,
        )
        raw = {
            "beginning_inventory": ctx.get_beginning_inventory_view(text),
            "in_transit": ctx.get_planning_intransit_view(text),
            "delivery_gr": ctx.get_delivery_gr_view(text),
            "all_production": ctx.get_all_production_view(text),
            "open_deployment": open_deployment_getter(text),
            "shipment": ctx.get_shipment_log_view(text),
            "delivery_shipment": ctx.get_delivery_shipment_log_view(text),
        }
        return {name: self._pl(self._normalise(frame)) for name, frame in raw.items()}

    @staticmethod
    def _grouped(frame: pl.DataFrame, keys: list[str], candidates=("quantity", "deployed_qty")) -> dict[tuple[str, str], float]:
        if frame.is_empty() or not set(keys).issubset(frame.columns):
            return {}
        column = next((name for name in candidates if name in frame.columns), None)
        if column is None:
            return {}
        return {
            (str(row[keys[0]]), str(row[keys[1]])): float(row["qty"] or 0)
            for row in frame.group_by(keys).agg(
                pl.col(column).cast(pl.Float64, strict=False).fill_null(0).sum().alias("qty")
            ).to_dicts()
        }

    def calculate_layers(self, day: pd.Timestamp, facts: dict, supply: dict[str, pl.DataFrame]) -> list[dict[str, Any]]:
        direct = facts["direct_demand"]
        routes = facts["routes"]
        horizon = {
            (str(row["material"]), str(row["node"])): row["horizon_end"]
            for row in facts["node_horizon"].to_dicts()
        }
        upstream = {
            (str(row["material"]), str(row["node"])): str(row["upstream"])
            for row in facts["active_network"].to_dicts()
            if row.get("upstream")
        }
        direct_map = defaultdict(lambda: {"AO": 0.0, "FC": 0.0, "SS": 0.0})
        for row in direct.to_dicts():
            element = str(row.get("demand_element") or "").casefold()
            if element not in {"ao", "forecast", "safety"}:
                continue
            kind = "AO" if element == "ao" else "SS" if element == "safety" else "FC"
            direct_map[(str(row["material"]), str(row["node"]))][kind] += float(row.get("demand_qty") or 0)
        available = self._grouped(supply["beginning_inventory"], ["material", "location"])
        for name, column in (("in_transit", "receiving"), ("delivery_gr", "receiving"), ("all_production", "location")):
            for key, qty in self._grouped(supply[name], ["material", column], ("con_planned_qty", "produced_qty", "quantity")).items():
                available[key] = available.get(key, 0) + qty
        for name, column in (("shipment", "location"), ("delivery_shipment", "sending")):
            for key, qty in self._grouped(supply[name], ["material", column]).items():
                available[key] = available.get(key, 0) - qty
        opened = supply["open_deployment"]
        if not opened.is_empty() and {"material", "sending", "receiving"}.issubset(opened.columns):
            cross = opened.filter(pl.col("sending") != pl.col("receiving"))
            for key, qty in self._grouped(cross, ["material", "sending"]).items():
                available[key] = available.get(key, 0) - qty
            date_col = "date" if "date" in cross.columns else "planned_deployment_date"
            future = cross.filter(
                pl.col(date_col).cast(pl.Date, strict=False) > day.date()
            ) if date_col in cross.columns else cross.head(0)
            for key, qty in self._grouped(future, ["material", "receiving"]).items():
                available[key] = available.get(key, 0) + qty
        route_map = {
            (str(row["material"]), str(row["sending"]), str(row["receiving"])): (
                max(0, int(row.get("moq") or 1)), max(1, int(row.get("rv") or 1)),
            )
            for row in routes.to_dicts()
        }
        gaps = defaultdict(lambda: {"AO": 0.0, "FC": 0.0, "SS": 0.0})
        records: list[dict[str, Any]] = []
        for layer in sorted(facts["layers"], reverse=True):
            for material, node in sorted(key for key, value in facts["layer_map"].items() if value == layer):
                demand = dict(direct_map[(material, node)])
                for kind in demand:
                    demand[kind] += gaps[(material, node)][kind]
                left = available.get((material, node), 0.0)
                shortages: dict[str, float] = {}
                for kind in ("AO", "FC", "SS"):
                    shortages[kind] = max(demand[kind] - left, 0.0)
                    left = max(left - min(left, demand[kind]), 0.0)
                horizon_days = max(1, (pd.Timestamp(horizon.get((material, node), day.date())) - day).days)
                for kind, qty in shortages.items():
                    if qty > 0:
                        records.append({
                            "material": material, "location": node,
                            "requirement_date": day + pd.Timedelta(days=1),
                            "quantity": -qty,
                            "demand_element": _DEMAND_ELEMENT[kind],
                            "layer": layer, "simulation_date": day,
                            "horizon_days": horizon_days,
                        })
                parent = upstream.get((material, node))
                total = sum(shortages.values())
                if parent and total > 0:
                    moq, rv = route_map.get((material, parent, node), (1, 1))
                    split = apportion_largest_remainder(
                        [shortages["AO"], shortages["FC"], shortages["SS"]],
                        apply_moq_rv(total, moq, rv),
                    )
                    for kind, qty in zip(("AO", "FC", "SS"), split):
                        gaps[(material, parent)][kind] += float(qty)
        return records

    def finalise_result(self, records: list[dict[str, Any]]) -> dict:
        if not records:
            return self.empty_result()
        frame = pl.DataFrame(records).group_by(
            ["material", "location", "requirement_date", "demand_element", "layer"],
            maintain_order=True,
        ).agg([
            pl.col("quantity").sum(),
            pl.col("simulation_date").first(),
            pl.col("horizon_days").first(),
        ]).select(_RESULT_COLUMNS).sort(
            ["material", "location", "requirement_date", "demand_element", "layer"]
        )
        output = frame.to_pandas()
        return {"net_demand_df": output, "net_demand_count": len(output)}
