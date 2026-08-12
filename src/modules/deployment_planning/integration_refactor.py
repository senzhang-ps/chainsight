"""Module 5 的独立 pandas 集成实现。

该模块不调用旧 M5 的 helper。核心数据流为：直接需求事实表 -> 每层残余
gap 事实表 -> 库存/未来供给分配 -> 调拨计划事实表。唯一保留的顺序依赖是网络
层级中的残余 gap 上推；层内的需求构造、聚合和输出均为 pandas 批处理。
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..module import Module

logger = logging.getLogger("SupplyChainSimulation")


class ModuleFive(Module):
	"""M5 部署规划。

	``state_context`` 保存可变仿真状态；``orch`` 是配置注入
	与持久化使用的新 ``Orch``。静态配置只在 :meth:`prepare` 中由
	``Orch.load_datas`` 注入一次。
	"""

	schema = {
		"M3_SafetyStock": {"material": "str", "location": "str", "date": "datetime", "quantity": "float"},
		"Global_Network": {"material": "str", "location": "str", "sourcing": "str", "eff_from": "datetime", "eff_to": "datetime"},
		"Global_LeadTime": {"sending": "str", "receiving": "str", "PDT": "int", "GR": "int", "MCT": "int"},
		"Global_DemandPriority": {"demand_element": "str", "priority": "int"},
		"M5_PushPullModel": {"material": "str", "sending": "str", "model": "str"},
		"M5_DeployConfig": {"material": "str", "sending": "str"},
		"M5_SupplyDemandLog": {"material": "str", "location": "str", "date": "datetime", "demand_element": "str", "quantity": "float"},
		"M4_MaterialLocationLineCfg": {"material": "str", "location": "str"},
	}

	_ID_COLUMNS = ("material", "location", "sending", "receiving", "sourcing")
	_DATE_COLUMNS = (
		"date", "simulation_date", "available_date", "actual_delivery_date",
		"planned_deployment_date", "eff_from", "eff_to",
	)

	def __init__(
		self,
		simulation_date,
		simulation_start_date,
		state_context=None,
		orch=None,
		verbose: bool = False,
		config: Optional[dict] = None,
	):
		super().__init__(simulation_date, orch, "M5", verbose, config=config)
		self.state_context = state_context
		self.simulation_start_date = pd.Timestamp(simulation_start_date).normalize()
		self.static: dict[str, pd.DataFrame] = {}
		self.static_config: dict[str, pd.DataFrame] = {}
		self.layer_map: dict[tuple[str, str], int] = {}
		self.location_to_layer: dict[tuple[str, str], int] = {}
		self.layers: list[int] = []
		self._prepared = False
		self._empty_result()

	# ------------------------------------------------------------------
	# 配置与通用 DataFrame 基础设施（不复用旧 M5 的 utils）。
	# ------------------------------------------------------------------
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
	def _empty(columns: list[str]) -> pd.DataFrame:
		return pd.DataFrame(columns=columns)

	@staticmethod
	def _quantity(frame: pd.DataFrame, candidates: tuple[str, ...] = ("quantity",)) -> pd.Series:
		"""按候选列优先级逐行合并数量，而非只选择第一个存在的列。

		``Production`` 会拼接 StateContext GR（``quantity``）和 M4 计划
		（``produced_qty``）。拼接后两列都存在，单纯选择 ``produced_qty``
		会把 GR 行的实际收货错误地变为零。
		"""
		quantity = pd.Series(np.nan, index=frame.index, dtype=float)
		for column in candidates:
			if column in frame:
				quantity = quantity.fillna(pd.to_numeric(frame[column], errors="coerce"))
		return quantity.fillna(0).clip(lower=0).astype(np.int64)

	@staticmethod
	def _number(frame: pd.DataFrame, column: str, default: int) -> pd.Series:
		source = frame[column] if column in frame else pd.Series(default, index=frame.index)
		return pd.to_numeric(source, errors="coerce").fillna(default)

	@staticmethod
	def _stable(frame: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
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
		"""按物料构建确定性 BFS 层级；异常网络保留在最大层，供验证日志报告。"""
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
			roots = sorted(node for node in nodes if not parents[node])
			queue = deque((node, 0) for node in roots)
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

	def prepare(self):
		if self._prepared:
			return
		if self.orchestrator is None:
			raise RuntimeError("ModuleFive.prepare() 需要由 Orch 注入配置")
		self.orchestrator.load_datas(self)
		names = {
			"SafetyStock": "M3_SafetyStock", "Network": "Global_Network",
			"LeadTime": "Global_LeadTime", "DemandPriority": "Global_DemandPriority",
			"PushPullModel": "M5_PushPullModel", "DeployConfig": "M5_DeployConfig",
			"SupplyDemandLog": "M5_SupplyDemandLog",
			"MaterialLocation": "M4_MaterialLocationLineCfg",
		}
		self.static = {target: self._normalise(self.datas.get(source, pd.DataFrame())) for target, source in names.items()}
		self.static_config = self.static
		network = self.static["Network"]
		for column in ("material", "location", "sourcing"):
			if column not in network:
				raise ValueError(f"Global_Network 缺少 {column}")
		network["eff_from"] = network.get("eff_from", pd.Series(pd.Timestamp.min, index=network.index)).fillna(pd.Timestamp.min)
		network["eff_to"] = network.get("eff_to", pd.Series(pd.Timestamp.max, index=network.index)).fillna(pd.Timestamp.max)
		self.layer_map = self._layers(network)
		self.location_to_layer = self.layer_map
		self.layers = sorted(set(self.layer_map.values()), reverse=True)
		self._prepared = True

	# ------------------------------------------------------------------
	# 当日事实表。
	# ------------------------------------------------------------------
	def _daily_inputs(self, day: pd.Timestamp) -> dict[str, pd.DataFrame]:
		ctx = self.state_context
		if ctx is None:
			raise RuntimeError("ModuleFive.run() 需要 StateContext")
		date_text = day.strftime("%Y-%m-%d")
		# 生产账本的 GR/未来计划门控由 StateContext 统一维护，M5 只消费视图。
		production = self._normalise(ctx.get_deployment_production_view(date_text))
		sdl = ctx.get_deployment_supply_demand_view(date_text)
		sdl = self.static["SupplyDemandLog"] if sdl is None or sdl.empty else self._normalise(sdl)
		orders = ctx.get_deployment_order_log_view(date_text)
		delivery = self._normalise(ctx.get_delivery_gr_view(date_text))
		if "location" in delivery and "receiving" not in delivery:
			delivery = delivery.rename(columns={"location": "receiving"})
		return {
			**self.static,
			"SupplyDemandLog": self._normalise(sdl),
			"OrderLog": self._normalise(orders),
			"TodayShipment": self._normalise(ctx.get_shipment_log_view(date_text)),
			"Inventory": self._normalise(ctx.get_beginning_inventory_view(date_text)),
			"InTransit": self._normalise(ctx.get_planning_intransit_view(date_text)),
			"DeliveryGR": delivery,
			"OpenDeployment": self._normalise(ctx.get_open_deployment_view(date_text)),
			"Production": production,
			"ReceivingSpace": self._normalise(ctx.get_space_quota_view(date_text)),
		}

	def _active_network(self, config: dict[str, pd.DataFrame], day: pd.Timestamp) -> pd.DataFrame:
		network = config["Network"]
		active = network.loc[network["eff_from"].le(day) & network["eff_to"].ge(day)].copy()
		duplicates = active.duplicated(["material", "location"], keep=False)
		if duplicates.any():
			duplicated = active.loc[duplicates, ["material", "location", "sourcing"]]
			raise ValueError(f"Network 同日存在多个 sourcing: {duplicated.to_dict('records')[:5]}")
		return active.rename(columns={"sourcing": "upstream", "location": "node"})

	def _validate(self, config: dict[str, pd.DataFrame], active: pd.DataFrame) -> tuple[dict[str, int], list[dict[str, Any]]]:
		"""复现旧 M5 的验证输出契约，同时建立本实现使用的优先级映射。

		旧端验证的是完整 Network（而不只是当日有效边），并且会为根节点的
		空 sourcing 记录缺失 LeadTime。这些记录不影响规划，但属于持久化的
		``Validation`` 输出，必须保留以保证重构前后数据可比。
		"""
		log: list[dict[str, Any]] = []
		priority = config["DemandPriority"].copy()
		priority_map = {}
		if not priority.empty and {"demand_element", "priority"}.issubset(priority):
			priority["demand_element"] = priority["demand_element"].astype(str)
			priority["priority"] = pd.to_numeric(priority["priority"], errors="coerce").fillna(9).astype(int)

		network = config["Network"]
		if not network.empty:
			multiple = network.groupby(["material", "location"], dropna=False)["sourcing"].nunique()
			for (material, location), count in multiple[multiple.gt(1)].sort_index().items():
				log.append({"No": len(log) + 1, "Issue": f"Network配置不合法: material={material}, location={location} 有多个sourcing"})

		lead = config["LeadTime"]
		lead_paths = set(lead.loc[:, ["sending", "receiving"]].astype(str).itertuples(index=False, name=None)) if {"sending", "receiving"}.issubset(lead) else set()
		for route in network.loc[:, ["material", "sourcing", "location"]].itertuples(index=False):
			if (str(route.sourcing), str(route.location)) not in lead_paths:
				log.append({"No": len(log) + 1, "Issue": f"Missing leadtime for {route.sourcing}->{route.location} ({route.material})"})

		deploy = config["DeployConfig"]
		push_pull = config["PushPullModel"]
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
		priority_map = dict(zip(priority["demand_element"].astype(str), priority["priority"].astype(int)))
		return priority_map, log

	def _legacy_inventory_keys(self, config: dict[str, pd.DataFrame]) -> pd.DataFrame:
		"""构建旧 ``_initialize_soh_dict`` 写入 SOH 日志的零库存键。

		该键空间是当日 SDL、SafetyStock、OrderLog 的 material/location 笛卡尔积。
		它仅决定库存日志行，不应被误当作可用库存或改变分配结果。
		"""
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
		return pd.DataFrame(
			[(material, location) for material in sorted(materials) for location in sorted(locations)],
			columns=["material", "node"],
		)

	def _route_parameters(self, active: pd.DataFrame, config: dict[str, pd.DataFrame]) -> pd.DataFrame:
		"""构建 (material, sender, receiver) 的提前期、MOQ/RV 事实表。"""
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
		# 常规需求的路由提前期保留既有 root 层处理；Push 的旧逻辑则通过
		# get_sending_location_type() 优先采纳活动 Network 的显式类型，故另外
		# 计算 push_leadtime，在 _push() 中使用。
		location_type_text = routes["location_type"].fillna("").astype(str).str.strip()
		location_is_plant = location_type_text.str.casefold().eq("plant")
		roots = routes.apply(lambda row: self.layer_map.get((str(row.material), str(row.sending)), 99) == 0, axis=1)
		plant = location_is_plant | roots
		ptf_source = config["MaterialLocation"].copy()
		ptf_source["ptf"] = self._number(ptf_source, "ptf" if "ptf" in ptf_source else "PTF", 0).astype(int)
		ptf_source["lsk"] = self._number(ptf_source, "lsk" if "lsk" in ptf_source else "LSK", 1).astype(int)
		routes = routes.merge(ptf_source.loc[:, ["material", "location", "ptf", "lsk"]].rename(columns={"location": "sending"}), on=["material", "sending"], how="left")
		routes[["ptf", "lsk"]] = routes[["ptf", "lsk"]].fillna({"ptf": 0, "lsk": 1}).astype(int)
		routes["leadtime"] = (routes["PDT"] + routes["GR"]).clip(lower=1)
		routes.loc[plant, "leadtime"] = np.maximum(1, np.maximum(routes.loc[plant, "MCT"], routes.loc[plant, "PDT"] + routes.loc[plant, "GR"]) + routes.loc[plant, "ptf"] + routes.loc[plant, "lsk"] - 1)
		routes["push_leadtime"] = (routes["PDT"] + routes["GR"]).clip(lower=1)
		push_plant = location_is_plant | (location_type_text.eq("") & roots)
		routes.loc[push_plant, "push_leadtime"] = np.maximum(1, np.maximum(routes.loc[push_plant, "MCT"], routes.loc[push_plant, "PDT"] + routes.loc[push_plant, "GR"]) + routes.loc[push_plant, "ptf"] + routes.loc[push_plant, "lsk"] - 1)
		deploy = config["DeployConfig"].copy()
		deploy["moq"] = self._number(deploy, "moq", 1).clip(lower=0).astype(int)
		deploy["rv"] = self._number(deploy, "rv", 1).clip(lower=1).astype(int)
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

	def _horizon(self, nodes: pd.DataFrame, day: pd.Timestamp, routes: pd.DataFrame) -> pd.DataFrame:
		"""构建与旧 collector 相同的节点需求窗口。"""
		horizon = nodes.loc[:, ["material", "node", "upstream"]].copy()
		incoming = routes.rename(columns={"sending": "upstream", "receiving": "node"})
		horizon = horizon.merge(incoming.loc[:, ["material", "upstream", "node", "leadtime"]], on=["material", "upstream", "node"], how="left")
		root_horizon = routes.groupby(["material", "sending"], as_index=False)["leadtime"].max().rename(columns={"sending": "node", "leadtime": "root_leadtime"})
		horizon = horizon.merge(root_horizon, on=["material", "node"], how="left")
		horizon["horizon_days"] = horizon["leadtime"].fillna(horizon["root_leadtime"]).fillna(1).astype(int).clip(lower=1)
		horizon["horizon_end"] = day + pd.to_timedelta(horizon["horizon_days"], unit="D")
		return horizon.loc[:, ["material", "node", "horizon_end"]]

	def _direct_demand(self, nodes: pd.DataFrame, day: pd.Timestamp, config: dict[str, pd.DataFrame], routes: pd.DataFrame) -> pd.DataFrame:
		"""批量构建 SDL、订单与窗口末日安全库存需求。"""
		horizon = self._horizon(nodes, day, routes)

		def windowed(source: pd.DataFrame, element: str, element_column: Optional[str] = None) -> pd.DataFrame:
			if source.empty or not {"material", "location", "date"}.issubset(source):
				return self._empty(["material", "node", "receiving", "demand_element", "demand_qty", "requirement_date", "orig_location"])
			data = source.merge(horizon.loc[:, ["material", "node", "horizon_end"]].rename(columns={"node": "location"}), on=["material", "location"], how="inner")
			data = data.loc[data["date"].ge(day) & data["date"].le(data["horizon_end"])].copy()
			if data.empty:
				return self._empty(["material", "node", "receiving", "demand_element", "demand_qty", "requirement_date", "orig_location"])
			return pd.DataFrame({
				"material": data["material"], "node": data["location"], "receiving": data["location"],
				"demand_element": data[element_column].astype(str) if element_column else element,
				"demand_qty": self._quantity(data), "requirement_date": data["date"], "orig_location": data["location"],
			})

		sdl = windowed(config["SupplyDemandLog"], "forecast", "demand_element")
		orders = windowed(config["OrderLog"], "normal", "demand_type")
		safety = config["SafetyStock"]
		if safety.empty or not {"material", "location", "date"}.issubset(safety):
			ss = self._empty(list(sdl.columns))
		else:
			ss = safety.merge(horizon.loc[:, ["material", "node", "horizon_end"]].rename(columns={"node": "location"}), on=["material", "location"], how="inner")
			ss = ss.loc[ss["date"].eq(ss["horizon_end"])].copy()
			quantity = self._quantity(ss, ("safety_stock_qty", "quantity"))
			# 旧 collector 仅在汇总安全库存严格大于零时追加需求行。
			# 因此零安全库存不能形成 self-loop 的零数量部署记录。
			ss = pd.DataFrame({"material": ss["material"], "node": ss["location"], "receiving": ss["location"], "demand_element": "safety", "demand_qty": quantity, "requirement_date": ss["date"], "orig_location": ss["location"]})
			ss = ss.loc[ss["demand_qty"].gt(0)]
			ss = ss.groupby(["material", "node", "receiving", "demand_element", "requirement_date", "orig_location"], as_index=False, sort=False)["demand_qty"].sum()
		return pd.concat([sdl, orders, ss], ignore_index=True, sort=False)

	# ------------------------------------------------------------------
	# 层内分配和残余 gap 上推。
	# ------------------------------------------------------------------
	@staticmethod
	def _round_routes(demand: pd.DataFrame) -> pd.DataFrame:
		"""路径汇总 MOQ/RV，并以稳定最大余数法回分。"""
		result = demand.copy()
		result["planned_qty"] = result["demand_qty"].astype(np.int64)
		cross = result["node"].ne(result["receiving"])
		if not cross.any():
			return result
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
	def _allocate_priority(demand: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
		"""按优先级用现有库存覆盖需求，并在同优先级内比例向下取整。"""
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

	@staticmethod
	def _allocate_pipeline(demand: pd.DataFrame, pools: pd.DataFrame) -> pd.DataFrame:
		"""覆盖自需求的未来供给；保留旧入口的三池顺序和扣减口径。"""
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
			shares = pd.Series(
				np.minimum(np.floor(ratio.to_numpy()).astype(np.int64), remaining.to_numpy()),
				index=result.index,
				dtype=np.int64,
			).where(candidate, 0).astype(np.int64)
			result[output] = shares
			previous_used = shares.groupby([result["material"], result["node"]], sort=False).transform("sum")
		result["deploy_qty_with_plan_order"] = result[["deploy_from_in_transit", "deploy_from_open_deployment_inbound", "deploy_from_future_production"]].sum(axis=1)
		return result.drop(columns=pool_columns)

	def _next_gap(self, demand: pd.DataFrame, active: pd.DataFrame) -> pd.DataFrame:
		shortage = demand.loc[demand["residual_qty"].gt(0)].copy()
		if shortage.empty:
			return self._empty(["material", "node", "receiving", "demand_element", "demand_qty", "requirement_date", "orig_location"])
		parent = active.loc[:, ["material", "node", "upstream"]]
		shortage = shortage.merge(parent, on=["material", "node"], how="left")
		shortage = shortage.loc[shortage["upstream"].notna() & shortage["upstream"].ne("")].copy()
		if shortage.empty:
			return self._empty(["material", "node", "receiving", "demand_element", "demand_qty", "requirement_date", "orig_location"])
		shortage["receiving"] = shortage["node"]
		shortage["node"] = shortage["upstream"]
		shortage["demand_element"] = "net demand for " + shortage["demand_element"].astype(str)
		shortage["demand_qty"] = shortage["residual_qty"].astype(np.int64)
		keys = ["material", "node", "receiving", "demand_element", "requirement_date", "orig_location"]
		return shortage.groupby(keys, as_index=False, sort=False)["demand_qty"].sum()

	# ------------------------------------------------------------------
	# Push、接收配额和输出。
	# ------------------------------------------------------------------
	def _push(self, plan: pd.DataFrame, direct: pd.DataFrame, active: pd.DataFrame, routes: pd.DataFrame, config: dict[str, pd.DataFrame], stock: pd.DataFrame, projected: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
		columns = list(plan.columns)
		if plan.empty or config["PushPullModel"].empty:
			return self._empty(columns)
		regular = plan.loc[~plan["demand_element"].astype(str).str.contains("push", case=False, na=False)]
		pending = regular.assign(pending=regular["deployed_qty_invCon"].lt(regular["planned_qty"])).groupby(["material", "sending"], as_index=False)["pending"].any()
		candidates = regular.loc[:, ["material", "sending"]].drop_duplicates().merge(pending.loc[~pending["pending"], ["material", "sending"]], on=["material", "sending"])
		model = config["PushPullModel"].copy()
		model["model"] = model.get("model", "").astype(str).str.casefold().str.strip()
		candidates = candidates.merge(model.loc[model["model"].isin(["push", "soft push"]), ["material", "sending", "model"]], on=["material", "sending"], how="inner")
		if candidates.empty:
			return self._empty(columns)
		allocated = regular.groupby(["material", "sending"], as_index=False)["deployed_qty_invCon"].sum()
		inventory = stock.rename(columns={"node": "sending", "qty": "stock"})
		candidates = candidates.merge(inventory, on=["material", "sending"], how="left").merge(allocated, on=["material", "sending"], how="left")
		candidates[["stock", "deployed_qty_invCon"]] = candidates[["stock", "deployed_qty_invCon"]].fillna(0)
		ss_today = config["SafetyStock"].loc[config["SafetyStock"].get("date", pd.Series(pd.NaT, index=config["SafetyStock"].index)).eq(day)].copy()
		ss_today["sending_safety"] = self._quantity(ss_today, ("safety_stock_qty", "quantity"))
		ss_today = ss_today.groupby(["material", "location"], as_index=False)["sending_safety"].sum().rename(columns={"location": "sending"})
		candidates = candidates.merge(ss_today, on=["material", "sending"], how="left")
		candidates["available"] = (candidates["stock"] - candidates["deployed_qty_invCon"] - np.where(candidates["model"].eq("soft push"), candidates["sending_safety"].fillna(0), 0)).clip(lower=0)
		children = active.loc[:, ["material", "upstream", "node"]].rename(columns={"upstream": "sending", "node": "receiving"})
		work = candidates.merge(children, on=["material", "sending"], how="inner").merge(routes, on=["material", "sending", "receiving"], how="left")
		if work.empty:
			return self._empty(columns)
		work["leadtime"] = work.get("push_leadtime", work["leadtime"]).fillna(work["leadtime"]).fillna(1).astype(int)
		work["planned_delivery_date"] = day + pd.to_timedelta(work["leadtime"], unit="D")
		safety = config["SafetyStock"].copy()
		safety["receiving_safety"] = self._quantity(safety, ("safety_stock_qty", "quantity"))
		work = work.merge(safety.loc[:, ["material", "location", "date", "receiving_safety"]].rename(columns={"location": "receiving", "date": "planned_delivery_date"}), on=["material", "receiving", "planned_delivery_date"], how="left")
		projected_map = projected.rename(columns={"node": "receiving", "qty": "projected"})
		work = work.merge(projected_map, on=["material", "receiving"], how="left")
		commitment_source = direct.loc[
			direct["demand_element"].astype(str).str.casefold().isin(["ao", "normal", "forecast", "safety"]),
			["material", "node", "requirement_date", "demand_qty"],
		].rename(columns={"node": "receiving"})
		if not commitment_source.empty:
			commitment_source["demand_element"] = direct.loc[commitment_source.index, "demand_element"].astype(str).str.casefold()
		if commitment_source.empty:
			commitment = self._empty(["material", "receiving", "planned_delivery_date", "commitment"])
		else:
			commitment = work.loc[:, ["material", "receiving", "planned_delivery_date"]].drop_duplicates().merge(
				commitment_source, on=["material", "receiving"], how="left"
			)
			is_safety = commitment["demand_element"].eq("safety")
			commitment = commitment.loc[
				(
					~is_safety
					& commitment["requirement_date"].gt(day)
					& commitment["requirement_date"].le(commitment["planned_delivery_date"])
				)
				| (is_safety & commitment["requirement_date"].eq(commitment["planned_delivery_date"]))
			].groupby(["material", "receiving", "planned_delivery_date"], as_index=False, sort=False)["demand_qty"].sum().rename(columns={"demand_qty": "commitment"})
		work = work.merge(commitment, on=["material", "receiving", "planned_delivery_date"], how="left")
		work[["receiving_safety", "projected", "commitment"]] = work[["receiving_safety", "projected", "commitment"]].fillna(0)
		work["baseline"] = (work["projected"] - work["commitment"]).clip(lower=0)
		work["push_level"] = 1.2
		for _, index in work.groupby(["material", "sending"], sort=False).groups.items():
			group = work.loc[index]
			available_soh = float(group["available"].iloc[0])
			selected = 1.2
			for level in (1.2, 1.5, 2.0, 2.5, 3.0):
				need = (level * group["receiving_safety"] - group["baseline"]).clip(lower=0).sum()
				if need <= available_soh + 1e-9:
					selected = level
				else:
					break
			work.loc[index, "push_level"] = selected
		work["need"] = (work["push_level"] * work["receiving_safety"] - work["baseline"]).clip(lower=0)
		totals = work.groupby(["material", "sending"], as_index=False)["need"].sum().rename(columns={"need": "total_need"})
		work = work.merge(totals, on=["material", "sending"], how="left")
		# 旧 _allocate_push_quantities() 不以 need 为上限；即使可用库存
		# 大于总 need，也会继续按比例用尽可用库存（仅保留下取整余数）。
		work["push_qty"] = np.floor(work["available"] * work["need"] / work["total_need"].replace(0, np.nan)).fillna(0).astype(np.int64)
		work = work.loc[work["push_qty"].gt(0)].copy()
		if work.empty:
			return self._empty(columns)
		output = pd.DataFrame({
			"date": day, "material": work["material"], "sending": work["sending"], "receiving": work["receiving"], "demand_qty": 0,
			"demand_element": np.where(work["model"].eq("push"), "push replenishment", "soft push replenishment"),
			"planned_qty": work["push_qty"], "deployed_qty_invCon": work["push_qty"], "deploy_qty_with_plan_order": 0,
			"deploy_from_in_transit": 0, "deploy_from_open_deployment_inbound": 0, "deploy_from_future_production": 0,
			"planned_delivery_date": work["planned_delivery_date"], "orig_location": work["receiving"], "leadtime": work["leadtime"], "is_cross_node": True,
		})
		return output.reindex(columns=columns, fill_value=0)

	@staticmethod
	def _apply_space(plan: pd.DataFrame, space: pd.DataFrame, priority: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
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

	def run(self):
		"""编排 M5 当日生命周期；不承载任何业务计算。"""
		self.prepare()
		self._plan_day(pd.Timestamp(self.simulation_date).normalize())

	def _plan_day(self, day: pd.Timestamp):
		"""执行当日的供给账本、分层 gap 传播、Push 与输出构建。"""
		config = self._daily_inputs(day)
		active = self._active_network(config, day)
		priority, validation = self._validate(config, active)
		routes = self._route_parameters(active, config)

		inventory = config["Inventory"].copy()
		inventory["qty"] = self._quantity(inventory)
		stock = inventory.groupby(["material", "location"], as_index=False)["qty"].sum().rename(columns={"location": "node"})
		# 旧端会把动态需求涉及的 material/location 笛卡尔积初始化为零库存，
		# 并完整写出到 StockOnHandLog。补齐这些行只用于日志兼容性。
		legacy_keys = self._legacy_inventory_keys(config)
		stock = pd.concat([stock, legacy_keys.assign(qty=0)], ignore_index=True, sort=False)
		stock = stock.groupby(["material", "node"], as_index=False)["qty"].sum()
		delivery = config["DeliveryGR"].copy(); delivery["qty"] = self._quantity(delivery)
		delivery = delivery.groupby(["material", "receiving"], as_index=False)["qty"].sum().rename(columns={"receiving": "node"}) if not delivery.empty else self._empty(["material", "node", "qty"])
		production = config["Production"].copy(); production["qty"] = self._quantity(production, ("produced_qty", "planned_qty", "quantity"))
		today_production = production.loc[production.get("available_date", pd.Series(pd.NaT, index=production.index)).eq(day)].groupby(["material", "location"], as_index=False)["qty"].sum().rename(columns={"location": "node"}) if not production.empty else self._empty(["material", "node", "qty"])
		future_production = production.loc[production.get("available_date", pd.Series(pd.NaT, index=production.index)).gt(day)].groupby(["material", "location"], as_index=False)["qty"].sum().rename(columns={"location": "node", "qty": "future_production"}) if not production.empty else self._empty(["material", "node", "future_production"])
		transit = config["InTransit"].copy(); transit["qty"] = self._quantity(transit)
		transit_date = transit.get("actual_delivery_date", transit.get("available_date", pd.Series(pd.NaT, index=transit.index)))
		today_transit = transit.loc[transit_date.eq(day)].groupby(["material", "receiving"], as_index=False)["qty"].sum().rename(columns={"receiving": "node"}) if not transit.empty else self._empty(["material", "node", "qty"])
		future_transit = transit.loc[transit_date.gt(day)].groupby(["material", "receiving"], as_index=False)["qty"].sum().rename(columns={"receiving": "node", "qty": "future_intransit"}) if not transit.empty else self._empty(["material", "node", "future_intransit"])
		open_deployment = config["OpenDeployment"].copy(); open_deployment["qty"] = self._quantity(open_deployment, ("deployed_qty", "quantity"))
		outbound = open_deployment.loc[open_deployment.get("sending", pd.Series("", index=open_deployment.index)).ne(open_deployment.get("receiving", pd.Series("", index=open_deployment.index)))].groupby(["material", "sending"], as_index=False)["qty"].sum().rename(columns={"sending": "node"}) if not open_deployment.empty else self._empty(["material", "node", "qty"])
		inbound = open_deployment.loc[open_deployment.get("sending", pd.Series("", index=open_deployment.index)).ne(open_deployment.get("receiving", pd.Series("", index=open_deployment.index)))].groupby(["material", "receiving"], as_index=False)["qty"].sum().rename(columns={"receiving": "node", "qty": "open_inbound"}) if not open_deployment.empty else self._empty(["material", "node", "open_inbound"])
		shipment = config["TodayShipment"].copy(); shipment["qty"] = self._quantity(shipment)
		shipment = shipment.loc[shipment.get("date", pd.Series(day, index=shipment.index)).eq(day)].groupby(["material", "location"], as_index=False)["qty"].sum().rename(columns={"location": "node"}) if not shipment.empty else self._empty(["material", "node", "qty"])

		ledger = pd.concat([stock.assign(kind="stock"), delivery.assign(kind="delivery"), today_production.assign(kind="production"), outbound.assign(kind="outbound")], ignore_index=True, sort=False)
		available = ledger.pivot_table(index=["material", "node"], columns="kind", values="qty", aggfunc="sum", fill_value=0).reset_index()
		for column in ("stock", "delivery", "production", "outbound"):
			if column not in available: available[column] = 0
		available["qty"] = (available["stock"] + available["delivery"] + available["production"] - available["outbound"]).clip(lower=0).astype(np.int64)
		pools = available.loc[:, ["material", "node"]].merge(future_transit, on=["material", "node"], how="left").merge(inbound, on=["material", "node"], how="left").merge(future_production, on=["material", "node"], how="left").fillna(0)
		projected = pools.loc[:, ["material", "node", "future_production"]].merge(available.loc[:, ["material", "node", "qty"]], on=["material", "node"], how="left").merge(today_transit.rename(columns={"qty": "today_transit"}), on=["material", "node"], how="left").merge(shipment.rename(columns={"qty": "shipment"}), on=["material", "node"], how="left").fillna(0)
		projected["qty"] = projected["qty"] + projected["today_transit"] + projected["future_production"] - projected["shipment"]

		all_direct: list[pd.DataFrame] = []
		plan_parts: list[pd.DataFrame] = []
		unfulfilled_parts: list[pd.DataFrame] = []
		gap = self._empty(["material", "node", "receiving", "demand_element", "demand_qty", "requirement_date", "orig_location"])
		row_id = 0
		for layer in self.layers:
			base = pd.DataFrame([(material, node) for (material, node), value in self.layer_map.items() if value == layer], columns=["material", "node"])
			nodes = base.merge(active.loc[:, ["material", "node", "upstream"]], on=["material", "node"], how="left")
			direct = self._direct_demand(nodes, day, config, routes)
			all_direct.append(direct)
			layer_gap = gap.merge(nodes.loc[:, ["material", "node"]].drop_duplicates(), on=["material", "node"], how="inner")
			if not layer_gap.empty:
				# 旧 _collect_gap_demands 会按当前节点 horizon_end 过滤 gap。
				# 不可把下游需求无限制地继续向上游传播。
				layer_gap = layer_gap.merge(self._horizon(nodes, day, routes), on=["material", "node"], how="left")
				layer_gap = layer_gap.loc[
					layer_gap["requirement_date"].ge(day)
					& layer_gap["requirement_date"].le(layer_gap["horizon_end"])
				].drop(columns="horizon_end")
			demand = pd.concat([direct, layer_gap], ignore_index=True, sort=False)
			if demand.empty:
				gap = self._empty(gap.columns.tolist())
				continue
			demand = demand.merge(routes.rename(columns={"sending": "node"}), on=["material", "node", "receiving"], how="left")
			demand["moq"] = demand["moq"].fillna(1).astype(int); demand["rv"] = demand["rv"].fillna(1).astype(int)
			demand["priority"] = demand["demand_element"].map(priority).fillna(99).astype(int)
			demand["row_id"] = np.arange(row_id, row_id + len(demand)); row_id += len(demand)
			demand = self._round_routes(demand)
			demand = self._allocate_priority(demand, available.loc[:, ["material", "node", "qty"]])
			demand = self._allocate_pipeline(demand, pools)
			demand["residual_qty"] = (demand["planned_qty"] - demand["deployed_qty_invCon"] - demand["deploy_qty_with_plan_order"]).clip(lower=0).astype(np.int64)
			shortage = demand.loc[demand["residual_qty"].gt(0)]
			unfulfilled_parts.append(pd.DataFrame({"date": day, "sending": shortage["node"], "receiving": shortage["receiving"], "demand_qty": shortage["demand_qty"], "demand_element": shortage["demand_element"], "unfulfilled_qty": shortage["residual_qty"], "reason": "supply shortage"}))
			planned_delivery = demand["requirement_date"].where(demand["node"].ne(demand["receiving"]), day)
			plan_parts.append(pd.DataFrame({"date": day, "material": demand["material"], "sending": demand["node"], "receiving": demand["receiving"], "demand_qty": demand["demand_qty"], "demand_element": demand["demand_element"], "planned_qty": demand["planned_qty"], "deployed_qty_invCon": demand["deployed_qty_invCon"], "deploy_qty_with_plan_order": demand["deploy_qty_with_plan_order"], "deploy_from_in_transit": demand["deploy_from_in_transit"], "deploy_from_open_deployment_inbound": demand["deploy_from_open_deployment_inbound"], "deploy_from_future_production": demand["deploy_from_future_production"], "planned_delivery_date": planned_delivery, "orig_location": demand["orig_location"], "leadtime": np.where(demand["node"].eq(demand["receiving"]), 0, demand["leadtime"].fillna(1).astype(int)), "is_cross_node": demand["node"].ne(demand["receiving"])}))
			gap = self._next_gap(demand, active)

		plan = pd.concat(plan_parts, ignore_index=True, sort=False) if plan_parts else self._empty(["date", "material", "sending", "receiving", "demand_qty", "demand_element", "planned_qty", "deployed_qty_invCon"])
		direct_all = pd.concat(all_direct, ignore_index=True, sort=False) if all_direct else self._empty(["material", "node", "demand_element", "demand_qty"])
		push = self._push(plan, direct_all, active, routes, config, available.loc[:, ["material", "node", "qty"]], projected.loc[:, ["material", "node", "qty"]], day)
		plan = pd.concat([plan, push], ignore_index=True, sort=False)
		plan, unfulfilled_space = self._apply_space(plan, config["ReceivingSpace"], priority)
		unfulfilled = pd.concat(unfulfilled_parts + [unfulfilled_space], ignore_index=True, sort=False) if unfulfilled_parts or not unfulfilled_space.empty else pd.DataFrame()
		deployed = plan.loc[plan["sending"].ne(plan["receiving"])].groupby(["material", "sending"], as_index=False)["deployed_qty_invCon"].sum().rename(columns={"sending": "node", "deployed_qty_invCon": "deployed_qty"}) if not plan.empty else self._empty(["material", "node", "deployed_qty"])
		soh = available.loc[:, ["material", "node", "stock", "production", "delivery"]].merge(today_transit.rename(columns={"qty": "in_transit"}), on=["material", "node"], how="outer").merge(shipment.rename(columns={"qty": "today_shipment"}), on=["material", "node"], how="outer").merge(deployed, on=["material", "node"], how="outer").fillna(0)
		soh["ending_soh"] = soh["stock"] + soh["production"] + soh["delivery"] + soh["in_transit"] - soh["today_shipment"] - soh["deployed_qty"]
		soh = soh.rename(columns={"node": "location", "stock": "beginning_soh", "production": "production", "delivery": "delivery_gr"}); soh.insert(0, "date", day)
		self._result = {"deployment_plan": self._stable(plan, ["date", "material", "sending", "receiving", "planned_delivery_date", "demand_element"]), "unfulfilled_log": self._stable(unfulfilled, ["date", "sending", "receiving", "demand_element"]), "stock_on_hand_log": self._stable(soh, ["date", "material", "location"]), "validation_log": pd.DataFrame(validation), "statistics": {"deployment_count": len(plan), "unfulfilled_count": len(unfulfilled), "processed_dates": 1}}

	def _empty_result(self):
		self._result = {"deployment_plan": pd.DataFrame(), "unfulfilled_log": pd.DataFrame(), "stock_on_hand_log": pd.DataFrame(), "validation_log": pd.DataFrame(), "statistics": {"deployment_count": 0, "unfulfilled_count": 0, "processed_dates": 1}}


__all__ = ["ModuleFive"]
