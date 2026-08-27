"""目标历史输入下 M5 订单需求收集的单日 trace。"""

# 测试文件说明
# 测试目的：集中验证部署计划、优先级分配与供给池扣减的一致性。
# 测试方法：按 `diagnostics` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保部署计划、优先级分配与供给池扣减的一致性变更时能够快速定位回归影响。



from __future__ import annotations

import json

import pandas as pd

from src.core.orchestrator import Orch
from src.modules.deployment_planning.integration_refactor import ModuleFive
from src.modules.deployment_planning.cache_utils import (
    assign_location_layers,
    build_active_network_cache,
    build_lead_time_cache,
    build_order_log_index,
    build_safety_stock_index,
    build_sdl_index,
)
from src.modules.deployment_planning.data_loader import load_integrated_config
from src.modules.deployment_planning.demand_collector import (
    collect_node_demands,
    collect_node_demands_fast,
)
from src.modules.deployment_planning.horizon_batch_calculator import build_horizon_cache
from src.modules.state_context import StateContext
from src.utils.ptf_lsk import build_ptf_lsk_cache
from tests.integration import test_m5_controlled_db_inputs as controlled
from tests.regression import test_m5_two_way_compare as m5

DAY = pd.Timestamp(controlled.START_DATE)
REPORT_DIR = controlled.REPORT_DIR
ORDER_SIGNATURE = ["material", "location", "date", "demand_type", "quantity"]
DIRECT_SIGNATURE = ["material", "location", "date", "demand_type", "quantity"]


def _normalise_orders(frame: pd.DataFrame) -> pd.DataFrame:
    """将订单统一成可保留多重集的 M5 比较边界。"""
    result = frame.copy()
    for column in ("material", "location", "demand_type"):
        if column not in result:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str).str.strip()
    result["date"] = pd.to_datetime(result.get("date"), errors="coerce").dt.normalize()
    quantity = result["quantity"] if "quantity" in result else result.get(
        "demand_qty", pd.Series(0, index=result.index)
    )
    result["quantity"] = pd.to_numeric(quantity, errors="coerce").fillna(0).astype(int)
    return result.loc[:, ORDER_SIGNATURE]


def _normalise_direct(frame: pd.DataFrame, location_column: str) -> pd.DataFrame:
    """将 direct-demand 统一到保留重复次数的比较边界。"""
    if frame.empty:
        return pd.DataFrame(columns=DIRECT_SIGNATURE)
    result = frame.copy()
    result["location"] = result[location_column].fillna("").astype(str).str.strip()
    result["date"] = pd.to_datetime(result["requirement_date"], errors="coerce").dt.normalize()
    result["demand_type"] = result["demand_element"].fillna("").astype(str).str.strip()
    result["quantity"] = pd.to_numeric(result["demand_qty"], errors="coerce").fillna(0).astype(int)
    return result.loc[:, DIRECT_SIGNATURE]


def _source_forecast_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """抽取目标物料地点的 SDL 事实，不修改或去重源行。"""
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["material"] = result["material"].astype(str)
    result["location"] = result["location"].astype(str)
    return result.loc[
        result["material"].eq("21133312") & result["location"].eq("C810"),
        [column for column in ("material", "location", "date", "demand_element", "quantity") if column in result],
    ].copy()


def _multiset_difference(left: pd.DataFrame, right: pd.DataFrame, label: str) -> pd.DataFrame:
    """返回 ``left`` 中按订单业务签名比 ``right`` 多出的行及出现次数。"""
    left_counts = left.groupby(ORDER_SIGNATURE, dropna=False, as_index=False).size().rename(columns={"size": "left_count"})
    right_counts = right.groupby(ORDER_SIGNATURE, dropna=False, as_index=False).size().rename(columns={"size": "right_count"})
    result = left_counts.merge(right_counts, on=ORDER_SIGNATURE, how="left")
    result["right_count"] = result["right_count"].fillna(0).astype(int)
    result["count_difference"] = result["left_count"] - result["right_count"]
    result = result.loc[result["count_difference"].gt(0)].copy()
    result.insert(0, "difference", label)
    return result.sort_values(["material", "location", "date", "demand_type", "quantity"], kind="mergesort")


def _legacy_direct_demands(
    config: dict, day: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """复现 legacy 需求收集及其 horizon-cache 失败时的生产回退路径。"""
    network_layers = assign_location_layers(config["Network"])
    pairs = {
        (str(row.material), str(row.location))
        for row in network_layers.itertuples(index=False)
    }
    layer_map = {
        (str(row.material), str(row.location)): int(row.layer)
        for row in network_layers.itertuples(index=False)
    }
    ptf_lsk_cache = build_ptf_lsk_cache(config.get("M4_MaterialLocationLineCfg", pd.DataFrame()))
    lead_time_cache = build_lead_time_cache(config["LeadTime"])
    active_network_cache = build_active_network_cache(config["Network"])
    sdl_index = build_sdl_index(config["SupplyDemandLog"])
    ss_index = build_safety_stock_index(config["SafetyStock"])
    order_index = build_order_log_index(config.get("OrderLog", pd.DataFrame()))
    rows, horizon_rows, paths = [], [], []
    for layer in sorted(set(layer_map.values()), reverse=True):
        layer_pairs = {pair for pair, value in layer_map.items() if value == layer}
        try:
            horizon_cache = build_horizon_cache(
                all_pairs=layer_pairs,
                sim_date=day,
                network_df=config["Network"],
                leadtime_df=config["LeadTime"],
                m4_mlcfg_df=config.get("M4_MaterialLocationLineCfg", pd.DataFrame()),
                ptf_lsk_cache=ptf_lsk_cache,
                lead_time_cache=lead_time_cache,
                active_network_cache=active_network_cache,
                location_layer_map=layer_map,
            )
            paths.append(f"layer={layer}:horizon_cache_fast")
        except Exception as exc:
            horizon_cache = None
            paths.append(f"layer={layer}:fallback:{type(exc).__name__}")
        if horizon_cache is not None:
            horizon_rows.extend({
                "material": material,
                "location": location,
                "legacy_upstream": value["upstream"],
                "legacy_horizon_end": value["horizon_end"],
            } for (material, location), value in horizon_cache.items())
        for material, location in sorted(layer_pairs):
            try:
                if horizon_cache is not None:
                    node_rows = collect_node_demands_fast(
                        material, location, day, config, {}, horizon_cache,
                        sdl_index=sdl_index, ss_index=ss_index, order_index=order_index,
                    )
                else:
                    node_rows = collect_node_demands(
                        material, location, day, config, {},
                        ptf_lsk_cache=ptf_lsk_cache,
                        lead_time_cache=lead_time_cache,
                        active_network_cache=active_network_cache,
                        sdl_index=sdl_index, ss_index=ss_index, order_index=order_index,
                    )
            except Exception as exc:
                # legacy `_process_layer_demands()` 对 future 异常同样将该节点置空。
                paths.append(f"layer={layer}:{material}/{location}:dropped:{type(exc).__name__}")
                node_rows = []
            rows.extend(node_rows)
    horizons = pd.DataFrame(
        horizon_rows,
        columns=["material", "location", "legacy_upstream", "legacy_horizon_end"],
    )
    dropped = [entry for entry in paths if ":dropped:" in entry]
    layer_paths = [entry for entry in paths if ":dropped:" not in entry]
    collection_path = "; ".join(layer_paths)
    if dropped:
        collection_path += f"; dropped_nodes={len(dropped)} ({dropped[0].rsplit(':', 1)[-1]})"
    return pd.DataFrame(rows), horizons, collection_path


def _legacy_order_demands(config: dict, day: pd.Timestamp) -> pd.DataFrame:
    """从 legacy 原始需求收集结果中保留订单行。"""
    result, _, _ = _legacy_direct_demands(config, day)
    return _orders_from_legacy_direct(result, config)


def _orders_from_legacy_direct(result: pd.DataFrame, config: dict) -> pd.DataFrame:
    """将已收集的 legacy 原始需求中的订单事实标准化。"""
    if result.empty:
        return pd.DataFrame(columns=ORDER_SIGNATURE)
    order_types = set(config.get("OrderLog", pd.DataFrame()).get("demand_type", pd.Series(dtype=str)).astype(str))
    result = result.loc[result["demand_element"].astype(str).isin(order_types)].copy()
    result = result.rename(columns={"requirement_date": "date", "demand_element": "demand_type"})
    return _normalise_orders(result)


def test_m5_order_collection_trace_for_target_day() -> None:
    # 测试目的：验证“m5、order、collection、trace、for、target、day”场景下部署计划、优先级分配与供给池扣减的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `exists()`，再通过 4 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止部署计划、优先级分配与供给池扣减的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    """定位订单在 StateContext、legacy 与 refactor direct-demand 三个边界的首次分歧。"""
    assert controlled.CONFIG_PATH.exists()
    orch = Orch(
        start_date=DAY.strftime("%Y-%m-%d"), end_date=DAY.strftime("%Y-%m-%d"),
        config_path=str(controlled.CONFIG_PATH), output_path=str(REPORT_DIR / "order_trace_config"),
        engine="pandas", skip_dq=True, enable_persistence=False,
    )
    context = StateContext(simulation_date=DAY, orch=orch)
    context.initialize(orch.all_config)

    db = controlled._db(); db.connect()
    try:
        history = {
            DAY.strftime("%Y-%m-%d"): {
                "all_orders_for_next_day": controlled._load(db, "module1_output_orderlog", DAY, active_orders=True),
                "shipment_df": controlled._load(db, "module1_output_shipmentlog", DAY),
                "supply_demand_df": controlled._load(db, "module1_output_supplydemandlog", DAY),
                "production_df": controlled._load(db, "module4_output_productionplan", DAY),
            }
        }
    finally:
        db.close()

    m1, m4 = m5._replay_inputs(context, DAY, history)
    legacy_config = load_integrated_config(
        orch.all_config, "", "", context, DAY, module1_result=m1, module4_result=m4,
    )
    module = ModuleFive(
        simulation_date=DAY, simulation_start_date=DAY, state_context=context, orch=orch,
    )
    module.prepare()
    inputs = module.load_daily_inputs(DAY)
    active = module.build_active_network(inputs, DAY)
    routes = module.build_route_parameters(active, inputs)
    refactor_direct = module.build_direct_demand(active, DAY, inputs, routes)

    raw_orders = _normalise_orders(history[DAY.strftime("%Y-%m-%d")]["all_orders_for_next_day"])
    context_orders = _normalise_orders(context.get_deployment_order_log_view(DAY.strftime("%Y-%m-%d")))
    legacy_direct, legacy_horizons, legacy_collection_path = _legacy_direct_demands(legacy_config, DAY)
    legacy_orders = _orders_from_legacy_direct(legacy_direct, legacy_config)
    refactor_orders = refactor_direct.loc[
        refactor_direct["demand_element"].astype(str).isin(set(raw_orders["demand_type"])),
        ["material", "node", "requirement_date", "demand_element", "demand_qty"],
    ].rename(columns={"node": "location", "requirement_date": "date", "demand_element": "demand_type", "demand_qty": "quantity"})
    refactor_orders = _normalise_orders(refactor_orders)

    legacy_direct_by_element = (
        legacy_direct.groupby("demand_element", dropna=False).size().to_dict()
        if not legacy_direct.empty else {}
    )
    refactor_direct_by_element = (
        refactor_direct.groupby("demand_element", dropna=False).size().to_dict()
        if not refactor_direct.empty else {}
    )
    legacy_direct_normalised = _normalise_direct(legacy_direct, "location")
    refactor_direct_normalised = _normalise_direct(refactor_direct, "node")
    legacy_only_direct = _multiset_difference(
        legacy_direct_normalised, refactor_direct_normalised, "legacy_only_direct"
    )
    refactor_only_direct = _multiset_difference(
        refactor_direct_normalised, legacy_direct_normalised, "refactor_only_direct"
    )
    legacy_only_forecast = legacy_only_direct.loc[
        legacy_only_direct["demand_type"].eq("forecast")
    ]

    differences = pd.concat([
        _multiset_difference(raw_orders, context_orders, "raw_orders_only"),
        _multiset_difference(context_orders, legacy_orders, "context_orders_not_collected_by_legacy"),
        _multiset_difference(context_orders, refactor_orders, "context_orders_not_collected_by_refactor"),
        _multiset_difference(refactor_orders, legacy_orders, "refactor_only_collected_orders"),
        _multiset_difference(legacy_orders, refactor_orders, "legacy_only_collected_orders"),
    ], ignore_index=True)
    horizons = module.build_node_horizon(active, DAY, routes).rename(columns={"node": "location"})
    network = active.loc[:, ["material", "node", "upstream"]].rename(columns={"node": "location"})
    c810_horizons = horizons.merge(
        legacy_horizons, on=["material", "location"], how="outer"
    ).loc[lambda frame: frame["material"].astype(str).eq("21133312") & frame["location"].astype(str).eq("C810")]
    refactor_only = differences.loc[
        differences["difference"].eq("refactor_only_collected_orders")
    ].merge(network, on=["material", "location"], how="left").merge(
        horizons, on=["material", "location"], how="left"
    ).merge(legacy_horizons, on=["material", "location"], how="left")
    refactor_only_by_location = refactor_only.groupby(
        ["location", "upstream", "legacy_upstream", "legacy_horizon_end", "horizon_end"],
        dropna=False, as_index=False,
    )["count_difference"].sum().sort_values("count_difference", ascending=False)
    summary = {
        "run_id": controlled.RUN_ID,
        "date": DAY.strftime("%Y-%m-%d"),
        "raw_order_rows": len(raw_orders),
        "state_context_order_rows": len(context_orders),
        "legacy_collection_path": legacy_collection_path,
        "legacy_direct_demand_rows": len(legacy_direct),
        "refactor_direct_demand_rows": len(refactor_direct),
        "legacy_direct_rows_by_demand_element": legacy_direct_by_element,
        "refactor_direct_rows_by_demand_element": refactor_direct_by_element,
        "legacy_only_forecast_direct": legacy_only_forecast.to_dict("records"),
        "target_c810_horizons": c810_horizons.to_dict("records"),
        "legacy_collected_order_rows": len(legacy_orders),
        "refactor_direct_collected_order_rows": len(refactor_orders),
        "difference_rows_by_type": differences.groupby("difference").agg(
            business_signatures=("difference", "size"),
            order_occurrences=("count_difference", "sum"),
        ).to_dict("index") if not differences.empty else {},
        "refactor_only_order_window_by_location": refactor_only_by_location.to_dict("records"),
        "interpretation": "先记录原始行数；随后仅以多重集计数定位订单差异。统计不会去重、合并或改写任何 M5 结果行；不含 MOQ、库存和 pipeline 分配。",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "m5_order_collection_trace_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    differences.to_csv(REPORT_DIR / "m5_order_collection_trace_differences.csv", index=False, encoding="utf-8-sig")
    refactor_only.to_csv(REPORT_DIR / "m5_order_collection_trace_refactor_only_orders.csv", index=False, encoding="utf-8-sig")
    legacy_only_direct.to_csv(REPORT_DIR / "m5_order_collection_trace_legacy_only_direct.csv", index=False, encoding="utf-8-sig")
    refactor_only_direct.to_csv(REPORT_DIR / "m5_order_collection_trace_refactor_only_direct.csv", index=False, encoding="utf-8-sig")
    _source_forecast_rows(legacy_config["SupplyDemandLog"]).to_csv(
        REPORT_DIR / "m5_order_collection_trace_legacy_c810_sdl.csv", index=False, encoding="utf-8-sig"
    )
    _source_forecast_rows(inputs["SupplyDemandLog"]).to_csv(
        REPORT_DIR / "m5_order_collection_trace_refactor_c810_sdl.csv", index=False, encoding="utf-8-sig"
    )
    c810_horizons.to_csv(
        REPORT_DIR / "m5_order_collection_trace_c810_horizons.csv", index=False, encoding="utf-8-sig"
    )

    # 该用例是诊断而非 parity gate；必须保证三个边界均实际读取并生成报告。
    assert len(raw_orders) == len(context_orders)
    assert len(legacy_orders) > 0
    assert len(refactor_orders) > 0
