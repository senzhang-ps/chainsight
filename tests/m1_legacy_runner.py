"""在 chainsight-main 代码路径下运行旧 M1，并将结果交给性能对比测试。"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from io import StringIO

import pandas as pd


MAIN_ROOT = Path(os.environ["CHAINSIGHT_MAIN_ROOT"]).resolve()
RESULT_PATH = Path(os.environ["M1_LEGACY_RESULT_PATH"]).resolve()
CONFIG_PATH = Path(os.environ["M1_LEGACY_CONFIG_PATH"]).resolve()
PROGRESS_PATH = Path(os.environ["M1_LEGACY_PROGRESS_PATH"]).resolve()
START_DATE = os.environ["M1_COMPARE_START_DATE"]
END_DATE = os.environ["M1_COMPARE_END_DATE"]

# 本脚本位于 Decoupling 的 tests 目录；强制优先导入主项目的 src / pgsql_db，
# 避免 Python 将两个工作区中同名的 src 包混用。
sys.path.insert(0, str(MAIN_ROOT))

from src.core.main_integration.seed import set_module_seeds
from src.core.orchestrator import create_orchestrator
from src.modules import module1
from src.utils.normalization import normalize_identifiers


def _records(frame: pd.DataFrame | None) -> list[dict]:
    if frame is None or frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso", default_handler=str))


def _progress(message: str) -> None:
    with PROGRESS_PATH.open("a", encoding="utf-8") as progress_file:
        progress_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [M1 compare] {message}\n")


def main() -> None:
    _progress("legacy main: 开始读取数据库配置快照")
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = {
        name: pd.read_json(StringIO(frame_json), orient="split", dtype=False)
        for name, frame_json in payload.items()
    }
    _progress("legacy main: 配置快照解析完成，开始设置随机种子")
    set_module_seeds(config)
    _progress("legacy main: 随机种子设置完成，开始初始化编排器")
    orch = create_orchestrator(
        start_date=START_DATE,
        output_dir=str(RESULT_PATH.parent / "legacy_scratch"),
        persist_to_disk=False,
    )
    inventory = config.get("M1_InitialInventory", pd.DataFrame())
    orch.initialize_inventory(inventory)
    _progress("legacy main: 编排器和初始库存初始化完成")

    days: list[dict] = []
    elapsed: list[float] = []
    previous_orders: pd.DataFrame | None = None
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        day_key = day.strftime("%Y-%m-%d")
        _progress(f"legacy main 计时: {day_key} 开始")
        orch.save_beginning_inventory(day_key)
        tick = time.perf_counter()
        result = module1.run_daily_order_generation(
            config_dict=config,
            simulation_date=day,
            output_dir=str(RESULT_PATH.parent / "legacy_scratch"),
            orchestrator=orch,
            previous_orders_df=previous_orders,
            skip_file_output=True,
        )
        seconds = time.perf_counter() - tick
        shipments = result.get("shipment_df", pd.DataFrame())
        if not shipments.empty:
            orch.process_module1_shipments(normalize_identifiers(shipments), day_key)
        previous_orders = result.get("all_orders_for_next_day")
        elapsed.append(seconds)
        days.append({
            "date": day_key,
            "elapsed_seconds": seconds,
            "m1_result": {
                name: _records(result.get(name))
                for name in (
                    "orders_df",
                    "shipment_df",
                    "cut_df",
                    "supply_demand_df",
                    "summary_df",
                )
            },
        })
        _progress(f"legacy main 计时: {day_key} 完成，{seconds:.2f}s")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(
            {"days": days, "seconds": elapsed, "total_seconds": sum(elapsed)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
