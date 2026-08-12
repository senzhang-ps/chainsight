"""独立运行 Decoupling polars M1，并将性能追加到既有 M1 对比报告。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from test_m1_two_way_compare import (
    LEGACY_RESULT_PATH,
    REPORT_DIR,
    _db,
    _load_prepared_config,
    _performance_summary,
    _progress,
    _run_refactor,
    _write_report,
)


def main() -> None:
    if not LEGACY_RESULT_PATH.exists():
        raise RuntimeError("缺少旧 M1 运行结果；请先执行 test_m1_two_way_compare.py")

    report_path = REPORT_DIR / "m1_two_way_compare.json"
    if not report_path.exists():
        raise RuntimeError("缺少 M1 对比报告 JSON；请先执行 test_m1_two_way_compare.py")

    db = _db()
    try:
        _progress("开始从数据库加载 polars M1 所需的真实配置数据")
        db.connect()
        config = _load_prepared_config(db)
    finally:
        db.close()
    _progress("数据库配置加载完成，开始 Decoupling polars M1 唯一一次真实对比运行并计时")
    polars = _run_refactor(config, engine="polars")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["performance"]["refactor_polars"] = _performance_summary(polars)
    legacy_mean = report["performance"]["legacy_main"]["mean_seconds"]
    for summary in report["performance"].values():
        summary["relative_to_legacy"] = summary["mean_seconds"] / legacy_mean
    report["standalone_polars_run"] = {
        "days": len(polars["days"]),
        "total_seconds": polars["total_seconds"],
    }
    markdown_path = _write_report(report)
    _progress(f"polars 结果已写入报告: {markdown_path}")


if __name__ == "__main__":
    main()
