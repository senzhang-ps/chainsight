#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_bc_10day_reports.py

Parse three simulation logs (Dev, Src, DB) and the 3-way content comparison
JSON to generate markdown reports for each 10-day interval (BC_S5 scenario).

The core comparison dimension is **per table (Sheet)** within each module,
showing daily row counts and content match status for Dev/Src/DB.

Output goes to  docs/  with filenames like:
    BC算法优化测试报告_Day01-10_20251005-20251014.md

Usage:
    python tools/generate_bc_10day_reports.py
"""

from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path(r"D:\PG\test\chainsight")

LOG_PATHS: dict[str, Path] = {
    "Dev": BASE_DIR / "ChainSight_Dev" / "BC_S5" / "run_20260211_181635" / "simulation_log_20260211_181635.txt",
    "Src": BASE_DIR / "outputs" / "BC_S5" / "run_20260211_145215" / "simulation_log_20260211_145215.txt",
    "DB":  BASE_DIR / "outputs" / "db_BC_S5_20260225_132106" / "simulation_log_20260225_132106.txt",
}

JSON_PATH = BASE_DIR / "tools" / "bc_3way_content_results_full.json"

OUTPUT_DIR = BASE_DIR / "docs"

SIM_START_DATE = datetime(2025, 10, 5)
SIM_TOTAL_DAYS = 87

# 10-day intervals  (day_start, day_end) — 1-indexed simulation days
INTERVALS: list[tuple[int, int]] = [
    (1, 10),
    (11, 20),
    (21, 30),
    (31, 40),
    (41, 50),
    (51, 60),
    (61, 70),
    (71, 80),
    (81, 87),
]

OUTPUT_PATHS: dict[str, str] = {
    "Dev": "ChainSight_Dev/BC_S5/run_20260211_181635/",
    "Src": "outputs/BC_S5/run_20260211_145215/",
    "DB":  "outputs/db_BC_S5_20260225_132106/",
}

# Ordered list of all tables, grouped by module
# (module_label, module_cn_name, [(table_key, table_cn_name), ...])
MODULE_TABLES: list[tuple[str, str, list[tuple[str, str]]]] = [
    ("Module1", "订单生成模块", [
        ("module1/OrderLog", "订单日志 (OrderLog)"),
        ("module1/ShipmentLog", "发货日志 (ShipmentLog)"),
        ("module1/CutLog", "削减日志 (CutLog)"),
        ("module1/SupplyDemandLog", "供需日志 (SupplyDemandLog)"),
        ("module1/Summary", "每日汇总 (Summary)"),
    ]),
    ("Module3", "净需求计算模块", [
        ("module3/NetDemand", "净需求 (NetDemand)"),
    ]),
    ("Module4", "生产计划模块", [
        ("module4/ProductionPlan", "生产计划 (ProductionPlan)"),
        ("module4/CapacityExceed", "超容量报告 (CapacityExceed)"),
        ("module4/ChangeoverLog", "换型日志 (ChangeoverLog)"),
    ]),
    ("Module5", "部署规划模块", [
        ("module5/DeploymentPlan", "部署计划 (DeploymentPlan)"),
        ("module5/UnfulfilledLog", "未满足日志 (UnfulfilledLog)"),
        ("module5/StockOnHandLog", "库存日志 (StockOnHandLog)"),
        ("module5/Validation", "验证 (Validation)"),
    ]),
    ("Module6", "物流执行模块", [
        ("module6/DeliveryPlan", "交付计划 (DeliveryPlan)"),
        ("module6/VehicleLog", "车辆日志 (VehicleLog)"),
        ("module6/TruckUsageLog", "卡车使用日志 (TruckUsageLog)"),
    ]),
]

# (KNOWN_DISCREPANCY_TABLES removed — bug fixed in main_integration.py, all versions now consistent)

# ──────────────────────────────────────────────────────────────────────
# Regex helpers
# ──────────────────────────────────────────────────────────────────────

RE_TIMESTAMP = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\[(\w+)\]\s*(.*)"
)

RE_DAY_HEADER = re.compile(
    r"={5,}\s*第\s+(\d+)/(\d+)\s+天:\s+(\d{4}-\d{2}-\d{2})\s*={5,}"
)

RE_DAY_COMPLETE = re.compile(r"第\s+(\d+)\s+天处理完成")

RE_MODULE_START = {
    "M1": re.compile(r"运行\s*Module\s*1"),
    "M3": re.compile(r"运行\s*Module\s*3"),
    "M4": re.compile(r"运行\s*Module\s*4"),
    "M5": re.compile(r"运行\s*Module\s*5"),
    "M6": re.compile(r"运行\s*Module\s*6"),
}

RE_MODULE_END = {
    "M1": re.compile(r"Module\s*1\s*完成"),
    "M3": re.compile(r"Module\s*3\s*完成"),
    "M4": re.compile(r"Module\s*4\s*完成"),
    "M5": re.compile(r"Module\s*5\s*完成"),
    "M6": re.compile(r"Module\s*6\s*完成"),
}

RE_M1_ORDER_TIME = re.compile(
    r"\[M1\]\s*当日订单生成完成.*耗时:\s*([\d.]+)s"
)
RE_M1_SDL_TIME = re.compile(
    r"\[M1\]\s*供需日志生成完成.*耗时:\s*([\d.]+)s"
)
RE_M3_TOTAL = re.compile(
    r"\[M3\]\s*total(?:\s*duration)?:\s*([\d.]+)s"
)
RE_M5_FULL_DAY = re.compile(
    r"\[M5\]\s*Full day total\s*用时:\s*([\d.]+)s"
)

RE_DAILY_STATS = re.compile(r"当日统计:\s*(\{.*\})")

# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────

class DayData:
    """Parsed data for a single simulation day."""

    __slots__ = (
        "sim_day", "date_str", "day_start_ts", "day_end_ts",
        "module_times", "m1_order_time", "m1_sdl_time",
        "m3_total", "m5_full_day", "stats",
    )

    def __init__(self, sim_day: int, date_str: str):
        self.sim_day: int = sim_day
        self.date_str: str = date_str
        self.day_start_ts: datetime | None = None
        self.day_end_ts: datetime | None = None
        self.module_times: dict[str, float] = {}
        self.m1_order_time: float | None = None
        self.m1_sdl_time: float | None = None
        self.m3_total: float | None = None
        self.m5_full_day: float | None = None
        self.stats: dict[str, Any] = {}

    @property
    def day_total_seconds(self) -> float:
        if self.day_start_ts and self.day_end_ts:
            return (self.day_end_ts - self.day_start_ts).total_seconds()
        return 0.0

    def module_sec(self, mod: str) -> float:
        if mod in self.module_times:
            return self.module_times[mod]
        if mod == "M3" and self.m3_total is not None:
            return self.m3_total
        if mod == "M5" and self.m5_full_day is not None:
            return self.m5_full_day
        return 0.0


# ──────────────────────────────────────────────────────────────────────
# Log Parsing
# ──────────────────────────────────────────────────────────────────────

def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def parse_log(path: Path) -> dict[int, DayData]:
    """Return {sim_day: DayData} from a single log file."""

    days: dict[int, DayData] = {}
    current_day: int | None = None
    module_starts: dict[str, datetime] = {}

    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n\r")
            m_ts = RE_TIMESTAMP.match(line)
            if not m_ts:
                continue

            ts_str, _level, body = m_ts.group(1), m_ts.group(2), m_ts.group(3)
            ts = _parse_ts(ts_str)

            # Day header
            m_dh = RE_DAY_HEADER.search(body)
            if m_dh:
                sim_day = int(m_dh.group(1))
                date_str = m_dh.group(3)
                dd = DayData(sim_day, date_str)
                dd.day_start_ts = ts
                days[sim_day] = dd
                current_day = sim_day
                module_starts.clear()
                continue

            if current_day is None:
                continue

            dd = days[current_day]

            # Day complete
            m_dc = RE_DAY_COMPLETE.search(body)
            if m_dc:
                completed_day = int(m_dc.group(1))
                if completed_day in days:
                    days[completed_day].day_end_ts = ts
                continue

            # Module start/end timestamps
            for mod, pat in RE_MODULE_START.items():
                if pat.search(body):
                    module_starts[mod] = ts

            for mod, pat in RE_MODULE_END.items():
                if pat.search(body) and mod in module_starts:
                    elapsed = (ts - module_starts[mod]).total_seconds()
                    dd.module_times[mod] = elapsed

            # Embedded timing
            m = RE_M1_ORDER_TIME.search(body)
            if m:
                dd.m1_order_time = float(m.group(1))

            m = RE_M1_SDL_TIME.search(body)
            if m:
                dd.m1_sdl_time = float(m.group(1))

            m = RE_M3_TOTAL.search(body)
            if m:
                dd.m3_total = float(m.group(1))

            m = RE_M5_FULL_DAY.search(body)
            if m:
                dd.m5_full_day = float(m.group(1))

            # Daily stats
            m = RE_DAILY_STATS.search(body)
            if m:
                try:
                    dd.stats = ast.literal_eval(m.group(1))
                except Exception:
                    pass

    return days


# ──────────────────────────────────────────────────────────────────────
# JSON comparison data loading
# ──────────────────────────────────────────────────────────────────────

def load_comparison_json(path: Path) -> dict[str, Any]:
    """Load the 3-way content comparison results JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _fmt_sec(sec: float) -> str:
    return f"{sec:.0f}秒 ({sec / 60:.1f}分钟)"


def _fmt_sec_min(sec: float) -> str:
    return f"{sec / 60:.1f}分钟"


def _speedup(base: float, opt: float) -> str:
    if opt == 0:
        return "N/A"
    return f"{base / opt:.2f}x"


def _pct_reduction(base: float, opt: float) -> str:
    if base == 0:
        return "N/A"
    return f"{(base - opt) / base * 100:.1f}%"


def _safe_avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sim_date(day_num: int) -> str:
    return (SIM_START_DATE + timedelta(days=day_num - 1)).strftime("%Y-%m-%d")


def _interval_dates(day_start: int, day_end: int) -> tuple[str, str]:
    return _sim_date(day_start), _sim_date(day_end)


# ──────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────

def generate_interval_report(
    day_start: int,
    day_end: int,
    all_data: dict[str, dict[int, DayData]],
    comparison_data: dict[str, Any],
) -> str:
    """Build the full markdown report for one 10-day interval."""

    date_start, date_end = _interval_dates(day_start, day_end)
    num_days = day_end - day_start + 1
    comparisons = comparison_data["comparisons"]

    # ── Collect per-version interval performance data ──
    interval_total: dict[str, float] = {}
    interval_avg: dict[str, float] = {}
    module_avg: dict[str, dict[str, float]] = {}
    last_day_modules: dict[str, dict[str, float]] = {}

    for ver in ("Dev", "Src", "DB"):
        ver_days = all_data[ver]
        total = 0.0
        mod_totals: dict[str, list[float]] = {
            "M1": [], "M3": [], "M4": [], "M5": [], "M6": []
        }

        for d in range(day_start, day_end + 1):
            dd = ver_days.get(d)
            if dd is None:
                continue
            total += dd.day_total_seconds
            for mod in mod_totals:
                sec = dd.module_sec(mod)
                if sec > 0:
                    mod_totals[mod].append(sec)

        interval_total[ver] = total
        interval_avg[ver] = total / num_days if num_days else 0

        for mod in ("M1", "M3", "M4", "M5", "M6"):
            module_avg.setdefault(mod, {})[ver] = _safe_avg(mod_totals[mod])

        last_dd = ver_days.get(day_end)
        for mod in ("M1", "M3", "M4", "M5", "M6"):
            last_day_modules.setdefault(mod, {})[ver] = (
                last_dd.module_sec(mod) if last_dd else 0.0
            )

    # ── Build markdown ──
    lines: list[str] = []
    a = lines.append

    day_label_start = f"{day_start:02d}"
    day_label_end = f"{day_end:02d}"

    # ═══════════════════════════════════════════════════════════════
    # Section 1: 测试概述
    # ═══════════════════════════════════════════════════════════════
    a(f"# ChainSight 供应链仿真系统 BC算法优化测试报告 (Day {day_label_start}-{day_label_end})")
    a("")
    a("## 1. 测试概述")
    a("")
    a("### 1.1 测试目的")
    a(
        f"本报告对 ChainSight 供应链仿真系统 Day {day_label_start}-{day_label_end}"
        f" ({date_start} 至 {date_end}) 的仿真结果进行三版本对比测试，"
        f"验证代码重构后的功能正确性和性能提升效果。"
    )
    a("")
    a("### 1.2 测试版本")
    a("| 版本 | 说明 | 代码位置 |")
    a("|------|------|----------|")
    a("| **Dev** | 原始开发版本（基准版本） | `ChainSight_Dev/` |")
    a("| **Src** | 重构本地运行版本 | `src/` |")
    a("| **DB** | 重构数据库集成版本 | `src/` + `--use-db` |")
    a("")
    a("### 1.3 测试环境")
    a("- **操作系统**: Windows 11")
    a("- **Python版本**: 3.12.9")
    a("- **数据库**: PostgreSQL 5432 + DuckDB (内存模式)")
    a("- **测试数据**: BC_S5.xlsx 配置文件")
    a("- **仿真周期**: 87天 (2025-10-05 至 2025-12-30)")
    a(f"- **当前区间**: Day {day_label_start}-{day_label_end} ({date_start} 至 {date_end})")
    a("- **随机种子**: 42（确保可重复性）")
    a("- **测试时间**: 2026-02-12")
    a("")
    a("### 1.4 测试输出目录")
    a("| 版本 | 输出路径 |")
    a("|------|----------|")
    for ver, opath in OUTPUT_PATHS.items():
        a(f"| {ver} | `{opath}` |")
    a("")
    a("---")
    a("")

    # ═══════════════════════════════════════════════════════════════
    # Section 2: 性能对比分析
    # ═══════════════════════════════════════════════════════════════
    a("## 2. 性能对比分析")
    a("")
    a("### 2.1 区间运行时间")
    a("| 版本 | 区间总时间 | 平均每天 |")
    a("|------|------------|----------|")
    for ver in ("Dev", "Src", "DB"):
        t = interval_total[ver]
        avg = interval_avg[ver]
        a(f"| **{ver}** | {_fmt_sec(t)} | {avg:.1f}秒 |")
    a("")

    a("### 2.2 性能提升比例")
    a("| 对比项 | 加速倍数 | 时间减少百分比 |")
    a("|--------|----------|----------------|")
    dev_t = interval_total["Dev"]
    src_t = interval_total["Src"]
    db_t = interval_total["DB"]
    a(f"| **Src vs Dev** | **{_speedup(dev_t, src_t)}** | {_pct_reduction(dev_t, src_t)} |")
    a(f"| **DB vs Dev** | **{_speedup(dev_t, db_t)}** | {_pct_reduction(dev_t, db_t)} |")
    a(f"| **DB vs Src** | **{_speedup(src_t, db_t)}** | {_pct_reduction(src_t, db_t)} |")
    a("")

    mod_names = {
        "M1": "M1 订单生成",
        "M3": "M3 净需求计算",
        "M4": "M4 生产计划",
        "M5": "M5 部署规划",
        "M6": "M6 物流执行",
    }

    a("### 2.3 各模块平均耗时对比（秒/天）")
    a("| 模块 | Dev | Src | DB | Src加速 | DB加速 |")
    a("|------|-----|-----|-----|---------|--------|")
    for mod in ("M1", "M3", "M5"):
        d = module_avg[mod]["Dev"]
        s = module_avg[mod]["Src"]
        b = module_avg[mod]["DB"]
        a(f"| **{mod_names[mod]}** | ~{d:.1f} | ~{s:.1f} | ~{b:.1f} | {_speedup(d, s)} | {_speedup(d, b)} |")
    a("")

    a("### 2.4 每日详细耗时（区间最后一天）")
    last_date = _sim_date(day_end)
    a(f"#### Day {day_label_end} ({last_date}) 各模块耗时（秒）")
    a("| 模块 | Dev | Src | DB |")
    a("|------|-----|-----|-----|")
    for mod in ("M1", "M3", "M5", "M6"):
        vals = []
        for ver in ("Dev", "Src", "DB"):
            v = last_day_modules[mod][ver]
            vals.append(f"~{v:.1f}" if v > 0 else "~0")
        a(f"| {mod_names[mod]} | {vals[0]} | {vals[1]} | {vals[2]} |")
    a("")
    a("---")
    a("")

    # ═══════════════════════════════════════════════════════════════
    # Section 3: 数据一致性验证（按表名对比）
    # ═══════════════════════════════════════════════════════════════
    a("## 3. 数据一致性验证")
    a("")
    a("### 3.1 总体验证结果")
    a("")

    # Build interval-level summary per table
    table_interval_summary: list[dict[str, Any]] = []
    for _mod_label, _mod_cn, tables in MODULE_TABLES:
        for tkey, tcn in tables:
            tbl_data = comparisons.get(tkey)
            if tbl_data is None:
                continue
            daily_list = tbl_data.get("daily", [])

            # Filter to this interval
            interval_days = [
                dd for dd in daily_list
                if day_start <= dd["day"] <= day_end
            ]

            dev_rows = sum(dd["dev_rows"] for dd in interval_days)
            src_rows = sum(dd["src_rows"] for dd in interval_days)
            db_rows = sum(dd["db_rows"] for dd in interval_days)

            ds_all_match = all(
                dd["dev_vs_src"]["content_match"] for dd in interval_days
            ) if interval_days else True
            dd_all_match = all(
                dd["dev_vs_db"]["content_match"] for dd in interval_days
            ) if interval_days else True
            sd_all_match = all(
                dd["src_vs_db"]["content_match"] for dd in interval_days
            ) if interval_days else True

            table_interval_summary.append({
                "module": _mod_label,
                "table_key": tkey,
                "table_cn": tcn,
                "dev_rows": dev_rows,
                "src_rows": src_rows,
                "db_rows": db_rows,
                "ds_content_match": ds_all_match,
                "dd_content_match": dd_all_match,
                "sd_content_match": sd_all_match,
            })

    a("| 模块 | 表名 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |")
    a("|------|------|---------|---------|--------|------------|-----------|-----------|")
    for ts in table_interval_summary:
        ds_mark = "PASS" if ts["ds_content_match"] else "FAIL"
        dd_mark = "PASS" if ts["dd_content_match"] else "FAIL"
        sd_mark = "PASS" if ts["sd_content_match"] else "FAIL"
        a(
            f"| {ts['module']} | {ts['table_cn']} | {ts['dev_rows']:,} | "
            f"{ts['src_rows']:,} | {ts['db_rows']:,} | "
            f"{ds_mark} | {dd_mark} | {sd_mark} |"
        )
    a("")

    # Count pass/fail
    total_tables = len(table_interval_summary)
    ds_pass = sum(1 for ts in table_interval_summary if ts["ds_content_match"])
    dd_pass = sum(1 for ts in table_interval_summary if ts["dd_content_match"])
    sd_pass = sum(1 for ts in table_interval_summary if ts["sd_content_match"])
    ds_fail = total_tables - ds_pass
    dd_fail = total_tables - dd_pass
    sd_fail = total_tables - sd_pass

    a(
        f"> **Dev vs Src**: {ds_pass}/{total_tables} 表全部PASS"
        + ("" if ds_fail == 0 else f"，{ds_fail}表FAIL")
        + "  "
    )
    a(
        f"> **Dev vs DB**: {dd_pass}/{total_tables} 表PASS"
        + ("" if dd_fail == 0 else f"，{dd_fail}表FAIL")
        + "  "
    )
    a(
        f"> **Src vs DB**: {sd_pass}/{total_tables} 表PASS"
        + ("" if sd_fail == 0 else f"，{sd_fail}表FAIL")
    )
    a("")

    # (Known discrepancy block removed — bug fixed, all versions now consistent)

    # ── 3.2+ Per-module detailed tables ──
    section_idx = 2
    for mod_label, mod_cn, tables in MODULE_TABLES:
        a(f"### 3.{section_idx} {mod_label} - {mod_cn}")
        a("")

        for tkey, tcn in tables:
            tbl_data = comparisons.get(tkey)
            if tbl_data is None:
                a(f"#### {tcn}")
                a("")
                a("> 无对比数据")
                a("")
                continue

            daily_list = tbl_data.get("daily", [])
            interval_days = [
                dd for dd in daily_list
                if day_start <= dd["day"] <= day_end
            ]

            a(f"#### {tcn}")
            a("")
            a(
                f"- **总行数 (全{SIM_TOTAL_DAYS}天)**: Dev={tbl_data['total_dev_rows']:,}, "
                f"Src={tbl_data['total_src_rows']:,}, "
                f"DB={tbl_data['total_db_rows']:,}"
            )
            all_daily = tbl_data.get("daily", [])
            global_ds = all(dd["dev_vs_src"]["content_match"] for dd in all_daily) if all_daily else True
            global_dd = all(dd["dev_vs_db"]["content_match"] for dd in all_daily) if all_daily else True
            global_sd = all(dd["src_vs_db"]["content_match"] for dd in all_daily) if all_daily else True
            a(
                f"- **全局一致性**: "
                f"Dev vs Src {'PASS' if global_ds else 'FAIL'}, "
                f"Dev vs DB {'PASS' if global_dd else 'FAIL'}, "
                f"Src vs DB {'PASS' if global_sd else 'FAIL'}"
            )
            a("")

            # Daily detail table for this interval
            a("| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |")
            a("|------|------|---------|---------|--------|------------|-----------|-----------|")

            interval_dev_total = 0
            interval_src_total = 0
            interval_db_total = 0
            interval_ds_pass = 0
            interval_dd_pass = 0
            interval_sd_pass = 0

            for dd in interval_days:
                day_num = dd["day"]
                date_str = dd["date"]
                dev_r = dd["dev_rows"]
                src_r = dd["src_rows"]
                db_r = dd["db_rows"]

                interval_dev_total += dev_r
                interval_src_total += src_r
                interval_db_total += db_r

                ds_match = dd["dev_vs_src"]["content_match"]
                dd_match = dd["dev_vs_db"]["content_match"]
                sd_match = dd["src_vs_db"]["content_match"]

                if ds_match:
                    interval_ds_pass += 1
                if dd_match:
                    interval_dd_pass += 1
                if sd_match:
                    interval_sd_pass += 1

                ds_mark = "PASS" if ds_match else "FAIL"
                dd_mark = "PASS" if dd_match else "FAIL"
                sd_mark = "PASS" if sd_match else "FAIL"

                # Add diff detail if FAIL (content mismatch)
                ds_detail = ""
                dd_detail = ""
                sd_detail = ""
                if not ds_match:
                    diff_cnt = dd["dev_vs_src"].get("diff_count", 0)
                    ds_detail = f" ({diff_cnt}处差异)"
                if not dd_match:
                    diff_cnt = dd["dev_vs_db"].get("diff_count", 0)
                    dd_detail = f" ({diff_cnt}处差异)"
                if not sd_match:
                    diff_cnt = dd["src_vs_db"].get("diff_count", 0)
                    sd_detail = f" ({diff_cnt}处差异)"

                a(
                    f"| Day {day_num:02d} | {date_str} | {dev_r:,} | {src_r:,} | "
                    f"{db_r:,} | {ds_mark}{ds_detail} | {dd_mark}{dd_detail} | {sd_mark}{sd_detail} |"
                )

            # Interval totals row
            a(
                f"| **合计** | | **{interval_dev_total:,}** | **{interval_src_total:,}** | "
                f"**{interval_db_total:,}** | "
                f"**{interval_ds_pass}/{len(interval_days)}天PASS** | "
                f"**{interval_dd_pass}/{len(interval_days)}天PASS** | "
                f"**{interval_sd_pass}/{len(interval_days)}天PASS** |"
            )
            a("")

        section_idx += 1

    a("---")
    a("")

    # ═══════════════════════════════════════════════════════════════
    # Section 4: 每日库存一致性验证（日志统计）
    # ═══════════════════════════════════════════════════════════════
    a("## 4. 每日库存一致性验证（日志统计）")
    a("")
    a("| 天数 | 日期 | Dev库存量 | Src库存量 | DB库存量 | 一致性 |")
    a("|------|------|-----------|-----------|----------|--------|")

    inconsistent_count = 0
    first_inv: int | None = None
    last_inv: int | None = None

    for d in range(day_start, day_end + 1):
        d_date = _sim_date(d)
        inv: dict[str, int] = {}
        for ver in ("Dev", "Src", "DB"):
            dd = all_data[ver].get(d)
            qty = (
                dd.stats.get("total_inventory_quantity", 0)
                if dd and dd.stats else 0
            )
            inv[ver] = qty

        if first_inv is None:
            first_inv = inv["Dev"]
        last_inv = inv["Dev"]

        consistent = (inv["Dev"] == inv["Src"] == inv["DB"])
        mark = "PASS" if consistent else "FAIL"
        if not consistent:
            inconsistent_count += 1

        a(
            f"| Day {d:02d} | {d_date} | {inv['Dev']:,} | {inv['Src']:,} | "
            f"{inv['DB']:,} | {mark} |"
        )

    a("")
    a("### 4.1 区间数据一致性总结")
    a("| 验证项 | 结果 |")
    a("|--------|------|")
    a(f"| **区间天数** | {num_days}天 |")
    a(f"| **库存不一致天数** | {inconsistent_count}天 |")
    pct = ((num_days - inconsistent_count) / num_days * 100) if num_days else 0
    a(f"| **一致性比例** | {pct:.0f}% |")
    if first_inv is not None:
        a(f"| **区间起始库存** | {first_inv:,} |")
    else:
        a("| **区间起始库存** | N/A |")
    if last_inv is not None:
        a(f"| **区间结束库存** | {last_inv:,} |")
    else:
        a("| **区间结束库存** | N/A |")
    a("")
    a("---")
    a("")

    # ═══════════════════════════════════════════════════════════════
    # Section 5: 测试结论
    # ═══════════════════════════════════════════════════════════════
    a("## 5. 测试结论")
    a("")
    a("### 5.1 功能验证结论")
    a("")

    # Check overall pass/fail for this interval (based on content consistency)
    all_ds_pass = all(ts["ds_content_match"] for ts in table_interval_summary)
    all_dd_pass = all(ts["dd_content_match"] for ts in table_interval_summary)
    all_sd_pass = all(ts["sd_content_match"] for ts in table_interval_summary)

    if all_ds_pass and all_dd_pass and all_sd_pass:
        a("**重构版本(Src/DB)与原始版本(Dev)业务数据100%一致**")
    elif all_ds_pass and all_dd_pass:
        a(f"**Dev vs Src**: 全部{ds_pass}张表数据100%一致")
        a(f"**Dev vs DB**: 全部{dd_pass}张表数据100%一致")
        a(f"**Src vs DB**: {sd_pass}/{total_tables}张表数据一致")
    elif all_ds_pass:
        a(f"**Dev vs Src**: 全部{ds_pass}张表数据100%一致")
        a(f"**Dev vs DB**: {dd_pass}/{total_tables}张表数据一致")
    else:
        a(f"Dev vs Src: {ds_pass}/{total_tables}张表数据一致")
        a(f"Dev vs DB: {dd_pass}/{total_tables}张表数据一致")
    a("")

    a("验证覆盖（本区间）：")
    for ts in table_interval_summary:
        a(
            f"- {ts['module']} / {ts['table_cn']}: {ts['dev_rows']:,}行 "
            f"(Dev vs Src {'PASS' if ts['ds_content_match'] else 'FAIL'}, "
            f"Dev vs DB {'PASS' if ts['dd_content_match'] else 'FAIL'})"
        )
    a("")

    a("### 5.2 区间性能总结")
    a("| 指标 | Dev | Src | DB | 最佳提升 |")
    a("|------|-----|-----|-----|----------|")

    best_total_pct = _pct_reduction(dev_t, min(src_t, db_t))
    a(
        f"| 区间总时间 | {_fmt_sec_min(dev_t)} | {_fmt_sec_min(src_t)} | "
        f"{_fmt_sec_min(db_t)} | {best_total_pct} |"
    )

    for mod in ("M1", "M3", "M5"):
        d = module_avg[mod]["Dev"]
        s = module_avg[mod]["Src"]
        b = module_avg[mod]["DB"]
        best_pct = _pct_reduction(d, min(s, b))
        a(
            f"| {mod_names[mod]}平均耗时/天 | ~{d:.1f}秒 | ~{s:.1f}秒 | "
            f"~{b:.1f}秒 | {best_pct} |"
        )

    a("")
    a("---")
    a("")
    a("**报告生成时间**: 2026-02-26  ")
    a("**测试执行人**: chenxianyue002@chinasofti.com  ")
    a("**版本**: v7.0")
    a("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Parse all three logs
    print("Parsing simulation logs...")
    all_data: dict[str, dict[int, DayData]] = {}
    for ver, path in LOG_PATHS.items():
        print(f"  [{ver}] {path}")
        if not path.exists():
            print(f"  WARNING: log file not found - {path}")
            all_data[ver] = {}
            continue
        all_data[ver] = parse_log(path)
        print(f"  [{ver}] parsed {len(all_data[ver])} days")

    # 2. Quick validation
    for ver in ("Dev", "Src", "DB"):
        days = all_data[ver]
        if not days:
            continue
        missing = [d for d in range(1, SIM_TOTAL_DAYS + 1) if d not in days]
        if missing:
            print(f"  [{ver}] WARNING: missing days: {missing}")

    # 3. Load 3-way comparison JSON
    print(f"\nLoading comparison data from {JSON_PATH}...")
    if not JSON_PATH.exists():
        print(f"  ERROR: JSON file not found - {JSON_PATH}")
        return
    comparison_data = load_comparison_json(JSON_PATH)
    num_tables = len(comparison_data.get("comparisons", {}))
    print(f"  Loaded {num_tables} table comparisons")

    # 4. Generate reports
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ds, de in INTERVALS:
        d_start_str = _sim_date(ds).replace("-", "")
        d_end_str = _sim_date(de).replace("-", "")
        fname = f"BC算法优化测试报告_Day{ds:02d}-{de:02d}_{d_start_str}-{d_end_str}.md"
        out_path = OUTPUT_DIR / fname

        print(f"Generating: {fname}")
        md = generate_interval_report(ds, de, all_data, comparison_data)
        out_path.write_text(md, encoding="utf-8")
        print(f"  -> {out_path}")

    print(f"\nDone. Generated {len(INTERVALS)} reports in {OUTPUT_DIR}")

    # 5. Print summary table for quick check
    print(f"\n=== Quick Performance Summary (full {SIM_TOTAL_DAYS} days) ===")
    print(f"{'Version':<6} {'Total (sec)':>12} {'Avg/day (sec)':>14}")
    for ver in ("Dev", "Src", "DB"):
        total = sum(dd.day_total_seconds for dd in all_data[ver].values())
        avg = total / len(all_data[ver]) if all_data[ver] else 0
        print(f"{ver:<6} {total:>12.0f} {avg:>14.1f}")


if __name__ == "__main__":
    main()
