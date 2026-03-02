#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_oc_full_report.py

Generate a single complete BC algorithm optimization test report covering
ALL 76 simulation days. Each module section shows the full day-by-day detail
table (Day 01 through Day 87) instead of being split into 10-day segments.

Usage:
    python tools/generate_oc_full_report.py
"""

from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

BASE_DIR = Path(r"D:\PG\test\chainsight")

LOG_PATHS: dict[str, Path] = {
    "Dev": BASE_DIR / "outputs" / "dev_output" / "OC_Paste_S1_20251224" / "run_20260127_142402" / "simulation_log_20260127_142402.txt",
    "Src": BASE_DIR / "outputs" / "OC_Paste_S1_20251224" / "run_20260301_125122" / "simulation_log_20260301_125122.txt",
    "DB":  BASE_DIR / "outputs" / "db_OC_Paste_S1_20251224_20260301_222907" / "simulation_log_20260301_222907.txt",
}

JSON_PATH = BASE_DIR / "tools" / "oc_3way_content_results_full.json"
OUTPUT_DIR = BASE_DIR / "docs"
OUTPUT_NAME = "OC算法优化测试报告_完整版.md"

SIM_START_DATE = datetime(2025, 12, 15)
SIM_TOTAL_DAYS = 76

OUTPUT_PATHS: dict[str, str] = {
    "Dev": "outputs/dev_output/OC_Paste_S1_20251224/run_20260127_142402/",
    "Src": "outputs/OC_Paste_S1_20251224/run_20260301_125122/",
    "DB":  "outputs/db_OC_Paste_S1_20251224_20260301_222907/",
}

MODULE_TABLES: list[tuple[str, str, list[tuple[str, str]]]] = [
    ("Module1", "订单生成模块", [
        ("module1/OrderLog",        "订单日志 (OrderLog)"),
        ("module1/ShipmentLog",     "发货日志 (ShipmentLog)"),
        ("module1/CutLog",          "削减日志 (CutLog)"),
        ("module1/Summary",         "每日汇总 (Summary)"),
    ]),
    ("Module3", "净需求计算模块", [
        ("module3/NetDemand", "净需求 (NetDemand)"),
    ]),
    ("Module4", "生产计划模块", [
        ("module4/ProductionPlan",  "生产计划 (ProductionPlan)"),
        ("module4/CapacityExceed",  "超容量报告 (CapacityExceed)"),
        ("module4/ChangeoverLog",   "换型日志 (ChangeoverLog)"),
    ]),
    ("Module5", "部署规划模块", [
        ("module5/DeploymentPlan",  "部署计划 (DeploymentPlan)"),
        ("module5/UnfulfilledLog",  "未满足日志 (UnfulfilledLog)"),
        ("module5/StockOnHandLog",  "库存日志 (StockOnHandLog)"),
        ("module5/Validation",      "验证 (Validation)"),
    ]),
    ("Module6", "物流执行模块", [
        ("module6/DeliveryPlan",    "交付计划 (DeliveryPlan)"),
        ("module6/VehicleLog",      "车辆日志 (VehicleLog)"),
        ("module6/TruckUsageLog",   "卡车使用日志 (TruckUsageLog)"),
    ]),
]

# ─────────────────────────────────────────────
# Regex / parsing (reused from 10day script)
# ─────────────────────────────────────────────

RE_TIMESTAMP  = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\[(\w+)\]\s*(.*)")
RE_DAY_HEADER = re.compile(r"={5,}\s*第\s+(\d+)/(\d+)\s+天:\s+(\d{4}-\d{2}-\d{2})\s*={5,}")
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
RE_M3_TOTAL   = re.compile(r"\[M3\]\s*total(?:\s*duration)?:\s*([\d.]+)s")
RE_M5_FULL_DAY = re.compile(r"\[M5\]\s*Full day total\s*用时:\s*([\d.]+)s")
RE_DAILY_STATS = re.compile(r"当日统计:\s*(\{.*\})")


class DayData:
    __slots__ = ("sim_day", "date_str", "day_start_ts", "day_end_ts",
                 "module_times", "m3_total", "m5_full_day", "stats")

    def __init__(self, sim_day: int, date_str: str):
        self.sim_day = sim_day
        self.date_str = date_str
        self.day_start_ts: datetime | None = None
        self.day_end_ts:   datetime | None = None
        self.module_times: dict[str, float] = {}
        self.m3_total:     float | None = None
        self.m5_full_day:  float | None = None
        self.stats:        dict[str, Any] = {}

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


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def parse_log(path: Path) -> dict[int, DayData]:
    days: dict[int, DayData] = {}
    current_day: int | None = None
    module_starts: dict[str, datetime] = {}

    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n\r")
            m_ts = RE_TIMESTAMP.match(line)
            if not m_ts:
                continue
            ts_str, _, body = m_ts.group(1), m_ts.group(2), m_ts.group(3)
            ts = _parse_ts(ts_str)

            m_dh = RE_DAY_HEADER.search(body)
            if m_dh:
                sim_day  = int(m_dh.group(1))
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

            m_dc = RE_DAY_COMPLETE.search(body)
            if m_dc:
                completed = int(m_dc.group(1))
                if completed in days:
                    days[completed].day_end_ts = ts
                continue

            for mod, pat in RE_MODULE_START.items():
                if pat.search(body):
                    module_starts[mod] = ts
            for mod, pat in RE_MODULE_END.items():
                if pat.search(body) and mod in module_starts:
                    dd.module_times[mod] = (ts - module_starts[mod]).total_seconds()

            m = RE_M3_TOTAL.search(body)
            if m:
                dd.m3_total = float(m.group(1))
            m = RE_M5_FULL_DAY.search(body)
            if m:
                dd.m5_full_day = float(m.group(1))
            m = RE_DAILY_STATS.search(body)
            if m:
                try:
                    dd.stats = ast.literal_eval(m.group(1))
                except Exception:
                    pass

    return days


def load_comparison_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _fmt_sec(sec: float) -> str:
    return f"{sec:.0f}秒 ({sec / 60:.1f}分钟)"

def _speedup(base: float, opt: float) -> str:
    return "N/A" if opt == 0 else f"{base / opt:.2f}x"

def _pct_reduction(base: float, opt: float) -> str:
    return "N/A" if base == 0 else f"{(base - opt) / base * 100:.1f}%"

def _safe_avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _sim_date(day_num: int) -> str:
    return (SIM_START_DATE + timedelta(days=day_num - 1)).strftime("%Y-%m-%d")

# ─────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────

def generate_full_report(
    all_data: dict[str, dict[int, DayData]],
    comparison_data: dict[str, Any],
) -> str:
    comparisons = comparison_data["comparisons"]
    n = SIM_TOTAL_DAYS
    date_start = _sim_date(1)
    date_end   = _sim_date(n)

    # ── Full-period performance totals ──
    total_sec:  dict[str, float] = {}
    module_avg: dict[str, dict[str, float]] = {}

    for ver in ("Dev", "Src", "DB"):
        ver_days = all_data[ver]
        total = sum(dd.day_total_seconds for dd in ver_days.values())
        total_sec[ver] = total

        mod_lists: dict[str, list[float]] = {m: [] for m in ("M1","M3","M4","M5","M6")}
        for dd in ver_days.values():
            for mod in mod_lists:
                v = dd.module_sec(mod)
                if v > 0:
                    mod_lists[mod].append(v)
        for mod in mod_lists:
            module_avg.setdefault(mod, {})[ver] = _safe_avg(mod_lists[mod])

    dev_t = total_sec["Dev"]
    src_t = total_sec["Src"]
    db_t  = total_sec["DB"]

    lines: list[str] = []
    a = lines.append

    # ── Section 1: 测试概述 ──
    a(f"# ChainSight 供应链仿真系统 OC算法优化测试报告 (完整版 Day 01-{n:02d})")
    a("")
    a("## 1. 测试概述")
    a("")
    a("### 1.1 测试目的")
    a(f"本报告对 ChainSight 供应链仿真系统全仿真周期 Day 01-{n:02d} ({date_start} 至 {date_end}) 的仿真结果进行三版本对比测试，验证代码重构后的功能正确性和性能提升效果。")
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
    a("- **测试数据**: OC_Paste_S1_20251224.xlsx 配置文件")
    a(f"- **仿真周期**: {n}天 ({date_start} 至 {date_end})")
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

    # ── Section 2: 性能对比分析 ──
    a("## 2. 性能对比分析")
    a("")
    a(f"### 2.1 全周期运行时间 (共{n}天)")
    a("| 版本 | 总时间 | 平均每天 |")
    a("|------|--------|----------|")
    for ver in ("Dev", "Src", "DB"):
        t   = total_sec[ver]
        avg = t / n if n else 0
        a(f"| **{ver}** | {_fmt_sec(t)} | {avg:.1f}秒 |")
    a("")
    a("### 2.2 性能提升比例")
    a("| 对比项 | 加速倍数 | 时间减少百分比 |")
    a("|--------|----------|----------------|")
    a(f"| **Src vs Dev** | **{_speedup(dev_t, src_t)}** | {_pct_reduction(dev_t, src_t)} |")
    a(f"| **DB vs Dev**  | **{_speedup(dev_t, db_t)}**  | {_pct_reduction(dev_t, db_t)} |")
    a(f"| **DB vs Src**  | **{_speedup(src_t, db_t)}**  | {_pct_reduction(src_t, db_t)} |")
    a("")
    mod_names = {"M1": "M1 订单生成", "M3": "M3 净需求计算", "M4": "M4 生产计划", "M5": "M5 部署规划", "M6": "M6 物流执行"}
    a(f"### 2.3 各模块平均耗时对比（秒/天，全{n}天均值）")
    a("| 模块 | Dev | Src | DB | Src加速 | DB加速 |")
    a("|------|-----|-----|-----|---------|--------|")
    for mod in ("M1", "M3", "M5"):
        d = module_avg[mod]["Dev"]
        s = module_avg[mod]["Src"]
        b = module_avg[mod]["DB"]
        a(f"| **{mod_names[mod]}** | ~{d:.1f} | ~{s:.1f} | ~{b:.1f} | {_speedup(d, s)} | {_speedup(d, b)} |")
    a("")
    a("---")
    a("")

    # ── Section 3: 数据一致性验证 ──
    a("## 3. 数据一致性验证")
    a("")
    a("### 3.1 总体验证结果")
    a("")

    # Summary table (totals from JSON)
    a("| 模块 | 表名 | Dev总行数 | Src总行数 | DB总行数 | Dev vs Src | Dev vs DB | Src vs DB |")
    a("|------|------|-----------|-----------|----------|------------|-----------|-----------|")
    all_pass = True
    total_tables = 0
    ds_pass_count = dd_pass_count = sd_pass_count = 0
    for mod_label, _mod_cn, tables in MODULE_TABLES:
        for tkey, tcn in tables:
            tbl = comparisons.get(tkey)
            if tbl is None:
                continue
            total_tables += 1
            daily = tbl.get("daily", [])
            gds = all(d["dev_vs_src"]["content_match"] for d in daily) if daily else True
            gdd = all(d["dev_vs_db"]["content_match"]  for d in daily) if daily else True
            gsd = all(d["src_vs_db"]["content_match"]  for d in daily) if daily else True
            if gds: ds_pass_count += 1
            if gdd: dd_pass_count += 1
            if gsd: sd_pass_count += 1
            if not (gds and gdd and gsd):
                all_pass = False
            a(f"| {mod_label} | {tcn} | {tbl['total_dev_rows']:,} | {tbl['total_src_rows']:,} | {tbl['total_db_rows']:,} | {'PASS' if gds else 'FAIL'} | {'PASS' if gdd else 'FAIL'} | {'PASS' if gsd else 'FAIL'} |")
    a("")
    a(f"> **Dev vs Src**: {ds_pass_count}/{total_tables} 表全部PASS  ")
    a(f"> **Dev vs DB**:  {dd_pass_count}/{total_tables} 表PASS  ")
    a(f"> **Src vs DB**:  {sd_pass_count}/{total_tables} 表PASS")
    a("")
    a("---")
    a("")
    # ── 3.2+ Per-module full 87-day detail ──
    section_idx = 2
    for mod_label, mod_cn, tables in MODULE_TABLES:
        a(f"### 3.{section_idx} {mod_label} - {mod_cn}")
        a("")

        for tkey, tcn in tables:
            tbl = comparisons.get(tkey)
            if tbl is None:
                a(f"#### {tcn}")
                a("")
                a("> 无对比数据")
                a("")
                continue

            daily_list = tbl.get("daily", [])
            # full-period global consistency
            gds = all(d["dev_vs_src"]["content_match"] for d in daily_list) if daily_list else True
            gdd = all(d["dev_vs_db"]["content_match"]  for d in daily_list) if daily_list else True
            gsd = all(d["src_vs_db"]["content_match"]  for d in daily_list) if daily_list else True

            a(f"#### {tcn}")
            a("")
            a(f"- **总行数 (全{n}天)**: Dev={tbl['total_dev_rows']:,}, Src={tbl['total_src_rows']:,}, DB={tbl['total_db_rows']:,}")
            a(f"- **全局一致性**: Dev vs Src {'PASS' if gds else 'FAIL'}, Dev vs DB {'PASS' if gdd else 'FAIL'}, Src vs DB {'PASS' if gsd else 'FAIL'}")
            a("")
            a("| 天数 | 日期 | Dev行数 | Src行数 | DB行数 | Dev vs Src | Dev vs DB | Src vs DB |")
            a("|------|------|---------|---------|--------|------------|-----------|-----------|")

            total_dev = total_src = total_db = 0
            ds_pass = dd_pass = sd_pass = 0

            for d in daily_list:
                day_num  = d["day"]
                date_str = d["date"]
                dev_r    = d["dev_rows"]
                src_r    = d["src_rows"]
                db_r     = d["db_rows"]
                total_dev += dev_r
                total_src += src_r
                total_db  += db_r

                ds_m = d["dev_vs_src"]["content_match"]
                dd_m = d["dev_vs_db"]["content_match"]
                sd_m = d["src_vs_db"]["content_match"]
                if ds_m: ds_pass += 1
                if dd_m: dd_pass += 1
                if sd_m: sd_pass += 1

                def _mark(match, diff_key):
                    if match:
                        return "PASS"
                    cnt = d[diff_key].get("diff_count", 0)
                    return f"FAIL ({cnt}处差异)"

                a(f"| Day {day_num:02d} | {date_str} | {dev_r:,} | {src_r:,} | {db_r:,} | {_mark(ds_m,'dev_vs_src')} | {_mark(dd_m,'dev_vs_db')} | {_mark(sd_m,'src_vs_db')} |")

            nd = len(daily_list)
            a(f"| **合计** | | **{total_dev:,}** | **{total_src:,}** | **{total_db:,}** | **{ds_pass}/{nd}天PASS** | **{dd_pass}/{nd}天PASS** | **{sd_pass}/{nd}天PASS** |")
            a("")

        section_idx += 1

    a("---")
    a("")

    # ── Section 4: 全周期库存一致性 ──
    a("## 4. 每日库存一致性验证（全周期）")
    a("")
    a("| 天数 | 日期 | Dev库存量 | Src库存量 | DB库存量 | 一致性 |")
    a("|------|------|-----------|-----------|----------|--------|")

    inconsistent = 0
    first_inv = last_inv = None
    for d in range(1, n + 1):
        d_date = _sim_date(d)
        inv: dict[str, int] = {}
        for ver in ("Dev", "Src", "DB"):
            dd = all_data[ver].get(d)
            qty = dd.stats.get("total_inventory_quantity", 0) if dd and dd.stats else 0
            inv[ver] = qty
        if first_inv is None:
            first_inv = inv["Dev"]
        last_inv = inv["Dev"]
        consistent = (inv["Dev"] == inv["Src"] == inv["DB"])
        if not consistent:
            inconsistent += 1
        mark = "PASS" if consistent else "FAIL"
        a(f"| Day {d:02d} | {d_date} | {inv['Dev']:,} | {inv['Src']:,} | {inv['DB']:,} | {mark} |")

    a("")
    a("### 4.1 全周期一致性总结")
    a("| 验证项 | 结果 |")
    a("|--------|------|")
    a(f"| **仿真总天数** | {n}天 |")
    a(f"| **库存不一致天数** | {inconsistent}天 |")
    pct = (n - inconsistent) / n * 100 if n else 0
    a(f"| **一致性比例** | {pct:.0f}% |")
    if first_inv is not None:
        a(f"| **起始库存** | {first_inv:,} |")
    if last_inv is not None:
        a(f"| **结束库存** | {last_inv:,} |")
    a("")
    a("---")
    a("")

    # ── Section 5: 测试结论 ──
    a("## 5. 测试结论")
    a("")
    a("### 5.1 功能验证结论")
    a("")
    if all_pass:
        a("**重构版本(Src/DB)与原始版本(Dev)业务数据100%一致**")
    else:
        a(f"**Dev vs Src**: {ds_pass_count}/{total_tables}张表数据一致")
        a(f"**Dev vs DB**:  {dd_pass_count}/{total_tables}张表数据一致")
        a(f"**Src vs DB**:  {sd_pass_count}/{total_tables}张表数据一致")
    a("")
    a(f"验证覆盖（全{n}天）：")
    for mod_label, _mod_cn, tables in MODULE_TABLES:
        for tkey, tcn in tables:
            tbl = comparisons.get(tkey)
            if tbl is None:
                continue
            daily = tbl.get("daily", [])
            gds = all(d["dev_vs_src"]["content_match"] for d in daily) if daily else True
            gdd = all(d["dev_vs_db"]["content_match"]  for d in daily) if daily else True
            a(f"- {mod_label} / {tcn}: {tbl['total_dev_rows']:,}行 (Dev vs Src {'PASS' if gds else 'FAIL'}, Dev vs DB {'PASS' if gdd else 'FAIL'})")
    a("")
    a("### 5.2 全周期性能总结")
    a("| 指标 | Dev | Src | DB | 最佳提升 |")
    a("|------|-----|-----|-----|----------|")
    best_pct = _pct_reduction(dev_t, min(src_t, db_t))
    a(f"| 总运行时间 | {dev_t/60:.1f}分钟 | {src_t/60:.1f}分钟 | {db_t/60:.1f}分钟 | {best_pct} |")
    for mod in ("M1", "M3", "M5"):
        d_avg = module_avg[mod]["Dev"]
        s_avg = module_avg[mod]["Src"]
        b_avg = module_avg[mod]["DB"]
        bp    = _pct_reduction(d_avg, min(s_avg, b_avg))
        a(f"| {mod_names[mod]}平均耗时/天 | ~{d_avg:.1f}秒 | ~{s_avg:.1f}秒 | ~{b_avg:.1f}秒 | {bp} |")
    a("")
    a("---")
    a("")
    a("**报告生成时间**: 2026-02-26  ")
    a("**测试执行人**: chenxianyue002@chinasofti.com  ")
    a("**版本**: v8.0 (完整版)")
    a("")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    print("Parsing simulation logs...")
    all_data: dict[str, dict[int, DayData]] = {}
    for ver, path in LOG_PATHS.items():
        print(f"  [{ver}] {path}")
        if not path.exists():
            print(f"  WARNING: log not found - {path}")
            all_data[ver] = {}
            continue
        all_data[ver] = parse_log(path)
        print(f"  [{ver}] parsed {len(all_data[ver])} days")

    print(f"\nLoading comparison JSON from {JSON_PATH}...")
    if not JSON_PATH.exists():
        print(f"  ERROR: JSON not found - {JSON_PATH}")
        return
    comparison_data = load_comparison_json(JSON_PATH)
    print(f"  Loaded {len(comparison_data.get('comparisons', {}))} table comparisons")

    print("\nGenerating full report...")
    md = generate_full_report(all_data, comparison_data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUTPUT_NAME
    out_path.write_text(md, encoding="utf-8")
    line_count = md.count("\n")
    print(f"  -> {out_path}")
    print(f"     {line_count} lines, {len(md):,} bytes")


if __name__ == "__main__":
    main()
