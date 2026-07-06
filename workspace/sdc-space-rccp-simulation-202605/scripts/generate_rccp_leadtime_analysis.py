from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import re

import pandas as pd
import psycopg


WORKSPACE_ROOT = Path(r"\\wsl.localhost\Ubuntu-22.04\home\zhangs37\ai_chainsight")
PROJECT_NAME = "sdc-space-rccp-simulation-202605"
SCENARIO_NAME = "baseline-current-params-should-be-rccp-otd"
PROJECT_DIR = WORKSPACE_ROOT / "workspace" / PROJECT_NAME
SCENARIO_DIR = PROJECT_DIR / "scenarios" / SCENARIO_NAME
DATA_DIR = PROJECT_DIR / "data"
DELIVERABLES_DIR = PROJECT_DIR / "deliverables"
BRIEF_FILE = PROJECT_DIR / "brief.md"
DESIGN_FILE = SCENARIO_DIR / "design.md"
RUN_RECORD_FILE = SCENARIO_DIR / "results" / "run_id.md"
DB_CREDENTIALS_FILE = WORKSPACE_ROOT / "db_credentials.yaml"
HTML_OUTPUT_FILE = PROJECT_DIR / "analysis.html"
CATEGORY_MAPPING_FILE = DATA_DIR / "ps_psc_sku_master_category_en.csv"


def load_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def extract_backtick_value(text: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}:\s*`([^`]+)`", text)
    if not match:
        raise ValueError(f"Could not find {label} in run record")
    return match.group(1)


def extract_section(text: str, heading: str) -> str:
    match = re.search(
        rf"^##\s+{re.escape(heading)}\s*$\n(.*?)(?=^##\s+|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def parse_bullets(section: str) -> list[str]:
  bullets: list[str] = []
  current: list[str] = []
  for raw_line in section.splitlines():
    line = raw_line.strip()
    if not line:
      continue
    if line.startswith("- "):
      if current:
        bullets.append(" ".join(current))
      current = [line[2:].strip()]
    elif current:
      current.append(line)
  if current:
    bullets.append(" ".join(current))
  return bullets


def normalize_paragraph(section: str) -> str:
    lines = [line.strip() for line in section.splitlines() if line.strip() and not line.startswith("-")]
    return " ".join(lines)


def extract_simulation_window(text: str) -> tuple[str | None, str | None]:
  match = re.search(r"Simulation period:\s*([0-9-]+)\s*to\s*([0-9-]+)", text)
  if not match:
    return None, None
  return match.group(1), match.group(2)


def to_frame(conn: psycopg.Connection, query: str, params: dict[str, object]) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=params)


def load_category_mapping(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["material", "category_en", "category", "bu_attr"])

    mapping = pd.read_csv(path, dtype={"material": str, "material_num": str})
    if "material" not in mapping.columns and "material_num" in mapping.columns:
        mapping = mapping.rename(columns={"material_num": "material"})

    keep_columns = [column for column in ["material", "category_en", "category", "bu_attr"] if column in mapping.columns]
    mapping = mapping.loc[:, keep_columns].copy()
    mapping["material"] = mapping["material"].astype(str).str.strip()
    return mapping.drop_duplicates(subset=["material"]).reset_index(drop=True)


def attach_category_mapping(frame: pd.DataFrame, category_mapping: pd.DataFrame) -> pd.DataFrame:
    merged = frame.copy()
    merged["material"] = merged["material"].astype(str).str.strip()
    if category_mapping.empty:
        merged["category_en"] = "(unmapped)"
        return merged

    merged = merged.merge(category_mapping.loc[:, ["material", "category_en"]], on="material", how="left")
    merged["category_en"] = merged["category_en"].fillna("(unmapped)")
    return merged


def build_category_rccp(
    daily_material_detail: pd.DataFrame,
    category_mapping: pd.DataFrame,
) -> pd.DataFrame:
    if daily_material_detail.empty:
        return pd.DataFrame(columns=["month", "location", "peak_date", "category_en", "peak_cbm"])

    merged = attach_category_mapping(daily_material_detail, category_mapping)

    daily_category = (
        merged.groupby(["month", "location", "inventory_date", "category_en"], as_index=False, dropna=False)["material_cbm"]
        .sum()
        .rename(columns={"inventory_date": "peak_date", "material_cbm": "daily_cbm"})
    )

    aggregated = (
        daily_category
        .sort_values(["month", "location", "category_en", "daily_cbm", "peak_date"], ascending=[True, True, True, False, True])
        .drop_duplicates(subset=["month", "location", "category_en"], keep="first")
        .rename(columns={"daily_cbm": "peak_cbm"})
        .sort_values(["month", "location", "peak_cbm", "category_en"], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )
    aggregated["peak_cbm"] = aggregated["peak_cbm"].round(2)
    return aggregated


def build_category_mapping_coverage(
    daily_material_detail: pd.DataFrame,
    category_mapping: pd.DataFrame,
    mapping_file_exists: bool,
) -> pd.DataFrame:
    if daily_material_detail.empty:
        return pd.DataFrame(columns=["metric", "value"])

    merged = attach_category_mapping(daily_material_detail, category_mapping)

    merged["mapped"] = merged["category_en"].notna()
    unique_materials = int(merged["material"].nunique())
    mapped_unique_materials = int(merged.loc[merged["mapped"], "material"].nunique())
    rows = [
        {"metric": "mapping_file_available", "value": "yes" if mapping_file_exists else "no"},
        {"metric": "mapping_file", "value": str(CATEGORY_MAPPING_FILE)},
        {"metric": "daily_material_rows", "value": int(len(merged))},
        {"metric": "unique_materials", "value": unique_materials},
        {"metric": "mapped_unique_materials", "value": mapped_unique_materials},
        {"metric": "unmapped_unique_materials", "value": int(merged.loc[~merged["mapped"], "material"].nunique())},
        {"metric": "mapped_unique_material_ratio", "value": round(mapped_unique_materials / unique_materials, 4) if unique_materials else None},
    ]
    return pd.DataFrame(rows)


def build_peakday_inventory(
    peak_days: pd.DataFrame,
    daily_inventory: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    join_cols = ["month", *group_cols, "peak_date"]
    inventory_join = daily_inventory.rename(columns={"inventory_date": "peak_date", "inventory_qty": "peak_inventory_qty"})
    result = peak_days.merge(inventory_join, on=join_cols, how="left")
    result["peak_inventory_qty"] = result["peak_inventory_qty"].fillna(0.0)
    return result


def calculate_dfc_metrics(
    inventory_qty: float,
    peak_date: object,
    weekly_forecast: dict[int, float],
    simulation_start_date: object,
    max_week: int,
) -> tuple[float | None, str, float, float]:
    inventory_qty = float(inventory_qty or 0.0)
    if inventory_qty <= 0:
        return 0.0, "zero_inventory", 0.0, 0.0

    peak_day = pd.Timestamp(peak_date).date()
    start_day = pd.Timestamp(simulation_start_date).date()
    day_index = (peak_day - start_day).days
    start_week = day_index // 7 + 1
    day_offset = day_index % 7
    modeled_days = 0.0
    remaining_qty = inventory_qty
    positive_future_demand = False

    for week in range(start_week, max_week + 1):
        week_qty = float(weekly_forecast.get(week, 0.0) or 0.0)
        available_days = (7 - day_offset) if week == start_week else 7
        if available_days <= 0:
            continue
        daily_qty = week_qty / 7.0
        if daily_qty <= 0:
            modeled_days += float(available_days)
            continue

        positive_future_demand = True
        covered_qty = daily_qty * available_days
        if remaining_qty <= covered_qty:
            modeled_days += remaining_qty / daily_qty
            return round(modeled_days, 2), "depletes_within_horizon", round(modeled_days, 2), 0.0

        remaining_qty -= covered_qty
        modeled_days += float(available_days)

    if not positive_future_demand:
        return None, "no_future_demand", round(modeled_days, 2), round(remaining_qty, 4)

    return None, "beyond_forecast_horizon", round(modeled_days, 2), round(remaining_qty, 4)


def build_peakday_dfc(
    peak_inventory: pd.DataFrame,
    weekly_forecast: pd.DataFrame,
    group_cols: list[str],
    simulation_start_date: object,
) -> pd.DataFrame:
    if peak_inventory.empty:
        return pd.DataFrame(columns=[*group_cols, "month", "peak_date", "peak_inventory_qty", "dfc_days", "dfc_status", "modeled_days", "remaining_qty_after_horizon"])

    forecast_lookup: dict[tuple[object, ...], dict[int, float]] = {}
    max_week = int(weekly_forecast["week"].max()) if not weekly_forecast.empty else 0

    if not weekly_forecast.empty:
        for key, group in weekly_forecast.groupby(group_cols, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            forecast_lookup[key_tuple] = {
                int(week): float(quantity)
                for week, quantity in zip(group["week"], group["forecast_qty"])
            }

    rows: list[dict[str, object]] = []
    for _, record in peak_inventory.iterrows():
        key_tuple = tuple(record[column] for column in group_cols)
        dfc_days, dfc_status, modeled_days, remaining_qty = calculate_dfc_metrics(
            inventory_qty=float(record.get("peak_inventory_qty", 0.0) or 0.0),
            peak_date=record["peak_date"],
            weekly_forecast=forecast_lookup.get(key_tuple, {}),
            simulation_start_date=simulation_start_date,
            max_week=max_week,
        )
        row = record.to_dict()
        row["dfc_days"] = dfc_days
        row["dfc_status"] = dfc_status
        row["modeled_days"] = modeled_days
        row["remaining_qty_after_horizon"] = remaining_qty
        rows.append(row)

    return pd.DataFrame(rows)


def number(value: object, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{decimals}f}"


def number_thousands(value: object, decimals: int = 1) -> str:
  if value is None or pd.isna(value):
    return "-"
  return f"{float(value) / 1000:,.{decimals}f}"


def percent(value: object, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.{decimals}f}%"


def days(value: object, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{decimals}f} d"


def frame_to_html(
    frame: pd.DataFrame,
    formatters: dict[str, callable] | None = None,
    max_rows: int | None = None,
) -> str:
    formatters = formatters or {}
    display_frame = frame.copy()
    if max_rows is not None:
        display_frame = display_frame.head(max_rows)

    head_html = "".join(f"<th>{escape(str(column))}</th>" for column in display_frame.columns)
    body_rows: list[str] = []
    for _, row in display_frame.iterrows():
        cells: list[str] = []
        for column in display_frame.columns:
            formatter = formatters.get(column, lambda item: "-" if pd.isna(item) else str(item))
            cells.append(f"<td>{escape(formatter(row[column]))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        "<div class='table-wrap'><table><thead><tr>"
        + head_html
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
    )


def series_bars(frame: pd.DataFrame, label_col: str, value_col: str, formatter: callable) -> str:
    if frame.empty:
        return "<p class='empty'>No data available.</p>"
    max_value = max(float(value) for value in frame[value_col] if not pd.isna(value))
    rows: list[str] = []
    for _, row in frame.iterrows():
        value = float(row[value_col])
        width = 0 if max_value == 0 else (value / max_value) * 100
        rows.append(
            "<div class='bar-row'>"
            f"<div class='bar-label'>{escape(str(row[label_col]))}</div>"
            f"<div class='bar-value'>{escape(formatter(row[value_col]))}</div>"
            f"<div class='bar-track'><div class='bar-fill' style='width:{width:.1f}%'></div></div>"
            "</div>"
        )
    return "<div class='bars'>" + "".join(rows) + "</div>"


def grouped_rccp_chart(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p class='empty'>No Space RCCP data available.</p>"

    pivot = (
        frame.loc[:, ["month", "location", "peak_cbm"]]
        .pivot(index="month", columns="location", values="peak_cbm")
        .fillna(0.0)
        .sort_index()
    )
    scaled = pivot / 1000.0
    months = list(scaled.index)
    locations = list(scaled.columns)
    max_value = max(float(scaled.to_numpy().max()), 1.0)

    colors = [
        "#0f766e", "#c97316", "#1d4ed8", "#b91c1c", "#7c3aed", "#0891b2",
        "#65a30d", "#be185d", "#9333ea", "#2563eb", "#b45309", "#0d9488",
    ]

    width = max(1120, 180 + len(months) * max(145, len(locations) * 16 + 36))
    height = 520
    left = 72
    right = 28
    top = 30
    bottom = 110
    plot_width = width - left - right
    plot_height = height - top - bottom
    group_width = plot_width / max(len(months), 1)
    bar_gap = 2.5
    inner_padding = 18
    available_width = max(group_width - inner_padding * 2, 24)
    bar_width = max(6.0, min(16.0, (available_width - bar_gap * (len(locations) - 1)) / max(len(locations), 1)))
    cluster_width = len(locations) * bar_width + (len(locations) - 1) * bar_gap
    tick_count = 5

    parts: list[str] = [
        f"<div class='chart-shell'><svg viewBox='0 0 {width} {height}' role='img' aria-label='Space RCCP by month by DC, displayed as CBM divided by 1000'>"
    ]
    parts.append(
        f"<text x='{left}' y='18' fill='#5b6670' font-size='13' font-weight='600'>Space RCCP by month by DC (CBM / 1000)</text>"
    )

    for tick in range(tick_count + 1):
        tick_value = max_value * tick / tick_count
        y = top + plot_height - (tick_value / max_value) * plot_height
        parts.append(
            f"<line x1='{left}' y1='{y:.2f}' x2='{width - right}' y2='{y:.2f}' stroke='rgba(31,41,51,0.12)' stroke-width='1' />"
        )
        parts.append(
            f"<text x='{left - 10}' y='{y + 4:.2f}' text-anchor='end' fill='#5b6670' font-size='11'>{tick_value:,.0f}</text>"
        )

    parts.append(
        f"<line x1='{left}' y1='{top + plot_height}' x2='{width - right}' y2='{top + plot_height}' stroke='#1f2933' stroke-width='1.2' />"
    )

    for month_index, month in enumerate(months):
        group_left = left + month_index * group_width + (group_width - cluster_width) / 2
        for location_index, location in enumerate(locations):
            value = float(scaled.loc[month, location])
            bar_height = (value / max_value) * plot_height
            x = group_left + location_index * (bar_width + bar_gap)
            y = top + plot_height - bar_height
            color = colors[location_index % len(colors)]
            parts.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_width:.2f}' height='{bar_height:.2f}' fill='{color}' rx='2'>"
                f"<title>{escape(str(month))} | {escape(str(location))} | {value:,.1f} (CBM / 1000)</title>"
                "</rect>"
            )
        month_x = left + month_index * group_width + group_width / 2
        parts.append(
            f"<text x='{month_x:.2f}' y='{height - 52}' text-anchor='middle' fill='#1f2933' font-size='12'>{escape(str(month))}</text>"
        )

    axis_y = top + plot_height / 2
    parts.append(
        f"<text x='20' y='{axis_y:.2f}' transform='rotate(-90 20 {axis_y:.2f})' fill='#5b6670' font-size='12'>CBM / 1000</text>"
    )
    parts.append("</svg></div>")

    legend_items = []
    for location_index, location in enumerate(locations):
        color = colors[location_index % len(colors)]
        legend_items.append(
            "<span class='legend-item'>"
            f"<span class='legend-swatch' style='background:{color}'></span>{escape(str(location))}"
            "</span>"
        )
    parts.append("<div class='legend'>" + "".join(legend_items) + "</div>")
    return "".join(parts)


def service_line_chart(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p class='empty'>No service data available.</p>"

    service_frame = frame.copy()
    service_frame["service"] = pd.to_numeric(service_frame["service"], errors="coerce")
    pivot = (
        service_frame.loc[:, ["month", "location", "service"]]
        .pivot(index="month", columns="location", values="service")
        .sort_index()
    )
    months = list(pivot.index)
    locations = list(pivot.columns)
    max_value = pivot.max(skipna=True).max()
    if pd.isna(max_value) or max_value <= 0:
        max_value = 1.0
    max_value = max(1.0, float(max_value))
    max_value = min(1.05, max_value)

    colors = [
        "#0f766e", "#c97316", "#1d4ed8", "#b91c1c", "#7c3aed", "#0891b2",
        "#65a30d", "#be185d", "#9333ea", "#2563eb", "#b45309", "#0d9488",
    ]

    width = max(1120, 180 + len(months) * 120)
    height = 500
    left = 72
    right = 28
    top = 30
    bottom = 90
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_pos(index: int) -> float:
        if len(months) == 1:
            return left + plot_width / 2
        return left + index * (plot_width / (len(months) - 1))

    def y_pos(value: float) -> float:
        return top + plot_height - (value / max_value) * plot_height

    parts: list[str] = [
        f"<div class='chart-shell'><svg viewBox='0 0 {width} {height}' role='img' aria-label='Service by month by location line chart'>"
    ]
    parts.append(
        f"<text x='{left}' y='18' fill='#5b6670' font-size='13' font-weight='600'>Service by month by location</text>"
    )

    tick_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    if max_value > 1.0:
        tick_values.append(max_value)
    tick_values = sorted(set(tick_values))

    for tick_value in tick_values:
        y = y_pos(tick_value)
        parts.append(
            f"<line x1='{left}' y1='{y:.2f}' x2='{width - right}' y2='{y:.2f}' stroke='rgba(31,41,51,0.12)' stroke-width='1' />"
        )
        parts.append(
            f"<text x='{left - 10}' y='{y + 4:.2f}' text-anchor='end' fill='#5b6670' font-size='11'>{tick_value * 100:.0f}%</text>"
        )

    parts.append(
        f"<line x1='{left}' y1='{top + plot_height}' x2='{width - right}' y2='{top + plot_height}' stroke='#1f2933' stroke-width='1.2' />"
    )

    for month_index, month in enumerate(months):
        x = x_pos(month_index)
        parts.append(
            f"<text x='{x:.2f}' y='{height - 36}' text-anchor='middle' fill='#1f2933' font-size='12'>{escape(str(month))}</text>"
        )

    for location_index, location in enumerate(locations):
        color = colors[location_index % len(colors)]
        path_points: list[str] = []
        for month_index, month in enumerate(months):
            value = pivot.loc[month, location]
            if pd.isna(value):
                if path_points:
                    parts.append(
                        f"<polyline fill='none' stroke='{color}' stroke-width='2.2' points='{' '.join(path_points)}' />"
                    )
                    path_points = []
                continue
            x = x_pos(month_index)
            y = y_pos(float(value))
            path_points.append(f"{x:.2f},{y:.2f}")
            parts.append(
                f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3.4' fill='{color}'>"
                f"<title>{escape(str(location))} | {escape(str(month))} | {percent(value, 1)}</title>"
                "</circle>"
            )
        if path_points:
            parts.append(
                f"<polyline fill='none' stroke='{color}' stroke-width='2.2' points='{' '.join(path_points)}' />"
            )

    axis_y = top + plot_height / 2
    parts.append(
        f"<text x='20' y='{axis_y:.2f}' transform='rotate(-90 20 {axis_y:.2f})' fill='#5b6670' font-size='12'>Service</text>"
    )
    parts.append("</svg></div>")

    legend_items = []
    for location_index, location in enumerate(locations):
        color = colors[location_index % len(colors)]
        legend_items.append(
            "<span class='legend-item'>"
            f"<span class='legend-swatch' style='background:{color}'></span>{escape(str(location))}"
            "</span>"
        )
    parts.append("<div class='legend'>" + "".join(legend_items) + "</div>")
    return "".join(parts)


def build_html(
    run_id: str,
    config_name: str,
    brief_text: str,
    design_text: str,
    dc_locations: pd.DataFrame,
    rccp_network: pd.DataFrame,
    rccp_location: pd.DataFrame,
    cfr_by_month_location: pd.DataFrame,
    lane_lt: pd.DataFrame,
    leadtime_summary: pd.DataFrame,
    volume_coverage: pd.DataFrame,
    output_window: pd.DataFrame,
    excel_name: str,
) -> str:
    goal = normalize_paragraph(extract_section(brief_text, "Goal"))
    scope = parse_bullets(extract_section(brief_text, "Scope Hypothesis"))
    constraints = parse_bullets(extract_section(brief_text, "Constraints & Assumptions"))
    scenario_intent = normalize_paragraph(extract_section(design_text, "Scenario Intent"))
    tradeoffs = parse_bullets(extract_section(design_text, "Expected Trade-offs"))
    output_dates = output_window.iloc[0]
    _, planned_end = extract_simulation_window(brief_text)
    inventory_end = str(output_dates["inventory_end_date"])
    horizon_note = (
        f"Observed output ends before the planned simulation horizon: inventory through {inventory_end}, versus planned end {planned_end}. This chart uses the actual available monthly peaks in the database output."
        if planned_end and inventory_end < planned_end
        else f"Observed inventory coverage runs from {output_dates['inventory_start_date']} to {inventory_end}."
    )
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    chart_frame = rccp_location.sort_values(["month", "location"]).copy()
    visible_months = sorted(chart_frame["month"].dropna().unique())
    service_frame = (
        cfr_by_month_location.rename(columns={"receiving": "location", "cfr": "service"})
        .loc[lambda frame: frame["month"].isin(visible_months)]
        .sort_values(["month", "location"])
        .reset_index(drop=True)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SDC Space RCCP by Month by DC</title>
<style>
:root {{
  --paper: #f6f0e8;
  --ink: #1f2933;
  --muted: #5b6670;
  --panel: rgba(255,255,255,0.76);
  --line: rgba(31,41,51,0.12);
  --teal: #0f766e;
  --amber: #c97316;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: Aptos, "Segoe UI", sans-serif;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, rgba(15,118,110,0.09), transparent 28%),
    radial-gradient(circle at top right, rgba(201,115,22,0.10), transparent 26%),
    linear-gradient(180deg, #fbf8f3 0%, var(--paper) 100%);
}}
.page {{ max-width: 1280px; margin: 0 auto; padding: 28px 24px 48px; }}
.hero {{
  padding: 28px 30px;
  border: 1px solid var(--line);
  border-radius: 24px;
  background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(215,232,221,0.84));
  box-shadow: 0 24px 60px rgba(31,41,51,0.08);
}}
.eyebrow {{ letter-spacing: 0.14em; text-transform: uppercase; color: var(--teal); font-size: 12px; font-weight: 700; }}
h1 {{ font-family: Georgia, "Times New Roman", serif; font-size: 40px; line-height: 1.1; margin: 10px 0 8px; }}
.hero p {{ max-width: 920px; margin: 0; color: var(--muted); font-size: 16px; }}
.meta {{ margin-top: 14px; display: flex; flex-wrap: wrap; gap: 10px; }}
.pill {{ padding: 8px 12px; border-radius: 999px; background: rgba(255,255,255,0.72); border: 1px solid var(--line); color: var(--muted); font-size: 13px; }}
.grid {{ display: grid; gap: 18px; margin-top: 24px; }}
.two-col {{ grid-template-columns: 1fr 1fr; align-items: start; }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 22px; padding: 22px; box-shadow: 0 12px 32px rgba(31,41,51,0.06); margin-top: 24px; }}
h2 {{ margin: 34px 0 14px; font-size: 24px; }}
h3 {{ margin: 0 0 12px; font-size: 18px; }}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
th {{ color: var(--teal); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
ul {{ margin: 0; padding-left: 18px; }}
li + li {{ margin-top: 8px; }}
.note {{ margin-top: 14px; padding: 14px 16px; border-left: 4px solid var(--amber); background: rgba(201,115,22,0.08); border-radius: 14px; color: var(--muted); }}
.empty {{ color: var(--muted); font-style: italic; }}
.chart-shell {{ overflow-x: auto; padding-bottom: 8px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 10px 14px; margin-top: 16px; }}
.legend-item {{ display: inline-flex; align-items: center; gap: 8px; font-size: 13px; color: var(--ink); }}
.legend-swatch {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
.small {{ color: var(--muted); font-size: 13px; }}
@media (max-width: 900px) {{
    .two-col {{ grid-template-columns: 1fr; }}
  h1 {{ font-size: 30px; }}
}}
</style>
</head>
<body>
<div class="page">
  <section class="hero">
    <div class="eyebrow">ChainSight Result Analysis</div>
    <h1>SDC Space RCCP by month by DC</h1>
    <p>{escape(goal or scenario_intent)}</p>
    <div class="meta">
      <div class="pill">Scenario: {escape(SCENARIO_NAME)}</div>
      <div class="pill">Run ID: {escape(run_id)}</div>
      <div class="pill">Config: {escape(config_name)}</div>
      <div class="pill">Generated: {escape(generated_at)}</div>
      <div class="pill">Unit: CBM / 1000</div>
    </div>
  </section>

  <section>
        <h2>Background and context</h2>
        <div class="grid two-col">
            <div class="panel">
                <h3>Business context</h3>
                <p>{escape(goal or scenario_intent)}</p>
                <h3 style="margin-top:16px;">Scenario intent</h3>
                <p>{escape(scenario_intent or goal)}</p>
                <h3 style="margin-top:16px;">Expected trade-offs</h3>
                <ul>{''.join(f'<li>{escape(item)}</li>' for item in tradeoffs) or '<li>No explicit trade-off notes recorded in design.md.</li>'}</ul>
            </div>
            <div class="panel">
                <h3>Scope</h3>
                <ul>{''.join(f'<li>{escape(item)}</li>' for item in scope) or '<li>No scope bullets recorded in brief.md.</li>'}</ul>
                <h3 style="margin-top:16px;">Assumptions and constraints</h3>
                <ul>{''.join(f'<li>{escape(item)}</li>' for item in constraints) or '<li>No assumption bullets recorded in brief.md.</li>'}</ul>
            </div>
        </div>
    </section>

    <section>
    <h2>Space RCCP chart</h2>
    <div class="panel">
      <p>This report only keeps the requested <strong>by month by DC</strong> Space RCCP bar chart. All displayed values are the original CBM values divided by <strong>1000</strong>.</p>
      {grouped_rccp_chart(chart_frame)}
      <div class="note">{escape(horizon_note)}</div>
    </div>
  </section>

    <section>
        <h2>Service by month by location</h2>
        <div class="panel">
            <p class="small">Service is calculated as shipment quantity divided by order quantity at month x location.</p>
            {service_line_chart(service_frame)}
        </div>
    </section>
</div>
</body>
</html>
"""


def main() -> None:
    credentials = load_simple_yaml(DB_CREDENTIALS_FILE)
    brief_text = BRIEF_FILE.read_text(encoding="utf-8")
    design_text = DESIGN_FILE.read_text(encoding="utf-8")
    run_record_text = RUN_RECORD_FILE.read_text(encoding="utf-8")
    run_id = extract_backtick_value(run_record_text, "Run ID")

    DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(
        host=credentials["host"],
        port=int(credentials["port"]),
        dbname=credentials["database"],
        user=credentials["user"],
        password=credentials["password"],
        connect_timeout=10,
    ) as conn:
        config_name = to_frame(
            conn,
            """
            select min(config_name) as config_name
            from orchestrator_unrestricted_inventory
            where run_id = %(run_id)s
            """,
            {"run_id": run_id},
        ).iloc[0]["config_name"]

        dc_locations = to_frame(
            conn,
            """
            select distinct location
            from cfg_global_network
            where config_name = %(config_name)s and location_type = 'DC'
            order by location
            """,
            {"config_name": config_name},
        )

        cfr_by_month_location = to_frame(
            conn,
            """
            with dc as (
                select distinct location
                from cfg_global_network
                where config_name = %(config_name)s and location_type = 'DC'
            ), ord as (
                select to_char(order_rows.date::date, 'YYYY-MM') as month,
                       order_rows.location as receiving,
                       sum(order_rows.quantity) as order_qty
                from (
                    select distinct date, material, location, demand_type, quantity, advance_days
                    from module1_output_orderlog
                    where run_id = %(run_id)s
                ) order_rows
                join dc on dc.location = order_rows.location
                group by 1, 2
            ), shp as (
                select to_char(shipment_rows.date::date, 'YYYY-MM') as month,
                       shipment_rows.location as receiving,
                       sum(shipment_rows.quantity) as shipment_qty
                from module1_output_shipmentlog shipment_rows
                join dc on dc.location = shipment_rows.location
                where shipment_rows.run_id = %(run_id)s
                group by 1, 2
            )
            select
                coalesce(ord.month, shp.month) as month,
                coalesce(ord.receiving, shp.receiving) as receiving,
                round(coalesce(shp.shipment_qty, 0)::numeric, 2) as shipment_qty,
                round(coalesce(ord.order_qty, 0)::numeric, 2) as order_qty,
                round((coalesce(shp.shipment_qty, 0) / nullif(ord.order_qty, 0))::numeric, 4) as cfr
            from ord
            full outer join shp
                on shp.month = ord.month
               and shp.receiving = ord.receiving
            order by 1, 2
            """,
            {"run_id": run_id, "config_name": config_name},
        )

        rccp_location = to_frame(
            conn,
            """
            with dc as (
                select distinct location
                from cfg_global_network
                where config_name = %(config_name)s and location_type = 'DC'
            ), daily as (
                select
                    to_char(i.date::date, 'YYYY-MM') as month,
                    i.date::date as peak_date,
                    i.location,
                    sum(i.quantity * coalesce(m.demand_unit_to_volume, 0)) as daily_cbm
                from orchestrator_unrestricted_inventory i
                join dc on dc.location = i.location
                left join cfg_m6_materialmd m
                    on m.config_name = i.config_name
                   and m.material = i.material
                where i.run_id = %(run_id)s
                group by 1, 2, 3
            ), ranked as (
                select
                    month,
                    peak_date,
                    location,
                    daily_cbm,
                    row_number() over (
                        partition by month, location
                        order by daily_cbm desc, peak_date asc
                    ) as row_num
                from daily
            )
            select month, location, peak_date, round(daily_cbm::numeric, 2) as peak_cbm
            from ranked
            where row_num = 1
            order by month, peak_cbm desc, location
            """,
            {"run_id": run_id, "config_name": config_name},
        )

        rccp_daily_material_detail = to_frame(
            conn,
            """
            with dc as (
                select distinct location
                from cfg_global_network
                where config_name = %(config_name)s and location_type = 'DC'
            )
            select
                to_char(i.date::date, 'YYYY-MM') as month,
                i.location,
                i.date::date as inventory_date,
                i.material,
                round(sum(i.quantity)::numeric, 4) as material_qty,
                round(sum(i.quantity * coalesce(m.demand_unit_to_volume, 0))::numeric, 2) as material_cbm
            from orchestrator_unrestricted_inventory i
            join dc on dc.location = i.location
            left join cfg_m6_materialmd m
                on m.config_name = i.config_name
               and m.material = i.material
            where i.run_id = %(run_id)s
            group by 1, 2, 3, 4
            order by month, location, inventory_date, material_cbm desc, i.material
            """,
            {"run_id": run_id, "config_name": config_name},
        )

        weekly_forecast = to_frame(
            conn,
            """
            select
                week,
                material,
                location,
                round(sum(quantity)::numeric, 6) as forecast_qty
            from cfg_m1_demandforecast
            where config_name = %(config_name)s
            group by 1, 2, 3
            order by week, material, location
            """,
            {"config_name": config_name},
        )

        rccp_network = to_frame(
            conn,
            """
            with dc as (
                select distinct location
                from cfg_global_network
                where config_name = %(config_name)s and location_type = 'DC'
            ), daily as (
                select
                    to_char(i.date::date, 'YYYY-MM') as month,
                    i.date::date as peak_date,
                    sum(i.quantity * coalesce(m.demand_unit_to_volume, 0)) as daily_cbm
                from orchestrator_unrestricted_inventory i
                join dc on dc.location = i.location
                left join cfg_m6_materialmd m
                    on m.config_name = i.config_name
                   and m.material = i.material
                where i.run_id = %(run_id)s
                group by 1, 2
            ), ranked as (
                select
                    month,
                    peak_date,
                    daily_cbm,
                    row_number() over (
                        partition by month
                        order by daily_cbm desc, peak_date asc
                    ) as row_num
                from daily
            )
            select month, peak_date, round(daily_cbm::numeric, 2) as peak_cbm
            from ranked
            where row_num = 1
            order by month
            """,
            {"run_id": run_id, "config_name": config_name},
        )

        lane_lt = to_frame(
            conn,
            """
            with deliveries as (
                select
                    sending,
                    receiving,
                    ori_deployment_uid,
                    min(planned_deployment_date::date) as planned_date,
                    min(actual_ship_date::date) as ship_date,
                    max(actual_delivery_date::date) as delivery_date,
                    sum(delivery_qty) as delivery_qty
                from module6_output_deliveryplan
                where run_id = %(run_id)s
                  and sending <> receiving
                  and planned_deployment_date is not null
                  and actual_ship_date is not null
                  and actual_delivery_date is not null
                group by 1, 2, 3
            ), lane_stats as (
                select
                    sending,
                    receiving,
                    count(*) as deployment_count,
                    round(sum(delivery_qty)::numeric, 2) as total_qty,
                    round(avg(ship_date - planned_date)::numeric, 2) as mean_wait_moq_days,
                    round(percentile_cont(0.5) within group (order by ship_date - planned_date)::numeric, 2) as median_wait_moq_days,
                    round(percentile_cont(0.9) within group (order by ship_date - planned_date)::numeric, 2) as p90_wait_moq_days,
                    round(avg(delivery_date - ship_date)::numeric, 2) as mean_otd_days,
                    round(percentile_cont(0.5) within group (order by delivery_date - ship_date)::numeric, 2) as median_otd_days,
                    round(percentile_cont(0.9) within group (order by delivery_date - ship_date)::numeric, 2) as p90_otd_days,
                    round(avg(delivery_date - planned_date)::numeric, 2) as mean_total_days,
                    round(percentile_cont(0.5) within group (order by delivery_date - planned_date)::numeric, 2) as median_total_days,
                    round(percentile_cont(0.9) within group (order by delivery_date - planned_date)::numeric, 2) as p90_total_days
                from deliveries
                group by 1, 2
            )
            select
                lane_stats.sending,
                lane_stats.receiving,
                lane_stats.deployment_count,
                lane_stats.total_qty,
                lane_stats.mean_wait_moq_days,
                lane_stats.median_wait_moq_days,
                lane_stats.p90_wait_moq_days,
                lane_stats.mean_otd_days,
                lane_stats.median_otd_days,
                lane_stats.p90_otd_days,
                lane_stats.mean_total_days,
                lane_stats.median_total_days,
                lane_stats.p90_total_days,
                leadtime.pdt as cfg_pdt_days,
                leadtime.gr as cfg_gr_days,
                leadtime.mct as cfg_mct_days,
                leadtime.otd as cfg_otd_days,
                round((coalesce(leadtime.pdt, 0) + coalesce(leadtime.gr, 0) + coalesce(leadtime.mct, 0) + coalesce(leadtime.otd, 0))::numeric, 2) as cfg_total_days,
                round((lane_stats.mean_total_days - (coalesce(leadtime.pdt, 0) + coalesce(leadtime.gr, 0) + coalesce(leadtime.mct, 0) + coalesce(leadtime.otd, 0)))::numeric, 2) as mean_gap_vs_cfg_days
            from lane_stats
            left join cfg_global_leadtime leadtime
                on leadtime.config_name = %(config_name)s
               and leadtime.sending = lane_stats.sending
               and leadtime.receiving = lane_stats.receiving
            order by lane_stats.total_qty desc, lane_stats.receiving, lane_stats.sending
            """,
            {"run_id": run_id, "config_name": config_name},
        )

        leadtime_summary = to_frame(
            conn,
            """
            with deliveries as (
                select
                    ori_deployment_uid,
                    min(planned_deployment_date::date) as planned_date,
                    min(actual_ship_date::date) as ship_date,
                    max(actual_delivery_date::date) as delivery_date,
                    sum(delivery_qty) as delivery_qty
                from module6_output_deliveryplan
                where run_id = %(run_id)s
                  and sending <> receiving
                  and planned_deployment_date is not null
                  and actual_ship_date is not null
                  and actual_delivery_date is not null
                group by 1
            )
            select
                count(*) as deployment_events,
                round(sum(delivery_qty)::numeric, 2) as total_qty,
                round(avg(ship_date - planned_date)::numeric, 2) as mean_wait_moq_days,
                round(avg(delivery_date - ship_date)::numeric, 2) as mean_otd_days,
                round(avg(delivery_date - planned_date)::numeric, 2) as mean_total_days
            from deliveries
            """,
            {"run_id": run_id},
        )

        volume_coverage = to_frame(
            conn,
            """
            with dc as (
                select distinct location
                from cfg_global_network
                where config_name = %(config_name)s and location_type = 'DC'
            )
            select
                count(distinct i.material) as inventory_materials,
                count(distinct case when coalesce(m.demand_unit_to_volume, 0) > 0 then i.material end) as materials_with_volume,
                count(distinct case when coalesce(m.demand_unit_to_volume, 0) <= 0 then i.material end) as zero_or_missing_volume_materials,
              round(
                (
                  sum(case when coalesce(m.demand_unit_to_volume, 0) > 0 then i.quantity else 0 end)
                  / nullif(sum(i.quantity), 0)
                )::numeric,
                4
              ) as qty_with_volume_ratio
            from orchestrator_unrestricted_inventory i
            join dc on dc.location = i.location
            left join cfg_m6_materialmd m
                on m.config_name = i.config_name
               and m.material = i.material
            where i.run_id = %(run_id)s
            """,
            {"run_id": run_id, "config_name": config_name},
        )

        output_window = to_frame(
            conn,
            """
            select
                (select min(date)::date from orchestrator_unrestricted_inventory where run_id = %(run_id)s) as inventory_start_date,
                (select max(date)::date from orchestrator_unrestricted_inventory where run_id = %(run_id)s) as inventory_end_date,
                (select min(actual_delivery_date)::date from module6_output_deliveryplan where run_id = %(run_id)s) as delivery_start_date,
                (select max(actual_delivery_date)::date from module6_output_deliveryplan where run_id = %(run_id)s) as delivery_end_date
            """,
            {"run_id": run_id},
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_name = f"{PROJECT_NAME}_result_summary_{timestamp}.xlsx"
    excel_output_file = DELIVERABLES_DIR / excel_name
    category_mapping_exists = CATEGORY_MAPPING_FILE.exists()
    category_mapping = load_category_mapping(CATEGORY_MAPPING_FILE)
    rccp_category_en = build_category_rccp(rccp_daily_material_detail, category_mapping)
    category_mapping_coverage = build_category_mapping_coverage(
        rccp_daily_material_detail,
        category_mapping,
        category_mapping_exists,
    )
    simulation_start_date = pd.to_datetime(rccp_daily_material_detail["inventory_date"]).min()

    daily_location_inventory = (
        rccp_daily_material_detail.groupby(["month", "location", "inventory_date"], as_index=False)["material_qty"]
        .sum()
        .rename(columns={"material_qty": "inventory_qty"})
    )
    dfc_location_peakday = build_peakday_dfc(
        peak_inventory=build_peakday_inventory(rccp_location, daily_location_inventory, ["location"]),
        weekly_forecast=(
            weekly_forecast.groupby(["location", "week"], as_index=False)["forecast_qty"]
            .sum()
            .sort_values(["location", "week"])
            .reset_index(drop=True)
        ),
        group_cols=["location"],
        simulation_start_date=simulation_start_date,
    )

    daily_category_inventory = (
        attach_category_mapping(rccp_daily_material_detail.loc[:, ["month", "location", "inventory_date", "material", "material_qty"]], category_mapping)
        .groupby(["month", "location", "inventory_date", "category_en"], as_index=False)["material_qty"]
        .sum()
        .rename(columns={"material_qty": "inventory_qty"})
    )
    weekly_category_forecast = (
        attach_category_mapping(weekly_forecast.loc[:, ["week", "location", "material", "forecast_qty"]], category_mapping)
        .groupby(["location", "category_en", "week"], as_index=False)["forecast_qty"]
        .sum()
        .sort_values(["location", "category_en", "week"])
        .reset_index(drop=True)
    )
    dfc_location_category_peakday = build_peakday_dfc(
        peak_inventory=build_peakday_inventory(rccp_category_en, daily_category_inventory, ["location", "category_en"]),
        weekly_forecast=weekly_category_forecast,
        group_cols=["location", "category_en"],
        simulation_start_date=simulation_start_date,
    )

    metadata = pd.DataFrame(
        [
            {"field": "project", "value": PROJECT_NAME},
            {"field": "scenario", "value": SCENARIO_NAME},
            {"field": "run_id", "value": run_id},
            {"field": "config_name", "value": config_name},
            {"field": "inventory_start_date", "value": output_window.iloc[0]["inventory_start_date"]},
            {"field": "inventory_end_date", "value": output_window.iloc[0]["inventory_end_date"]},
            {"field": "delivery_start_date", "value": output_window.iloc[0]["delivery_start_date"]},
            {"field": "delivery_end_date", "value": output_window.iloc[0]["delivery_end_date"]},
            {"field": "generated_at", "value": datetime.now().isoformat(timespec="seconds")},
        ]
    )

    with pd.ExcelWriter(excel_output_file, engine="openpyxl") as writer:
        metadata.to_excel(writer, sheet_name="metadata", index=False)
        cfr_by_month_location.to_excel(writer, sheet_name="service_month_location", index=False)
        rccp_network.to_excel(writer, sheet_name="rccp_month", index=False)
        rccp_location.to_excel(writer, sheet_name="rccp_month_location", index=False)
        rccp_category_en.to_excel(writer, sheet_name="rccp_month_location_category_en", index=False)
        dfc_location_peakday.to_excel(writer, sheet_name="dfc_month_location_peakday", index=False)
        dfc_location_category_peakday.to_excel(writer, sheet_name="dfc_month_location_category_en", index=False)
        category_mapping_coverage.to_excel(writer, sheet_name="category_mapping_coverage", index=False)
        lane_lt.to_excel(writer, sheet_name="lane_leadtime", index=False)
        leadtime_summary.to_excel(writer, sheet_name="leadtime_summary", index=False)
        volume_coverage.to_excel(writer, sheet_name="conversion_coverage", index=False)

    html_text = build_html(
        run_id=run_id,
        config_name=config_name,
        brief_text=brief_text,
        design_text=design_text,
        dc_locations=dc_locations,
        rccp_network=rccp_network,
        rccp_location=rccp_location,
        cfr_by_month_location=cfr_by_month_location,
        lane_lt=lane_lt,
        leadtime_summary=leadtime_summary,
        volume_coverage=volume_coverage,
        output_window=output_window,
        excel_name=excel_name,
    )
    HTML_OUTPUT_FILE.write_text(html_text, encoding="utf-8")

    print(f"Excel written: {excel_output_file}")
    print(f"HTML written: {HTML_OUTPUT_FILE}")


if __name__ == "__main__":
    main()