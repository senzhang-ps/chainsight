"""Apply holiday-aware rescheduling to project_tracking_plan.xlsx.

2026 Chinese statutory holidays in project window (5/6 - 10/22):
  端午: 6/19 Fri
  中秋: 9/25 Fri
  国庆: 10/1 Thu, 10/2 Fri, 10/5 Mon, 10/6 Tue, 10/7 Wed (10/3-10/4 weekend)

Changes:
  1. Mark all weekend + holiday cells gray; blue only on workdays in task range
  2. MLE-2: end 6/26 → 6/29 (recover 1 day lost to 端午)
  3. DOC-4: end 9/25 → 9/28 (recover 1 day lost to 中秋)
  4. DOC-1: 10/1-10/9 → 10/8-10/16 (skip 国庆)
  5. FINAL: 10/12-10/15 → 10/19-10/22 (push back after DOC-1)
  6. Extend Gantt date columns from 10/15 to 10/22
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter

XLSX = Path("docs/project_implementation_details/project_tracking_plan.xlsx")
BLUE = PatternFill(start_color="FF9DC3E6", end_color="FF9DC3E6", fill_type="solid")
GRAY = PatternFill(start_color="FFE7E6E6", end_color="FFE7E6E6", fill_type="solid")
HOLIDAY = PatternFill(start_color="FFF4B084", end_color="FFF4B084", fill_type="solid")

HOLIDAYS_2026 = {
    date(2026, 6, 19): "端午",
    date(2026, 9, 25): "中秋",
    date(2026, 10, 1): "国庆",
    date(2026, 10, 2): "国庆",
    date(2026, 10, 5): "国庆",
    date(2026, 10, 6): "国庆",
    date(2026, 10, 7): "国庆",
}


def is_offday(d: date) -> bool:
    return d.weekday() >= 5 or d in HOLIDAYS_2026


def workdays_between(start: date, end: date) -> list[date]:
    days = []
    d = start
    while d <= end:
        if not is_offday(d):
            days.append(d)
        d += timedelta(days=1)
    return days


def date_col_map(ws):
    m = {}
    for c in range(1, ws.max_column + 1):
        v = ws.cell(row=1, column=c).value
        if hasattr(v, "strftime"):
            m[v.date()] = c
    return m


def extend_gantt_columns(ws, end_date: date):
    """Insert new date columns up to end_date, keeping trailing summary cols."""
    dcol = date_col_map(ws)
    last_existing = max(dcol.keys())
    if end_date <= last_existing:
        return dcol

    last_col = max(dcol.values())  # last date column
    n_to_add = (end_date - last_existing).days
    insert_at = last_col + 1  # position to insert (before 工时版/备注)

    ws.insert_cols(insert_at, n_to_add)

    # Fill in the new date headers and copy formatting from the prior column
    template_col = last_col
    for i in range(n_to_add):
        new_col = insert_at + i
        new_date = last_existing + timedelta(days=i + 1)
        cell = ws.cell(row=1, column=new_col)
        cell.value = datetime(new_date.year, new_date.month, new_date.day)
        # Copy header format from template
        tc = ws.cell(row=1, column=template_col)
        if tc.font:
            cell.font = Font(
                name=tc.font.name, size=tc.font.size, bold=tc.font.bold,
                color=tc.font.color
            )
        if tc.fill and tc.fill.fgColor:
            cell.fill = PatternFill(
                start_color=tc.fill.fgColor.rgb or "FFFFFFFF",
                end_color=tc.fill.fgColor.rgb or "FFFFFFFF",
                fill_type=tc.fill.fill_type,
            )
        if tc.alignment:
            cell.alignment = Alignment(
                horizontal=tc.alignment.horizontal,
                vertical=tc.alignment.vertical,
            )
        # Set column width similar to existing date cols
        ws.column_dimensions[get_column_letter(new_col)].width = (
            ws.column_dimensions[get_column_letter(template_col)].width or 4
        )

    return date_col_map(ws)


def repaint_row(ws, dcol, row, start: date, end: date):
    """Paint a Gantt row: blue on workdays inside range, gray on offdays inside range."""
    # First clear all cells in [first_date_col, last_date_col]
    first_col = min(dcol.values())
    last_col = max(dcol.values())
    for c in range(first_col, last_col + 1):
        d = next((dt for dt, col in dcol.items() if col == c), None)
        if d is None:
            continue
        cell = ws.cell(row=row, column=c)
        in_range = start <= d <= end
        if is_offday(d):
            cell.fill = GRAY
        elif in_range:
            cell.fill = BLUE
        else:
            cell.fill = PatternFill(fill_type=None)


def main():
    wb = openpyxl.load_workbook(XLSX)
    ws = wb["Gantt Chart"]

    # Step 1: extend Gantt columns to cover 10/22
    dcol = extend_gantt_columns(ws, date(2026, 10, 22))
    print(f"Gantt columns now span {min(dcol)} → {max(dcol)}")

    # Step 2: define final task schedule (post-holiday adjustment)
    schedule = {
        # task_id, gantt_row, start, end
        "P0":     (2,  date(2026, 5, 6),   date(2026, 5, 9)),
        "MLE-1":  (3,  date(2026, 5, 11),  date(2026, 5, 29)),
        "MLE-2":  (4,  date(2026, 6, 1),   date(2026, 6, 29)),  # +1 day for 端午
        "MLE-3":  (5,  date(2026, 7, 1),   date(2026, 7, 7)),
        "VAL-1":  (6,  date(2026, 7, 1),   date(2026, 7, 7)),
        "MLE-4":  (7,  date(2026, 7, 16),  date(2026, 7, 22)),
        "VAL-2":  (8,  date(2026, 7, 16),  date(2026, 7, 22)),
        "TEST-1": (9,  date(2026, 8, 3),   date(2026, 8, 14)),
        "DOC-2":  (10, date(2026, 8, 17),  date(2026, 8, 28)),
        "DOC-3":  (11, date(2026, 8, 31),  date(2026, 9, 11)),
        "DOC-4":  (12, date(2026, 9, 14),  date(2026, 9, 28)),  # +1 day for 中秋
        "DOC-1":  (13, date(2026, 10, 8),  date(2026, 10, 16)),  # skip 国庆
        "FINAL":  (14, date(2026, 10, 19), date(2026, 10, 22)),  # post-DOC-1
    }

    # Step 3: update Gantt D/E and repaint each row
    for tid, (row, st, en) in schedule.items():
        ws.cell(row=row, column=4).value = datetime(st.year, st.month, st.day)
        ws.cell(row=row, column=5).value = datetime(en.year, en.month, en.day)
        repaint_row(ws, dcol, row, st, en)
        wd = len(workdays_between(st, en))
        print(f"Gantt {tid} (row {row}): {st} → {en} ({wd} workdays)")

    # Step 4: paint holiday columns gray on row 1's secondary header? Not needed; row 1 stays as date headers. We mark holidays implicitly through gray fill on each task row.

    # Step 5: update Overview phases
    ov = wb["Overview"]
    # Phase IDs we touch: Phase 2 (row 4), Phase 5 (row 7), Phase 6 (row 8)
    ov.cell(row=4, column=2).value = "6/1 - 6/29"
    ov.cell(row=7, column=2).value = "8/17 - 9/28"
    ov.cell(row=8, column=2).value = "10/8 - 10/22"
    print("Overview Phase 2/5/6 dates updated")

    # Step 6: Daily Plan adjustments
    dp = wb["Daily Plan"]

    def set_date(row, d: date):
        dp.cell(row=row, column=1).value = datetime(d.year, d.month, d.day)

    # MLE-2: row 34 (6/19 端午) → 6/29
    set_date(34, date(2026, 6, 29))
    print("Daily Plan: MLE-2 row 34 6/19 → 6/29 (端午 调整)")

    # DOC-4: row 99 (9/25 中秋) → 9/28
    set_date(99, date(2026, 9, 25))  # noop test
    set_date(99, date(2026, 9, 28))
    print("Daily Plan: DOC-4 row 99 9/25 → 9/28 (中秋 调整)")

    # DOC-1: 7 rows currently 10/1, 10/2, 10/5, 10/6, 10/7, 10/8, 10/9
    # → 10/8, 10/9, 10/12, 10/13, 10/14, 10/15, 10/16 (skip 国庆)
    doc1_rows = [100, 101, 102, 103, 104, 105, 106]
    new_doc1 = [date(2026, 10, 8), date(2026, 10, 9), date(2026, 10, 12),
                date(2026, 10, 13), date(2026, 10, 14), date(2026, 10, 15),
                date(2026, 10, 16)]
    for r, d in zip(doc1_rows, new_doc1):
        if dp.cell(row=r, column=2).value != "DOC-1":
            print(f"WARN: row {r} not DOC-1, skipping")
            continue
        set_date(r, d)
    print(f"Daily Plan: DOC-1 7 rows shifted to 10/8-10/16 (skip 国庆)")

    wb.save(XLSX)
    print(f"Saved {XLSX}")


if __name__ == "__main__":
    main()
