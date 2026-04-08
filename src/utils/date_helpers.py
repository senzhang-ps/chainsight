"""Shared date and lead-time helpers with opt-in wrappers.

These helpers intentionally keep tiny, deterministic behavior so module-level
wrappers can reuse them without changing business semantics.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd


def compute_planning_window(
    simulation_date: pd.Timestamp,
    ptf: int,
    lsk: int,
):
    """Return the inclusive planning window start/end pair."""
    window_start = simulation_date + timedelta(days=ptf)
    window_end = simulation_date + timedelta(days=ptf + lsk - 1)
    return window_start, window_end


def is_offset_review_day(
    simulation_date: pd.Timestamp,
    simulation_start: pd.Timestamp,
    lsk: int,
    day: int,
) -> bool:
    """Return whether a review day falls on an offset-based cycle."""
    days_since_start = (simulation_date - simulation_start).days
    first_review_day = int(day) - 1
    is_on_cycle = (days_since_start - first_review_day) % int(lsk) == 0
    is_after_first = days_since_start >= first_review_day
    return is_on_cycle and is_after_first


def is_calendar_review_day(dt: pd.Timestamp, lsk: str, day: int) -> bool:
    """Return whether a date is a daily/weekly/monthly review day."""
    if lsk == "daily":
        return True
    if lsk == "weekly":
        return dt.weekday() == (int(day) - 1)
    if lsk == "monthly":
        return dt.day == int(day)
    raise ValueError(f"Unknown LSK: {lsk}")


def calculate_transport_lead_time(
    *,
    pdt: int,
    gr: int,
    mct: int,
    location_type: str,
    ptf: int = 0,
    lsk: int = 1,
    minimum: int = 0,
) -> int:
    """Calculate Plant/DC lead time while preserving caller-specific minima."""
    if str(location_type).lower() == "plant":
        base_lt = max(mct, pdt + gr)
        leadtime = base_lt + ptf + lsk - 1
    else:
        leadtime = pdt + gr
    return max(int(minimum), int(leadtime))
