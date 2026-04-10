"""Shared runtime defaults with stable values.

These values are intentionally tiny and conservative so existing module-level
constants can re-export them without changing runtime behavior.
"""

# Shared MOQ / RV defaults used by M3 and M5.
DEFAULT_MOQ: int = 1
DEFAULT_RV: int = 1

# Shared planning-window defaults used by M3 and M5 lead-time helpers.
DEFAULT_PTF: int = 0
DEFAULT_LSK: int = 1
DEFAULT_HORIZON: int = 1
DEFAULT_LEAD_TIME: int = 1
