"""Smoke test for M4 refactoring changes."""
import pandas as pd
import numpy as np

from src.modules.production_planning.integration_refactor import (
    ModuleFour, compute_planning_window, safe_float_conversion, dedup_issues,
)
from src.modules.state_context import StateContext


def test_m4_override_logic():
    # Empty overrides → empty states
    m4 = ModuleFour(
        simulation_date=pd.Timestamp('2025-12-16'),
        simulation_start_date=pd.Timestamp('2025-12-16'),
        previous_line_states_override={},
        allocated_capacity_override={},
    )
    m4._load_previous_states()
    assert m4.previous_line_states == {}
    assert m4.previously_allocated == {}

    # Override injection
    m4 = ModuleFour(
        simulation_date=pd.Timestamp('2025-12-17'),
        simulation_start_date=pd.Timestamp('2025-12-16'),
        previous_line_states_override={'LINE_A': {'last_material': 'MAT_X', 'last_activity': 'production', 'changeover_info': None}},
        allocated_capacity_override={'LOC|LINE_A|2025-12-17': 5.0},
    )
    m4._load_previous_states()
    assert m4.previous_line_states == {'LINE_A': {'last_material': 'MAT_X', 'last_activity': 'production', 'changeover_info': None}}
    assert m4.previously_allocated == {'LOC|LINE_A|2025-12-17': 5.0}


def test_state_context_m4_methods():
    ctx = StateContext(simulation_date='2025-12-16')
    ctx.m4_line_states = {'2025-12-16': {'LINE_A': {'last_material': 'MAT_X', 'last_activity': 'production', 'changeover_info': None}}}
    ctx.m4_allocated_capacity = {'2025-12-16': {'LOC|LINE_A|2025-12-16': 5.0}}

    assert ctx.get_previous_line_state('2025-12-17') == {'LINE_A': {'last_material': 'MAT_X', 'last_activity': 'production', 'changeover_info': None}}
    assert ctx.get_all_previous_allocated_capacity('2025-12-17') == {'LOC|LINE_A|2025-12-16': 5.0}

    ctx.apply_line_state({'LINE_B': {'last_material': 'MAT_Y', 'last_activity': 'changeover', 'changeover_info': {'remaining_time': 2.0, 'changeover_id': 'CO1', 'to_material': 'MAT_Z'}}}, '2025-12-17')
    assert '2025-12-17' in ctx.m4_line_states

    ctx.apply_allocated_capacity({'LOC|LINE_B|2025-12-18': 3.0}, '2025-12-17')
    assert '2025-12-17' in ctx.m4_allocated_capacity


def test_inline_functions():
    ws, we = compute_planning_window(pd.Timestamp('2025-12-16'), 7, 14)
    assert ws == pd.Timestamp('2025-12-23')
    assert we == pd.Timestamp('2026-01-05')  # 2025-12-16 + (7+14-1) = +20 days

    assert safe_float_conversion(np.int64(5)) == 5.0
    assert safe_float_conversion(0) == 0.0

    assert dedup_issues([{'a': 1}, {'a': 1}, {'a': 2}]) == [{'a': 1}, {'a': 2}]


def test_no_removed_module_imports():
    """Verify ModuleFour no longer imports from removed modules."""
    import ast, inspect
    src = inspect.getsource(ModuleFour)
    tree = ast.parse(src)
    removed_modules = ['state_manager', 'capacity_allocator', 'plan_builder', 'constants']
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            for x in removed_modules:
                assert x not in mod, f'Still imports from {mod}'
