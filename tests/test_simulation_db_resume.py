import unittest

import pandas as pd

from src.core.main_integration.simulation_db import (
    _load_m1_previous_orders_from_orderlog,
)


class _FakeDb:
    def __init__(self, rows: pd.DataFrame):
        self.rows = rows
        self.query = None
        self.params = None

    def execute_query_df(self, query, params=None):
        self.query = query
        self.params = params
        return self.rows.copy()


class SimulationDbResumeTest(unittest.TestCase):
    def test_load_m1_previous_orders_preserves_orderlog_columns(self):
        rows = pd.DataFrame({
            'run_id': ['run-1'],
            'sim_date': [pd.Timestamp('2026-05-11')],
            'date': [pd.Timestamp('2026-05-12')],
            'material': ['MAT-1'],
            'location': ['LOC-1'],
            'demand_type': ['normal'],
            'quantity': [10],
            'simulation_date': [pd.Timestamp('2026-05-11')],
            'advance_days': [0],
            'config_name': ['cfg'],
            'db_write_time': [pd.Timestamp('2026-05-12 17:00:00')],
        })
        db = _FakeDb(rows)

        result = _load_m1_previous_orders_from_orderlog(
            db,
            run_id='run-1',
            previous_batch_end='2026-05-11',
        )

        self.assertEqual(
            list(result.columns),
            [
                'date',
                'material',
                'location',
                'demand_type',
                'quantity',
                'simulation_date',
                'advance_days',
            ],
        )
        self.assertEqual(result.at[0, 'material'], 'MAT-1')
        self.assertEqual(result.at[0, 'quantity'], 10)
        self.assertEqual(result.at[0, 'date'], pd.Timestamp('2026-05-12'))
        self.assertEqual(db.params, ('run-1', '2026-05-11'))

