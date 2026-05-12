import contextlib
import io
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.modules.demand_planning.integration import _merge_with_history


class MergeWithHistoryDiagnosticsTest(unittest.TestCase):
    def test_reports_non_finite_quantity_rows_before_int_cast(self):
        previous_orders = pd.DataFrame({
            'date': [pd.Timestamp('2026-01-01')],
            'material': ['MAT-1'],
            'location': ['LOC-1'],
            'demand_type': ['AO'],
            'quantity': [np.nan],
            'simulation_date': [pd.Timestamp('2026-01-01')],
            'advance_days': [0],
        })

        today_orders = pd.DataFrame(columns=previous_orders.columns)
        stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(
                ValueError,
                'quantity contains non-finite or non-numeric values',
            ):
                with contextlib.redirect_stderr(stderr):
                    _merge_with_history(
                        output_dir=output_dir,
                        simulation_date=pd.Timestamp('2026-01-01'),
                        today_orders_df=today_orders,
                        ao_config=pd.DataFrame(),
                        previous_orders_df=previous_orders,
                    )

        diagnostic = stderr.getvalue()
        self.assertIn('src/modules/demand_planning/integration.py', diagnostic)
        self.assertIn('previous_orders_df', diagnostic)
        self.assertIn('MAT-1', diagnostic)
        self.assertIn('LOC-1', diagnostic)
        self.assertIn('NaN', diagnostic)

