import unittest
from contextlib import contextmanager

import pandas as pd

from pgsql_db.db_connection import DatabaseConnection


class _FakeCursor:
    def __init__(self):
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchall(self):
        return [('date', 'date')]


class _FakeDb(DatabaseConnection):
    def __init__(self):
        self.cursor = _FakeCursor()

    @contextmanager
    def get_cursor(self, commit=True):
        yield self.cursor


class DatabaseColumnTypeTest(unittest.TestCase):
    def test_object_date_column_uses_text_for_all_wildcard_values(self):
        db = DatabaseConnection.__new__(DatabaseConnection)

        pg_type = db._pandas_to_pg_type(
            pd.Series(['ALL']).dtype,
            col_name='date',
        )

        self.assertEqual(pg_type, 'TEXT')

    def test_datetime_date_column_still_uses_date(self):
        db = DatabaseConnection.__new__(DatabaseConnection)

        pg_type = db._pandas_to_pg_type(
            pd.Series([pd.Timestamp('2026-05-12')]).dtype,
            col_name='date',
        )

        self.assertEqual(pg_type, 'DATE')

    def test_existing_date_column_is_upgraded_to_text_for_object_date_input(self):
        db = _FakeDb()
        df = pd.DataFrame({'date': ['ALL']})

        self.assertTrue(db._check_table_compatible(
            'cfg_m6_deliverydelaydistribution',
            df,
        ))

        alter_queries = [
            query for query, _ in db.cursor.executed
            if query.__class__.__name__ == 'Composed'
        ]
        self.assertEqual(len(alter_queries), 1)

