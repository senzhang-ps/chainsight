import logging
import time
from abc import ABC, ABCMeta
from functools import wraps

import numpy as np
import pandas as pd

_logger = logging.getLogger("SupplyChainSimulation")


def make_timed(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = func(self, *args, **kwargs)
        elapsed = time.time() - start
        self.spends[func.__name__] = elapsed
        if getattr(self, 'verbose', False):
            _logger.warning(
                "[%s] %s took %.4fs", self.__class__.__name__, func.__name__, elapsed,
            )
        return result
    return wrapper


class TimedMeta(ABCMeta):
    def __new__(mcs, name, bases, namespace, **kwargs):
        exclude = set()
        for base in bases:
            if hasattr(base, '_timed_exclude'):
                exclude |= base._timed_exclude
        if '_timed_exclude' in namespace:
            exclude |= namespace['_timed_exclude']

        for key, value in list(namespace.items()):
            if (
                callable(value)
                and not key.startswith('_')
                and not isinstance(value, (staticmethod, classmethod))
                and not getattr(value, '__isabstractmethod__', False)
                and key not in exclude
            ):
                namespace[key] = make_timed(value)
        return super().__new__(mcs, name, bases, namespace)


class Module(ABC, metaclass=TimedMeta):
    _timed_exclude = {'run', 'output', 'validate_data'}

    def __init__(self, simulation_date, orchestrator, module_config, verbose=False):
        self.simulation_date = simulation_date
        self.orchestrator = orchestrator
        self.verbose = verbose
        self.params = orchestrator.load_params(module_config)
        self.datas = orchestrator.load_datas(module_config)
        self.spends = {}
        self._result = {}

    def run(self):
        raise NotImplementedError

    def output(self):
        return self._result

    def validate_data(
        self,
        df: pd.DataFrame,
        name: str = 'data',
        numeric_columns=None,
        required_columns=None,
        strict: bool = False,
    ) -> pd.DataFrame:
        """通用 DataFrame 数值校验：检测并修复 NaN/inf/不可转换值。"""
        from ..utils.numeric_safe import safe_int_series

        if df.empty:
            return df

        if required_columns:
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                raise ValueError(f"validate_data({name}): 缺失必选列 {missing}")

        if numeric_columns is not None:
            cols_to_check = [c for c in numeric_columns if c in df.columns]
        else:
            cols_to_check = list(df.select_dtypes(include='number').columns)

        if not cols_to_check:
            return df

        issues = {}
        for col in cols_to_check:
            series = df[col]
            numeric = pd.to_numeric(series, errors='coerce')
            bad_count = int((numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype='float64'))).sum())
            if bad_count > 0:
                if strict:
                    raise ValueError(
                        f"validate_data({name}): 列 '{col}' 含 {bad_count} 个异常值（NaN/inf/不可转换）"
                    )
                df[col] = safe_int_series(series, context=f'{name}.{col}')
                issues[col] = bad_count

        if issues:
            _logger.warning(
                "validate_data(%s): %d 列存在异常值 %s", name, len(issues), issues
            )

        return df
