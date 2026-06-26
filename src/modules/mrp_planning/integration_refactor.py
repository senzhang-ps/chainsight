"""Module3 集成模式主入口 — ModuleThree 类。

M3 净需求模块的面向对象封装（Module 子类）。当前为**历史回放占位**：
从 DB 表 ``module3_output_netdemand`` 按当日 ``sim_date`` 直读净需求，作为当日 M3 输出。

与 M4 的衔接遵循旧链路语义（1 天 lag）：

    循环内顺序：M4.run() → M3.run() → m4.module3_result = m3.output()
    即 M4 产出日 D 消费的是 (D-1) 日产出的 M3（M3 每日末尾产出、次日 M4 消费），
    首日无 M3 → M4 当日无需求、产出为空。

未来接真实 MRP 时，只需替换 ``run()`` 内部的取数逻辑（``_read_net_demand``），
对外契约（``output()`` → ``{'net_demand_df': df, ...}``）保持不变。
"""

import logging
import traceback
from typing import Optional

import pandas as pd

from ..module import Module

logger = logging.getLogger("SupplyChainSimulation")


class ModuleThree(Module):
    """M3 净需求模块（历史回放占位）：从 DB 读 module3_output_netdemand 作为当日净需求。

    Parameters
    ----------
    simulation_date : 仿真日期（每日循环里由驱动方覆写）。
    simulation_start_date : 仿真起始日期（占位，当前回放逻辑按当日 sim_date 直读）。
    orchestrator / orch : Orch 实例，提供 ``orch.db`` / ``orch.run_id``。
    m3_run_id : 用于筛选历史 M3 数据的 run_id。当 DB 中已有旧仿真落库的
        module3_output_netdemand 时，传入旧 run_id 即可读取（无需重新种子）。
        若未传则回退到 ``orch.run_id``。后续接真实 MRP 后可废弃此参数。
    """

    # DB 表名（input schema 下）；写入/读取均按 run_id + sim_date 索引
    M3_TABLE = 'module3_output_netdemand'
    # 落库时附加的元数据列，读取后剥离（业务侧不应见到）
    META_COLS = ('run_id', 'sim_date', 'config_name', 'db_write_time')

    def __init__(self, simulation_date, simulation_start_date, output_dir='',
                 orchestrator=None, orch=None, skip_file_output=False,
                 m3_run_id=None,
                 verbose=False, config=None):
        super().__init__(simulation_date, orch, 'M3', verbose, config=config)
        self.legacy_orchestrator = orchestrator
        self.output_dir = output_dir
        self.skip_file_output = skip_file_output
        self.simulation_start_date = simulation_start_date
        self.m3_run_id = m3_run_id

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def prepare(self):
        """一次性准备：本模块不依赖 cfg_* 配置表，DB 由 orch.db 持有，故无操作。

        （与 M1/M4 不同：M3 净需求的真实源是输出表而非输入配置表，
        load_datas 无法读取，故 prepare 留空，取数在 run() 内按当日进行。）
        """
        pass

    def _read_net_demand(self) -> pd.DataFrame:
        """从 DB 读当日净需求（module3_output_netdemand，按 run_id + 当日 sim_date）。

        run_id 优先使用 ``self.m3_run_id``（构造入参），未传时回退 ``orchestrator.run_id``。
        无 DB / 无表 / 无数据 / 读异常 → 返回空 DataFrame（不抛异常）。
        剥离 META_COLS（run_id/sim_date/config_name/db_write_time）。
        """
        db = getattr(self.orchestrator, 'db', None) if self.orchestrator is not None else None
        if db is None:
            return pd.DataFrame()

        # run_id：入参优先，回退 orchestrator.run_id
        query_run_id = self.m3_run_id or getattr(self.orchestrator, 'run_id', None)
        if query_run_id is None:
            return pd.DataFrame()

        sim_date = pd.Timestamp(self.simulation_date).strftime('%Y-%m-%d')
        try:
            if not db.table_exists(self.M3_TABLE):
                return pd.DataFrame()
            df = db.read(self.M3_TABLE, run_id=query_run_id, sim_date=sim_date)
            if df is None or df.empty:
                return pd.DataFrame()
            return df.drop(
                columns=[c for c in self.META_COLS if c in df.columns],
                errors='ignore',
            )
        except Exception:
            traceback.print_exc()
            return pd.DataFrame()

    @staticmethod
    def _normalize(net: pd.DataFrame) -> pd.DataFrame:
        """类型归一：material/location 转 str，requirement_date 转 datetime。

        注：``layer==0`` 过滤与 ``abs(quantity)`` 是 M4 的业务解读，仍由
        ``ModuleFour.load_net_demand`` 负责，这里不重复。
        """
        if net.empty:
            return net
        if 'material' in net.columns:
            net['material'] = net['material'].astype(str)
        if 'location' in net.columns:
            net['location'] = net['location'].astype(str)
        if 'requirement_date' in net.columns:
            net['requirement_date'] = pd.to_datetime(net['requirement_date'])
        return net

    def run(self):
        """逐日：读当日净需求 → 类型归一 → 组装 _result。"""
        net = self._normalize(self._read_net_demand())
        self._result = {
            'net_demand_df': net,
            'net_demand_count': len(net),
        }

    def output(self):
        """返回 run() 产出的结果字典（含 net_demand_df）。"""
        return self._result
