import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Orchestrator:
    """调度总管（协调器）：装配 ConfigManager + PersistenceManager，负责调度与分发。

    职责边界：
    - 身份：config_name / run_id
    - 调度：日期迭代、模块执行顺序、输出目录
    - 配置分发：按 module.schema 把 all_config 分发给模块
    - 委托：配置生命周期交给 ``ConfigManager``，状态持久化交给 ``PersistenceManager``
    - 不管：可变状态持有（交给 StateContext）、业务计算（交给 Module）、
      任何 DB/Excel/hash/DQ 直连 IO（交给两个协作者）
    """

    def __init__(self, start_date, end_date,
                 config_path=None, output_path=None,
                 config_dict=None, engine='pandas', skip_dq=False):
        self.start_date = start_date if isinstance(start_date, date) else pd.Timestamp(start_date).date()
        self.end_date = end_date if isinstance(end_date, date) else pd.Timestamp(end_date).date()
        self.module_idx = [1, 3, 4, 5, 6]
        self.engine = engine
        self.output_path = output_path or './output'

        # ── 身份 ──
        self._config_name = None
        self._run_id = None

        # ── 协作者 ──
        from .config_manager import ConfigManager
        from .persistence_manager import PersistenceManager
        self.config = ConfigManager(self)
        self.persistence = PersistenceManager(self)

        # ── 配置加载 + 持久化（委托 ConfigManager） ──
        self.config.bootstrap(config_path=config_path)

        # ── 续跑检测（在 load 之前，以便复用 run_id + 短路 DQ）──
        # config_name 预置（DB 模式可由调用方显式赋值；config_path 模式取 stem）
        if self._config_name is None and config_path is not None:
            self._config_name = Path(config_path).stem
        self._resume_date = None      # 断点日期（progress_date）：iter_dates 从此日起重跑
        self._restore_date = None     # 上一完成周期（progress_date - 1）：ctx/m1 快照从此日恢复
        self._resuming = False
        self._total_days = None
        if self.db is not None and self._config_name:
            pending = self.persistence.find_unfinished(self._config_name)
            if pending is not None:
                self._run_id = pending['run_id']            # 复用，绕过惰性生成
                self._resume_date = pending['current_date']  # 断点日期
                self._total_days = pending.get('total_days')
                # progress_date 在每日 day_end(save_daily_state) 与每日迭代起点(save_checkpoint)
                # 都会写，故中断时 progress_date=被中断当天，而该天的 viewcontext/快照尚未落库。
                # 续跑恢复 ctx/m1 应读上一完成周期 = progress_date - 1。
                if self._resume_date is not None:
                    self._restore_date = (
                        pd.Timestamp(self._resume_date) - pd.Timedelta(days=1)
                    ).strftime('%Y-%m-%d')
                self._resuming = True
                logger.info(
                    f"🔁 检测到未完成运行 {pending['run_id']}，"
                    f"从 {self._resume_date} 续跑（恢复快照取 {self._restore_date}）"
                )

        try:
            dq_result, needs_write = self.config.load(
                config_path, config_dict, skip_dq=skip_dq,
            )
            if needs_write:
                self.config.persist(dq_result, write_config=True)
        except Exception:
            # DQ 阻断：仍持久化检测结果（不含配置数据），供调试排查
            if self.config.last_dq_result is not None and self.db is not None:
                self.config.persist(self.config.last_dq_result, write_config=False)
            raise SystemError('数据质量检测阻断')

        # ── 随机种子 ──
        from ..main_integration.seed import set_module_seeds
        set_module_seeds(self.all_config)

        # ── 输出目录 ──
        self.all_results = {}
        self.sim_dates = []

    # ══════════════════════════════════════════
    # 透传属性（真实数据归 ConfigManager 持有，保持对外契约不变）
    # ══════════════════════════════════════════

    @property
    def db(self):
        return self.config.db

    @property
    def sys_config(self) -> dict:
        return self.config.sys_config

    @sys_config.setter
    def sys_config(self, value: dict):
        self.config.sys_config = value

    @property
    def all_config(self) -> dict:
        return self.config.all_config

    @all_config.setter
    def all_config(self, value: dict):
        self.config.all_config = value

    # ══════════════════════════════════════════
    # 身份：config_name / run_id
    # ══════════════════════════════════════════

    @property
    def config_name(self) -> str | None:
        """配置名称，默认取配置文件名（不含扩展名）。

        从 config_path 加载时自动赋值为文件名 stem；
        DB 模式下可由调用方显式赋值（orch.config_name = 'BC_S5'）。
        """
        return self._config_name

    @config_name.setter
    def config_name(self, value: str):
        self._config_name = value

    @property
    def run_id(self) -> str:
        """运行唯一标识，格式: {config_name}_{YYYYMMDD_HHMMSS}。

        首次访问时自动生成，后续访问返回同一值。
        若 config_name 未设置则回退为 'unknown'。
        """
        if self._run_id is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = self.config_name or 'unknown'
            self._run_id = f"{name}_{ts}"
            logger.info(f"run_id 已生成: {self._run_id}")
        return self._run_id

    @property
    def total_days(self) -> int:
        """仿真总天数（iter_dates 分母）。

        续跑时优先用 orch_run_event 记录的值（首次运行写入）；
        否则按 start_date→end_date 实时计算。
        """
        if self._total_days:
            return self._total_days
        return len(pd.date_range(self.start_date, self.end_date, freq='D'))

    def finish(self):
        """整个流程真正结束后调用：标 status='finished' + 写 finished_at。"""
        self.persistence.mark_finished(self.run_id)

    # ══════════════════════════════════════════
    # 模块配置分发
    # ══════════════════════════════════════════

    def load_datas(self, module) -> None:
        """按 module.schema 从 all_config 加载数据到 module.datas。"""
        from ...modules.module import Module
        if not isinstance(module, Module):
            raise TypeError(f"load_datas 期望 Module 实例，收到 {type(module).__name__}")
        schema = getattr(module, 'schema', {})
        if not schema:
            return
        datas = {}
        for sheet_name, col_schema in schema.items():
            df = self.all_config.get(sheet_name, pd.DataFrame())
            datas[sheet_name] = df
        module.datas = datas

    def get_module_config(self, module_name: str) -> dict:
        """从 sys_config 提取模块参数（shared + module 专属节）。"""
        if not self.sys_config:
            return {}
        result = self.sys_config.get('shared', {})
        result.update(self.sys_config.get(module_name, {}))
        return result

    # ══════════════════════════════════════════
    # 调度
    # ══════════════════════════════════════════

    def iter_dates(self, actual_start_date=None, resume_date=None):
        """日期迭代器。

        actual_start_date : 重新设定仿真起点（保留原义，total 随之变化 → 1/96 语义）。
        resume_date       : 续跑断点 —— total 仍按完整范围（首次记录的 total_days），
                            序号从断点位置开始，进度显示 (n/100)。
        两者不可同时指定。orch 续跑检测设过 _resume_date 时，resume_date 可省略自动复用。
        """
        if actual_start_date is not None and resume_date is not None:
            raise ValueError("actual_start_date 与 resume_date 不可同时指定")
        if resume_date is None:
            resume_date = self._resume_date

        full_dates = pd.date_range(self.start_date, self.end_date, freq='D')
        # total_days：续跑用表记录值（首次写入），否则实时算
        total = self._total_days or len(full_dates)

        if resume_date is not None:
            # ── 续跑：分母=完整范围，序号=断点位置，initial=断点前已完天数 ──
            self.sim_dates = full_dates
            rts = pd.Timestamp(resume_date)
            mask = full_dates >= rts
            start_idx = int(mask.argmax()) if mask.any() else len(full_dates)
            iterate_dates = full_dates[start_idx:]
            start_count, initial_done = start_idx + 1, start_idx
            logger.info(
                f"续跑：从第 {start_count}/{total} 天 ({rts:%Y-%m-%d}) 恢复"
            )
        else:
            # ── 全新运行（actual_start_date 保留原义）──
            sim_start = actual_start_date or self.start_date
            iterate_dates = pd.date_range(sim_start, self.end_date, freq='D')
            self.sim_dates = iterate_dates
            start_count, initial_done = 1, 0
            logger.info(f"仿真日期范围: {len(iterate_dates)} 天 (total={total})")

        pbar = tqdm(enumerate(iterate_dates, start_count),
                    total=total, initial=initial_done,   # ← 显示 (5/100) 的关键
                    desc='仿真进度', unit='天', ncols=80, leave=True)
        for i, current_date in pbar:
            # 记录断点：每进入新一天即更新 current_date（status 保持 running）
            try:
                self.save_checkpoint(self.run_id, current_date=current_date)
            except Exception:
                logger.warning("current_date 断点记录失败", exc_info=True)
            progress_info = f"第 {i}/{total} 天"
            pbar.write(f"{'=' * 20} {progress_info}: {current_date.strftime('%Y-%m-%d')} {'=' * 20}")
            pbar.set_postfix(date=current_date.strftime('%Y-%m-%d'), day=progress_info)
            yield i, current_date

    def get_output(self, module_name: str) -> Path:
        """返回模块输出目录路径。"""
        if not hasattr(self, '_output_dirs') or module_name not in self._output_dirs:
            raise KeyError(f"未找到模块输出路径: {module_name}")
        return self._output_dirs[module_name]

    def build_output_folder(self):
        """创建输出目录。"""
        self._output_dirs = {}
        for i in self.module_idx:
            path = Path(self.output_path) / f'module{i}'
            path.mkdir(parents=True, exist_ok=True)
            self._output_dirs[f'module{i}'] = path

    # ══════════════════════════════════════════
    # 状态持久化（薄委托到 PersistenceManager）
    # ══════════════════════════════════════════

    def save_daily_state(self, ctx, date_str):
        """委托到 PersistenceManager 写 DB。"""
        self.persistence.save_daily_state(ctx, date_str)

    def save_module_output(self, module, sim_date: str):
        """委托到 PersistenceManager 写 DB。"""
        self.persistence.save_module_output(module, sim_date)

    def restore_state(self, ctx, run_id: str, sim_date: str):
        """委托到 PersistenceManager 从 ViewContext 表恢复状态。"""
        self.persistence.restore_state_from_views(ctx, run_id, sim_date)

    def save_checkpoint(self, run_id: str, status: str = 'running',
                        current_date=None):
        """委托到 PersistenceManager 更新 orch 执行态（写 orch_run_event）。"""
        self.persistence.save_checkpoint(run_id, status, current_date=current_date)

    def load_checkpoint(self, run_id: str) -> dict | None:
        """委托到 PersistenceManager 加载运行元信息。"""
        return self.persistence.load_checkpoint(run_id)
