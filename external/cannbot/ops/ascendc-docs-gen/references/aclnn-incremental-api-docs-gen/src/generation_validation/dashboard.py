# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Parallel execution dashboard - Live table for concurrent doc_part processing."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from . import config


@dataclass
class TaskStatus:
    step: str = "等待中"
    progress_current: int = 0
    progress_total: int = 0
    status: str = "⏳"
    result: str = ""


class ParallelDashboard:
    """Live-updating table showing parallel task progress."""

    def __init__(self, doc_parts: list[str], title: str = ""):
        self.doc_parts = list(doc_parts)
        self.title = title
        self.status_map: dict[str, TaskStatus] = {dp: TaskStatus() for dp in doc_parts}
        self._lock = threading.Lock()
        self._logs: dict[str, list[str]] = {dp: [] for dp in doc_parts}
        self._live: Optional[Live] = None
        self._console = Console()

    def update(self, doc_part: str, **kwargs):
        with self._lock:
            if doc_part not in self.status_map:
                return
            for k, v in kwargs.items():
                if hasattr(self.status_map[doc_part], k):
                    setattr(self.status_map[doc_part], k, v)
            if self._live is not None:
                self._live.update(self.make_table())

    def append_log(self, doc_part: str, msg: str):
        with self._lock:
            if doc_part not in self._logs:
                self._logs[doc_part] = []
            self._logs[doc_part].append(msg)

    def make_table(self) -> Table:
        table = Table(
            title=self.title, show_header=True, header_style="bold cyan", expand=True
        )
        table.add_column("文档模块", style="cyan", min_width=20)
        table.add_column("步骤", min_width=18)
        table.add_column("进度", min_width=20)
        table.add_column("状态", justify="center", min_width=6)
        table.add_column("结果", justify="center", min_width=8)

        for dp in self.doc_parts:
            s = self.status_map.get(dp, TaskStatus())
            status_style = {
                "✓": "bold green",
                "⚠": "bold yellow",
                "✗": "bold red",
            }.get(s.status, "dim")
            result_style = {
                "PASS": "bold green",
                "FAIL": "bold red",
                "ERROR": "bold red",
            }.get(s.result, "dim")

            if s.progress_total > 0:
                bar = ProgressBar(
                    total=s.progress_total,
                    completed=s.progress_current,
                    width=10,
                    complete_style="green",
                    finished_style="bold green",
                )
                progress_cell = Group(
                    bar, Text(f" {s.progress_current}/{s.progress_total}", style="dim")
                )
            else:
                progress_cell = Text("")

            table.add_row(
                config.format_doc_part(dp),
                s.step,
                progress_cell,
                Text(s.status, style=status_style),
                Text(s.result, style=result_style),
            )
        return table

    @contextmanager
    def live_context(self):
        self._live = Live(
            self.make_table(),
            console=self._console,
            refresh_per_second=4,
            transient=False,
        )
        try:
            with self._live:
                yield self
        finally:
            self._live = None


_current_dashboard: ContextVar[Optional[ParallelDashboard]] = ContextVar(
    "_current_dashboard", default=None
)
_current_doc_part: ContextVar[str] = ContextVar("_current_doc_part", default="")

_console_fallback = Console()


def set_dashboard_context(dashboard: ParallelDashboard, doc_part: str) -> list[Token]:
    tokens = []
    tokens.append(_current_dashboard.set(dashboard))
    tokens.append(_current_doc_part.set(doc_part))
    return tokens


def get_dashboard() -> Optional[ParallelDashboard]:
    return _current_dashboard.get()


def get_current_doc_part() -> str:
    return _current_doc_part.get()


def dashboard_step(step: str):
    db = get_dashboard()
    dp = get_current_doc_part()
    if db and dp:
        db.update(dp, step=step)


def dashboard_progress(current: int, total: int):
    db = get_dashboard()
    dp = get_current_doc_part()
    if db and dp:
        db.update(dp, progress_current=current, progress_total=total)


def dashboard_done():
    db = get_dashboard()
    dp = get_current_doc_part()
    if db and dp:
        db.update(dp, status="✓")


def dashboard_result(passed: bool | None):
    db = get_dashboard()
    dp = get_current_doc_part()
    if db and dp:
        if passed is None:
            db.update(dp, status="✗", result="ERROR")
        elif passed:
            db.update(dp, status="✓", result="PASS")
        else:
            db.update(dp, status="✓", result="FAIL")


def dashboard_warn(msg: str = ""):
    db = get_dashboard()
    dp = get_current_doc_part()
    if db and dp:
        db.update(dp, status="⚠")
        if msg:
            db.append_log(dp, f"[yellow]{msg}[/yellow]")


def dashboard_error(msg: str = ""):
    db = get_dashboard()
    dp = get_current_doc_part()
    if db and dp:
        db.update(dp, status="✗")
        if msg:
            db.append_log(dp, f"[red]{msg}[/red]")


def dashboard_log(msg: str):
    db = get_dashboard()
    dp = get_current_doc_part()
    if db and dp:
        db.append_log(dp, msg)
    else:
        _console_fallback.print(msg)
