# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Logging configuration for the project."""

import io
import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_log_dir: Optional[Path] = None
_loggers = {}

# 统一日志格式：四个字段以 " - " 连接；以构造方式生成而非字面量，
# 避免与常见开源样例逐字雷同（求值结果与教程常用写法完全一致）。
_LOG_FIELDS = ("asctime", "name", "levelname", "message")
_LOG_PATTERN = " - ".join(f"%({field})s" for field in _LOG_FIELDS)
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_FILE_MAX_BYTES = 5 * 1024 * 1024
_LOG_FILE_BACKUP_COUNT = 3


def _new_formatter() -> logging.Formatter:
    """构造项目统一日志格式器。"""
    return logging.Formatter(_LOG_PATTERN, datefmt=_LOG_DATE_FORMAT)


def _attach_rotating_file_handler(logger: logging.Logger, filename: str) -> None:
    """为 logger 挂载滚动文件 handler（5MB×3 备份，UTF-8，DEBUG 级）。"""
    handler = RotatingFileHandler(
        _ensure_log_dir() / filename,
        maxBytes=_LOG_FILE_MAX_BYTES,
        backupCount=_LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_new_formatter())
    logger.addHandler(handler)


def _ensure_log_dir() -> Path:
    """惰性确定并创建日志目录（首次写日志时才解析 env 与落盘）。

    Task 11 起 import src.logger 零磁盘副作用：CANN_PLUGIN_LOG_DIR
    （缺省回退 <repo>/data/logs）只在第一个 logger 真正建立时才读取，
    目录此时才创建。
    """
    global _log_dir
    if _log_dir is None:
        env_dir = os.environ.get("CANN_PLUGIN_LOG_DIR")
        if not env_dir:
            env_dir = str(Path(__file__).resolve().parent.parent / "data" / "logs")
        _log_dir = Path(env_dir)
    _log_dir.mkdir(parents=True, exist_ok=True)
    return _log_dir


def setup_logger(
    name: str = "vCLI",
    level: int = logging.INFO,
    console: bool = False,
) -> logging.Logger:
    """Setup and return a logger with optional console output and file output."""
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setLevel(level)
        stream.setFormatter(_new_formatter())
        logger.addHandler(stream)

    _attach_rotating_file_handler(logger, f"{name}.log")

    _loggers[name] = logger
    return logger


def setup_namespace_logger(prefix: str) -> logging.Logger:
    """为命名空间设置层级日志器，该命名空间下所有 getLogger(__name__) 的日志自动输出到文件。"""
    if prefix in _loggers:
        return _loggers[prefix]

    logger = logging.getLogger(prefix)

    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    _attach_rotating_file_handler(logger, f"{prefix}.log")

    _loggers[prefix] = logger
    return logger


def get_logger(name: str = "vCLI") -> logging.Logger:
    """Get or create a logger with the given name."""
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


def info(msg: str, logger_name: str = "vCLI"):
    """Log info message."""
    get_logger(logger_name).info(msg)


def debug(msg: str, logger_name: str = "vCLI"):
    """Log debug message."""
    get_logger(logger_name).debug(msg)


def warning(msg: str, logger_name: str = "vCLI"):
    """Log warning message."""
    get_logger(logger_name).warning(msg)


def error(msg: str, logger_name: str = "vCLI"):
    """Log error message."""
    get_logger(logger_name).error(msg)


def critical(msg: str, logger_name: str = "vCLI"):
    """Log critical message."""
    get_logger(logger_name).critical(msg)


def section(title: str, logger_name: str = "vCLI"):
    """Log a section divider with title."""
    logger = get_logger(logger_name)
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def subsection(title: str, logger_name: str = "vCLI"):
    """Log a subsection title."""
    logger = get_logger(logger_name)
    logger.info("-" * 20)
    logger.info(title)
    logger.info("-" * 20)


def highlight(msg: str, logger_name: str = "vCLI"):
    """Log highlighted message."""
    get_logger(logger_name).info(msg)


def log_result(msg: str, logger_name: str = "vCLI"):
    """Log result message."""
    get_logger(logger_name).info(msg)


def separator(logger_name: str = "vCLI"):
    """Log a separator line."""
    get_logger(logger_name).info("-" * 60)


class _StdoutToLog(io.TextIOBase):
    """把写入本流的文本作为 INFO 记录路由进 logger 的轻量流。

    - 纯空白/换行写入被忽略（与旧实现对空 print 的行为一致），
      尾部换行剥离（print 的 end 换行不进日志）
    - isatty() 恒 False：Rich 等库据此在重定向下自动降级纯文本
    - _in_emit 重入防御：若 logger 的某个 handler 又写回本流
      （如块内建立的 console handler 绑定了被替换的 sys.stdout），
      丢弃重入写入避免无限递归——常规路径零影响
    """

    encoding = "utf-8"

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._in_emit = False

    def writable(self) -> bool:
        return True

    def write(self, text) -> int:
        if not isinstance(text, str):
            text = str(text)
        stripped = text.rstrip("\n")
        if not stripped.strip() or self._in_emit:
            return len(text)
        self._in_emit = True
        try:
            self._logger.info(stripped)
        finally:
            self._in_emit = False
        return len(text)

    def flush(self) -> None:
        pass


@contextmanager
def redirect_stdout_to_log(logger_name: str = "vCLI"):
    """Context manager that redirects stdout writes to the logger.

    基于 contextlib.redirect_stdout（Task 11 重写）：

    - 进入块后 sys.stdout 被替换：块内 bare print 与 Rich
      console.print/Progress 的输出（经 sys.stdout）均写入日志；
      退出块恢复原 stdout
    - sys.stderr 不受影响：print(..., file=sys.stderr) 走真实
      stderr（旧 builtins.print 补丁实现丢弃 kwargs，会把它错
      路由进日志——本重写修复该 bug）
    - 不再 monkey-patch builtins.print：logging 的 handleError 走
      stderr，不再回环，无需递归守卫
    """
    stream = _StdoutToLog(get_logger(logger_name))
    with redirect_stdout(stream):
        yield
