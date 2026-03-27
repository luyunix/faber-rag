"""可观测性日志工具。

提供：
- ``get_logger``: 标准的人类可读日志记录器（与 C 阶段保持一致）。
- ``JSONFormatter``: 自定义的 :class:`logging.Formatter`，输出 JSON 格式。
- ``get_trace_logger``: 返回一个使用 JSON Lines 文件处理程序的日志记录器。
- ``write_trace``: 便捷函数，用于将追踪字典追加到
  ``logs/traces.jsonl``。
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.settings import resolve_path

# 追踪文件的默认路径（绝对路径，不依赖于当前工作目录）
_DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


# ── 人类可读的日志记录器（已有）────────────────────────────────────────


def get_logger(name: str = "modular-rag", log_level: Optional[str] = None) -> logging.Logger:
    """获取配置好的日志记录器。

    参数：
        name: 日志记录器名称。
        log_level: 可选的日志级别字符串（例如 "INFO"）。

    返回：
        配置好的日志记录器实例。
    """

    if log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = logging.INFO

    # 配置根 logger（如果尚未配置）
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            stream=sys.stderr,
        )
    else:
        # 如果已经有 handler，确保级别正确
        logging.getLogger().setLevel(level)

    # 抑制 httpx 日志（包含敏感的端点 URL）
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logging.getLogger(name)


# ── JSON Lines 格式化器 ───────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    """输出每行一个 JSON 对象的日志格式化器。

    每条日志记录都被序列化为一个字典，至少包含：
    ``timestamp``、``level``、``logger``、``message``。如果记录
    携带 ``exc_info`` 元组，则堆栈跟踪会作为
    ``exception`` 包含在内。

    通过 logger 调用时附加的额外属性（*extra=*）会被
    合并到顶层字典中（除了内部 Python 字段）。
    """

    _INTERNAL_ATTRS = frozenset({
        "args", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module",
        "msecs", "message", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread",
        "threadName", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        """将日志记录返回为单行 JSON 字符串。"""
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 合并调用者附加的额外字段
        for key, val in record.__dict__.items():
            if key not in self._INTERNAL_ATTRS and key not in payload:
                try:
                    json.dumps(val)  # cheap serialisability test
                    payload[key] = val
                except (TypeError, ValueError):
                    payload[key] = str(val)

        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


# ── 追踪日志记录器 ─────────────────────────────────────────────────


def get_trace_logger(
    traces_path: str | Path = _DEFAULT_TRACES_PATH,
    *,
    name: str = "modular-rag.trace",
) -> logging.Logger:
    """返回一个将 JSON Lines 写入 *traces_path* 的日志记录器。

    该日志记录器使用 :class:`JSONFormatter` 和一个配置为追加模式的
    :class:`FileHandler`。使用相同的 *name* 重复调用会返回
    相同的日志记录器（标准的 :mod:`logging` 语义）。

    参数：
        traces_path: JSONL 输出的文件路径。父目录
            会自动创建。
        name: 日志记录器名称。

    返回：
        准备好输出 JSON Lines 的 :class:`logging.Logger`。
    """
    path = Path(traces_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免在重复调用时添加重复的处理程序
    if not logger.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False  # don't echo to console

    return logger


# ── 追踪字典的便捷写入器 ────────────────────────────────────────────


def write_trace(
    trace_dict: Dict[str, Any],
    traces_path: str | Path = _DEFAULT_TRACES_PATH,
) -> None:
    """将单个追踪字典作为一行 JSON 追加。

    这是一个直接写入的轻量级包装器 — 不涉及日志记录
    框架 — 因此输出与
    :class:`~src.core.trace.trace_collector.TraceCollector` 产生的输出相同。

    参数：
        trace_dict: 一个 JSON 可序列化的字典（通常来自
            ``TraceContext.to_dict()``）。
        traces_path: 输出文件路径；父目录会自动
            创建。
    """
    path = Path(traces_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(trace_dict, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")