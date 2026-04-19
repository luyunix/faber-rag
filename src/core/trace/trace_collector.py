"""追踪收集器 – 接收已完成的 TraceContext 并持久化它们。

收集器是内存中 TraceContext 对象与仪表板使用的
磁盘 JSON Lines 日志之间的桥梁。它有意与
日志模块解耦，以便追踪持久化保持可预测和可测试。
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.core.settings import resolve_path
from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

# 追踪文件的默认绝对路径（与 CWD 无关）
_DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


class TraceCollector:
    """收集已完成的追踪并将它们追加到 JSON Lines 文件。

    参数：
        traces_path: ``traces.jsonl`` 输出的文件路径。
            父目录会自动创建。
    """

    def __init__(self, traces_path: str | Path = _DEFAULT_TRACES_PATH) -> None:
        self._path = Path(traces_path)
        # 确保目录存在
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def collect(self, trace: TraceContext) -> None:
        """将单个追踪作为一行 JSON 持久化。

        如果追踪尚未完成，会自动调用 ``finish()``，
        因此输出始终包含时间数据。

        参数：
            trace: 一个已填充的 :class:`TraceContext`。
        """
        try:
            if trace.finished_at is None:
                trace.finish()

            line = json.dumps(trace.to_dict(), ensure_ascii=False)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            logger.exception("写入追踪 %s 失败", trace.trace_id)

    @property
    def path(self) -> Path:
        """返回追踪文件的解析路径。"""
        return self._path
