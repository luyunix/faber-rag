"""TraceService – 从 logs/traces.jsonl 读取和解析追踪记录。

为原始 JSONL 追踪日志提供类型化、可过滤的接口。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import resolve_path

logger = logging.getLogger(__name__)

# 追踪文件的默认路径（绝对路径，不依赖于当前工作目录）
DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


class TraceService:
    """用于查询已记录追踪的只读服务。

    参数：
        traces_path: JSONL 文件的路径。默认为
            ``logs/traces.jsonl``。
    """

    def __init__(self, traces_path: Optional[str | Path] = None) -> None:
        self.traces_path = Path(traces_path) if traces_path else DEFAULT_TRACES_PATH

    # ------------------------------------------------------------------
    # 公共 API    # ------------------------------------------------------------------

    def list_traces(
        self,
        trace_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """按时间倒序返回追踪记录。

        参数：
            trace_type: 按 ``trace_type`` 字段过滤（例如
                ``"ingestion"`` 或 ``"query"``）。``None`` = 全部。
            limit: 返回的最大追踪记录数。

        返回：
            追踪字典列表（最新的在前）。
        """
        traces = self._load_all()

        if trace_type:
            traces = [t for t in traces if t.get("trace_type") == trace_type]

        # 最新的在前
        traces.sort(key=lambda t: t.get("started_at", ""), reverse=True)

        return traces[:limit]

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """通过 ``trace_id`` 检索单个追踪记录。

        返回：
            追踪字典，如未找到则返回 ``None``。
        """
        for t in self._load_all():
            if t.get("trace_id") == trace_id:
                return t
        return None

    def get_stage_timings(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从追踪记录中提取阶段计时。

        返回：
            包含以下键的字典列表：stage_name、elapsed_ms、data。
            按出现顺序排列。
        """
        stages = trace.get("stages", [])
        timings: List[Dict[str, Any]] = []
        for s in stages:
            # 原始阶段字典包含：stage、timestamp、data（字典）、elapsed_ms
            # 直接提取内部的 'data' 字典，而不是平铺
            stage_data = s.get("data", {})
            if not isinstance(stage_data, dict):
                stage_data = {}
            timings.append(
                {
                    "stage_name": s.get("stage"),
                    "elapsed_ms": s.get("elapsed_ms", 0),
                    "data": stage_data,
                }
            )
        return timings

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _load_all(self) -> List[Dict[str, Any]]:
        """解析 JSONL 文件中的每一行。

        静默跳过格式错误的行。
        """
        if not self.traces_path.exists():
            return []

        traces: List[Dict[str, Any]] = []
        with self.traces_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed trace line: %s", line[:80])
        return traces
