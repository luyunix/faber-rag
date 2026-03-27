"""追踪上下文，用于跨管道阶段的可观测性。

提供 trace_id、trace_type（query/ingestion）、每阶段时间、
finish() 生命周期和 to_dict() 序列化用于 JSON Lines 输出。
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional


@dataclass
class TraceContext:
    """请求范围的追踪上下文，记录管道阶段和时间。

    属性：
        trace_id: 此追踪的唯一标识符。
        trace_type: ``"query"`` 或 ``"ingestion"``。
        started_at: 追踪创建时的 ISO-8601 时间戳。
        finished_at: 调用 ``finish()`` 时的 ISO-8601 时间戳，或 None。
        stages: 记录的阶段字典的有序列表。
        metadata: 附加到追踪的任意键/值对。
    """

    trace_type: Literal["query", "ingestion"] = "query"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = field(default=None)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 内部单调时钟用于准确计算耗时
    _start_mono: float = field(default_factory=time.monotonic, repr=False)
    _finish_mono: Optional[float] = field(default=None, repr=False)
    _stage_timings: Dict[str, float] = field(default_factory=dict, repr=False)

    # ---- 记录 ---------------------------------------------------

    def record_stage(
        self,
        stage_name: str,
        data: Dict[str, Any],
        elapsed_ms: Optional[float] = None,
    ) -> None:
        """记录来自管道阶段的数据。

        参数：
            stage_name: 阶段名称（例如 ``"dense_retrieval"``）。
            data: 阶段特定的载荷（方法、提供方、详细信息 …）。
            elapsed_ms: 预先计算的耗时（毫秒）。为 *None* 时
                调用者应在外部测量，或留给
                ``stage_timer`` 上下文管理器处理。
        """
        entry: Dict[str, Any] = {
            "stage": stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        if elapsed_ms is not None:
            entry["elapsed_ms"] = round(elapsed_ms, 2)
            self._stage_timings[stage_name] = elapsed_ms
        self.stages.append(entry)

    # ---- 生命周期 ----------------------------------------------------

    def finish(self) -> None:
        """将追踪标记为完成并记录挂钟结束时间。"""
        self._finish_mono = time.monotonic()
        self.finished_at = datetime.now(timezone.utc).isoformat()

    # ---- 时间辅助函数 -----------------------------------------------

    def elapsed_ms(self, stage_name: Optional[str] = None) -> float:
        """返回耗时（毫秒）。

        参数：
            stage_name: 如果给定，返回该阶段记录的耗时。
                为 *None* 时返回总追踪耗时
                （开始 → 结束，或如果尚未结束则为开始 → 现在）。

        返回：
            耗时毫秒数。

        异常：
            KeyError: 如果提供了 *stage_name* 但未找到。
        """
        if stage_name is not None:
            if stage_name not in self._stage_timings:
                raise KeyError(f"Stage '{stage_name}' has no recorded timing")
            return self._stage_timings[stage_name]

        end = self._finish_mono if self._finish_mono is not None else time.monotonic()
        return (end - self._start_mono) * 1000.0

    # ---- 序列化 ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """将追踪序列化为适合 ``json.dumps`` 的普通字典。

        返回：
            包含所有追踪数据的字典。
        """
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": round(self.elapsed_ms(), 2),
            "stages": list(self.stages),
            "metadata": dict(self.metadata),
        }

    # ---- 向后兼容的辅助函数，用于 C5 / C6 -----------------------
    def get_stage_data(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """检索特定阶段记录的数据。

        搜索阶段列表（重复名称时后写入优先）。

        参数：
            stage_name: 要检索的阶段名称。

        返回：
            匹配阶段的 ``data`` 字典，或 *None*。
        """
        for entry in reversed(self.stages):
            if entry.get("stage") == stage_name:
                return entry.get("data")
        return None
