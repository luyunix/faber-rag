"""用于轻量级指标的自定义评估器实现。

此评估器计算简单、确定性的指标，如命中率（hit rate）和 MRR（平均倒数排名）。
它专为快速回归检查和合理性验证而设计。
"""

from __future__ import annotations

from pathlib import PurePath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """用于轻量级指标（hit_rate、mrr）的自定义评估器。

    该评估器期望检索到的块包含标识符字段。
    支持的 ID 字段：id、chunk_id、document_id、doc_id。
    """

    SUPPORTED_METRICS = {"hit_rate", "mrr"}
    _ID_FIELDS = ("id", "chunk_id", "document_id", "doc_id")
    _SOURCE_FIELDS = ("source", "source_path", "file_path", "filename")
    requires_retrieval_ground_truth = True

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalized = [str(metric).strip().lower() for metric in (metrics or [])]
        if not normalized:
            normalized = ["hit_rate", "mrr"]

        unsupported = [metric for metric in normalized if metric not in self.SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                "不支持的自定义指标："
                f"{', '.join(unsupported)}。支持的指标：{', '.join(sorted(self.SUPPORTED_METRICS))}"
            )

        self.metrics = normalized

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """计算给定检索结果的请求指标。

        参数：
            query: 用户查询字符串。
            retrieved_chunks: 检索到的块或记录。
            generated_answer: 可选的生成答案（未使用）。
            ground_truth: 基准真实 ID 或结构。
            trace: 可选的 TraceContext（未使用）。
            **kwargs: 额外的参数（未使用）。

        返回：
            指标名称到浮点数值的字典。
        """
        self.validate_query(query)
        if not isinstance(retrieved_chunks, list):
            raise ValueError("retrieved_chunks 必须是列表")

        retrieved_ids = self._extract_ids(retrieved_chunks, label="retrieved_chunks")
        retrieved_sources = [self._extract_sources(item) for item in retrieved_chunks]
        ground_truth_ids, ground_truth_sources = self._extract_ground_truth(ground_truth)

        if not ground_truth_ids and not ground_truth_sources:
            raise ValueError(
                "CustomEvaluator 需要正确答案标签。请在测试用例中提供 "
                "expected_chunk_ids 或 expected_sources。"
            )

        relevance = [
            self._is_relevant(
                retrieved_id=retrieved_id,
                retrieved_sources=retrieved_sources[index],
                ground_truth_ids=ground_truth_ids,
                ground_truth_sources=ground_truth_sources,
            )
            for index, retrieved_id in enumerate(retrieved_ids)
        ]

        results: Dict[str, float] = {}

        if "hit_rate" in self.metrics:
            results["hit_rate"] = 1.0 if any(relevance) else 0.0
        if "mrr" in self.metrics:
            results["mrr"] = self._compute_mrr_from_relevance(relevance)

        return results

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """如果可用，从设置中提取指标列表。"""
        if settings is None:
            return []
        metrics = getattr(getattr(settings, "evaluation", None), "metrics", None)
        if metrics is None:
            return []
        return [str(metric) for metric in metrics]

    def _extract_ground_truth_ids(self, ground_truth: Optional[Any]) -> List[str]:
        """从各种输入形式中提取基准真实 ID。"""
        if ground_truth is None:
            return []
        if isinstance(ground_truth, str):
            return [ground_truth]
        if isinstance(ground_truth, dict):
            if "ids" in ground_truth and isinstance(ground_truth["ids"], list):
                return self._extract_ids(ground_truth["ids"], label="ground_truth.ids")
            if any(field in ground_truth for field in self._ID_FIELDS):
                return self._extract_ids([ground_truth], label="ground_truth")
            return []
        if isinstance(ground_truth, list):
            return self._extract_ids(ground_truth, label="ground_truth")

        raise ValueError(
            f"不支持的 ground_truth 类型：{type(ground_truth).__name__}。"
            "期望 str、dict、list 或 None。"
        )

    def _extract_ground_truth(
        self,
        ground_truth: Optional[Any],
    ) -> Tuple[List[str], Set[str]]:
        """提取正确 chunk ID 和来源文件标签。"""
        ids = self._extract_ground_truth_ids(ground_truth)
        sources: Set[str] = set()

        if isinstance(ground_truth, dict):
            raw_sources = ground_truth.get("sources", [])
            if isinstance(raw_sources, str):
                raw_sources = [raw_sources]
            if not isinstance(raw_sources, list):
                raise ValueError("ground_truth.sources 必须是字符串或字符串列表")
            sources = {
                normalized
                for source in raw_sources
                if (normalized := self._normalize_source(source))
            }

        return ids, sources

    def _extract_ids(self, items: Iterable[Any], label: str) -> List[str]:
        """从项目列表中提取 ID。"""
        ids: List[str] = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                ids.append(item)
                continue
            if isinstance(item, dict):
                for field in self._ID_FIELDS:
                    if field in item:
                        ids.append(str(item[field]))
                        break
                else:
                    raise ValueError(
                        f"{label}[{index}] 中缺少 ID 字段。"
                        f"期望以下字段之一：{', '.join(self._ID_FIELDS)}"
                    )
                continue
            if hasattr(item, "id"):
                ids.append(str(getattr(item, "id")))
                continue
            if hasattr(item, "chunk_id"):
                ids.append(str(getattr(item, "chunk_id")))
                continue

            raise ValueError(
                f"无法从 {label}[{index}]（类型 {type(item).__name__}）中提取 ID"
            )

        return ids

    def _extract_sources(self, item: Any) -> Set[str]:
        """从检索结果本身及其 metadata 中提取来源路径。"""
        candidates: List[Any] = []

        if isinstance(item, dict):
            candidates.extend(item.get(field) for field in self._SOURCE_FIELDS)
            metadata = item.get("metadata", {})
        else:
            candidates.extend(getattr(item, field, None) for field in self._SOURCE_FIELDS)
            metadata = getattr(item, "metadata", {})

        if isinstance(metadata, dict):
            candidates.extend(metadata.get(field) for field in self._SOURCE_FIELDS)

        return {
            normalized
            for candidate in candidates
            if (normalized := self._normalize_source(candidate))
        }

    @staticmethod
    def _normalize_source(source: Any) -> str:
        if source is None:
            return ""
        return str(source).strip().replace("\\", "/").lower()

    def _is_relevant(
        self,
        retrieved_id: str,
        retrieved_sources: Set[str],
        ground_truth_ids: Sequence[str],
        ground_truth_sources: Set[str],
    ) -> bool:
        if retrieved_id in ground_truth_ids:
            return True

        for actual in retrieved_sources:
            actual_name = PurePath(actual).name
            for expected in ground_truth_sources:
                if actual == expected or actual_name == PurePath(expected).name:
                    return True
        return False

    @staticmethod
    def _compute_mrr_from_relevance(relevance: Sequence[bool]) -> float:
        for rank, matched in enumerate(relevance, start=1):
            if matched:
                return 1.0 / rank
        return 0.0

    def _compute_hit_rate(self, retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
        """计算命中率（二值）。"""
        if not ground_truth_ids:
            return 0.0
        return 1.0 if any(item in ground_truth_ids for item in retrieved_ids) else 0.0

    def _compute_mrr(self, retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
        """计算平均倒数排名（MRR）。"""
        if not ground_truth_ids:
            return 0.0
        for rank, item in enumerate(retrieved_ids, start=1):
            if item in ground_truth_ids:
                return 1.0 / rank
        return 0.0
