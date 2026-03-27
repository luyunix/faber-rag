"""用于批量质量评估的评估运行器。

EvalRunner 读取黄金测试集，为每个测试用例运行 HybridSearch，
可选地生成答案，然后调用配置的评估器为
每个结果打分，生成结构化的评估报告。

设计原则：
- 配置驱动：通过 settings.yaml 选择评估器。
- 可观测：生成包含每个查询详情的 EvalReport。
- 解耦：适用于任何 BaseEvaluator 实现。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


@dataclass
class GoldenTestCase:
    """来自黄金测试集的单个评估测试用例。

    属性：
        query: 测试查询字符串。
        expected_chunk_ids: 用于 IR 指标的基准真值片段 ID。
        expected_sources: 基准真值源文件名（可选）。
        reference_answer: 用于 LLM-as-Judge 的参考答案文本（可选）。
    """

    query: str
    expected_chunk_ids: List[str] = field(default_factory=list)
    expected_sources: List[str] = field(default_factory=list)
    reference_answer: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GoldenTestCase:
        return cls(
            query=data["query"],
            expected_chunk_ids=data.get("expected_chunk_ids", []),
            expected_sources=data.get("expected_sources", []),
            reference_answer=data.get("reference_answer"),
        )


@dataclass
class QueryResult:
    """评估单个测试用例的结果。

    属性：
        query: 测试查询。
        retrieved_chunk_ids: 实际检索到的片段 ID。
        generated_answer: 生成的答案（如适用）。
        metrics: 此查询的评估指标。
        elapsed_ms: 检索 + 评估所花费的时间。
    """

    query: str
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    generated_answer: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class EvalReport:
    """所有测试用例的聚合评估报告。

    属性：
        query_results: 每个查询的评估结果。
        aggregate_metrics: 所有查询的平均指标。
        total_elapsed_ms: 整个评估的总时间。
        evaluator_name: 使用的评估器名称。
        test_set_path: 黄金测试集文件的路径。
    """

    query_results: List[QueryResult] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    total_elapsed_ms: float = 0.0
    evaluator_name: str = ""
    test_set_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """将报告序列化为字典。"""
        return {
            "evaluator_name": self.evaluator_name,
            "test_set_path": self.test_set_path,
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
            "aggregate_metrics": {
                k: round(v, 4) for k, v in self.aggregate_metrics.items()
            },
            "query_count": len(self.query_results),
            "query_results": [
                {
                    "query": qr.query,
                    "retrieved_chunk_ids": qr.retrieved_chunk_ids,
                    "generated_answer": qr.generated_answer,
                    "metrics": {k: round(v, 4) for k, v in qr.metrics.items()},
                    "elapsed_ms": round(qr.elapsed_ms, 1),
                }
                for qr in self.query_results
            ],
        }


def load_test_set(path: str | Path) -> List[GoldenTestCase]:
    """从 JSON 文件加载黄金测试集。

    参数：
        path: 黄金测试集 JSON 文件的路径。

    返回：
        TestCase 实例列表。

    抛出：
        FileNotFoundError: 如果文件不存在。
        ValueError: 如果文件格式无效。
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Golden test set not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "test_cases" not in data:
        raise ValueError(
            "Invalid golden test set format: missing 'test_cases' key."
        )

    return [GoldenTestCase.from_dict(tc) for tc in data["test_cases"]]


class EvalRunner:
    """针对黄金测试集运行批量评估。

    此类负责：
    1. 加载黄金测试集
    2. 为每个查询运行 HybridSearch
    3. 可选地生成答案
    4. 调用评估器为每个结果打分
    5. 将指标聚合到 EvalReport 中

    示例::

        runner = EvalRunner(
            settings=settings,
            hybrid_search=hybrid_search,
            evaluator=evaluator,
        )
        report = runner.run("tests/fixtures/golden_test_set.json")
        print(report.aggregate_metrics)
    """

    def __init__(
        self,
        settings: Any = None,
        hybrid_search: Any = None,
        evaluator: Optional[BaseEvaluator] = None,
        answer_generator: Any = None,
        answer_overrides: Optional[Dict[int, str]] = None,
        reranker: Any = None,
    ) -> None:
        """初始化 EvalRunner。

        参数：
            settings: 应用程序设置。
            hybrid_search: 用于检索的 HybridSearch 实例。
            evaluator: 用于评分的 BaseEvaluator 实例。
            answer_generator: 可选的可调用对象(query, chunks) -> str
                用于生成答案。如果为 None，则使用简单的拼接
                作为占位符。
            answer_overrides: 可选的字典，将测试用例索引（从 0 开始）
                映射到用户提供的答案字符串。如果存在，则使用覆盖
                的答案代替该测试用例的自动生成答案。
            reranker: 可选的 CoreReranker 实例，用于对结果进行重排序。
        """
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.evaluator = evaluator
        self.answer_generator = answer_generator
        self.answer_overrides = answer_overrides or {}
        self.reranker = reranker

    def run(
        self,
        test_set_path: str | Path,
        top_k: int = 10,
        collection: Optional[str] = None,
    ) -> EvalReport:
        """在黄金测试集上运行评估。

        参数：
            test_set_path: golden_test_set.json 的路径。
            top_k: 每个查询检索的片段数量。
            collection: 可选的集合名称过滤器。

        返回：
            包含每个查询和聚合指标的 EvalReport。

        抛出：
            FileNotFoundError: 如果测试集文件不存在。
            ValueError: 如果 evaluator 或 hybrid_search 未设置。
        """
        if self.evaluator is None:
            raise ValueError("EvalRunner requires an evaluator.")

        test_cases = load_test_set(test_set_path)
        if not test_cases:
            raise ValueError("Golden test set is empty.")

        logger.info(
            "Starting evaluation: %d test cases, evaluator=%s",
            len(test_cases),
            type(self.evaluator).__name__,
        )

        report = EvalReport(
            evaluator_name=type(self.evaluator).__name__,
            test_set_path=str(test_set_path),
        )

        t0 = time.monotonic()

        for idx, tc in enumerate(test_cases):
            logger.info("Evaluating [%d/%d]: %s", idx + 1, len(test_cases), tc.query[:60])
            # 如果此索引有用户提供的答案覆盖，则使用它
            answer_override = self.answer_overrides.get(idx)
            qr = self._evaluate_single(
                tc, top_k=top_k, collection=collection,
                answer_override=answer_override,
            )
            report.query_results.append(qr)

        report.total_elapsed_ms = (time.monotonic() - t0) * 1000.0
        report.aggregate_metrics = self._aggregate_metrics(report.query_results)

        logger.info(
            "Evaluation complete: %d queries, aggregate=%s",
            len(report.query_results),
            report.aggregate_metrics,
        )

        return report

    def _evaluate_single(
        self,
        test_case: GoldenTestCase,
        top_k: int = 10,
        collection: Optional[str] = None,
        answer_override: Optional[str] = None,
    ) -> QueryResult:
        """评估单个测试用例。

        参数：
            test_case: 要评估的测试用例。
            top_k: 要检索的结果数量。
            collection: 可选的集合过滤器。
            answer_override: 用户提供的答案文本。设置时，使用
                此答案代替从片段自动生成的答案。

        返回：
            包含此测试用例指标的 QueryResult。
        """
        t0 = time.monotonic()
        qr = QueryResult(query=test_case.query)

        # 第 1 步：检索片段
        retrieved_chunks = self._retrieve(test_case.query, top_k, collection)
        qr.retrieved_chunk_ids = [
            self._get_chunk_id(c) for c in retrieved_chunks
        ]

        # 第 2 步：生成答案 — 优先使用用户覆盖，然后是生成器，最后是回退
        if answer_override:
            answer = answer_override
        else:
            answer = self._generate_answer(test_case.query, retrieved_chunks)
        qr.generated_answer = answer

        # 第 3 步：构建基准真值
        ground_truth = (
            {"ids": test_case.expected_chunk_ids}
            if test_case.expected_chunk_ids
            else None
        )

        # 第 4 步：评估
        try:
            metrics = self.evaluator.evaluate(  # type: ignore[union-attr]
                query=test_case.query,
                retrieved_chunks=retrieved_chunks,
                generated_answer=answer,
                ground_truth=ground_truth,
            )
            qr.metrics = metrics
        except Exception as exc:
            logger.warning("Evaluation failed for '%s': %s", test_case.query[:40], exc)
            qr.metrics = {}

        qr.elapsed_ms = (time.monotonic() - t0) * 1000.0
        return qr

    def _retrieve(
        self,
        query: str,
        top_k: int,
        collection: Optional[str],
    ) -> List[Any]:
        """使用 HybridSearch + 可选重排序检索片段。

        如果搜索未配置，则返回空列表。
        """
        if self.hybrid_search is None:
            logger.warning("No HybridSearch configured; returning empty results.")
            return []

        try:
            # 如果启用了重排序器，则检索更多候选结果
            has_reranker = self.reranker is not None and getattr(self.reranker, 'is_enabled', False)
            initial_top_k = top_k * 2 if has_reranker else top_k

            results = self.hybrid_search.search(
                query=query,
                top_k=initial_top_k,
            )
            results = results if isinstance(results, list) else results.results

            # 如果启用则应用重排序
            if has_reranker and results:
                rerank_result = self.reranker.rerank(query=query, results=results, top_k=top_k)
                results = rerank_result.results

            return results
        except Exception as exc:
            logger.warning("Retrieval failed for '%s': %s", query[:40], exc)
            return []

    def _generate_answer(self, query: str, chunks: List[Any]) -> str:
        """从检索到的片段生成答案。

        如果提供了自定义 answer_generator，则使用它。
        否则，将片段文本拼接作为简单的占位符。
        """
        if self.answer_generator is not None:
            try:
                return self.answer_generator(query, chunks)
            except Exception as exc:
                logger.warning("Answer generation failed: %s", exc)

        # 回退：拼接片段文本
        texts = []
        for c in chunks:
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, dict):
                texts.append(c.get("text", str(c)))
            elif hasattr(c, "text"):
                texts.append(str(getattr(c, "text")))
            else:
                texts.append(str(c))

        return " ".join(texts[:5])  # first 5 chunks

    def _get_chunk_id(self, chunk: Any) -> str:
        """从各种表示中提取片段 ID。"""
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            for key in ("id", "chunk_id"):
                if key in chunk:
                    return str(chunk[key])
            return str(chunk)
        if hasattr(chunk, "chunk_id"):
            return str(getattr(chunk, "chunk_id"))
        if hasattr(chunk, "id"):
            return str(getattr(chunk, "id"))
        return str(chunk)

    @staticmethod
    def _aggregate_metrics(results: List[QueryResult]) -> Dict[str, float]:
        """计算所有查询结果的平均指标。

        参数：
            results: 包含每个查询指标的 QueryResult 列表。

        返回：
            平均指标值的字典。
        """
        if not results:
            return {}

        # 收集所有指标键
        all_keys: set[str] = set()
        for qr in results:
            all_keys.update(qr.metrics.keys())

        # 计算每个指标的平均值
        averages: Dict[str, float] = {}
        for key in sorted(all_keys):
            values = [qr.metrics[key] for qr in results if key in qr.metrics]
            averages[key] = sum(values) / len(values) if values else 0.0

        return averages
