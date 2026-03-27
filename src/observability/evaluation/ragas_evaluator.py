"""基于 Ragas 的 RAG 质量评估器。

此评估器包装 Ragas 框架以计算 LLM-as-Judge 指标：
- Faithfulness: 答案是否坚持检索到的上下文？
- Answer Relevancy: 答案是否与查询相关？
- Context Precision: 检索到的片段是否相关且排序良好？

设计原则：
- 可插拔：实现 BaseEvaluator 接口，可通过工厂切换。
- 配置驱动：LLM/Embedding 后端从 settings.yaml 读取。
- 优雅降级：如果未安装 ragas，会显示清晰的 ImportError。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# 指标名称常量
FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"

SUPPORTED_METRICS = {FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION}


def _import_ragas() -> None:
    """验证 ragas 是否可导入，如不可导入则抛出清晰的错误。"""
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'ragas' package is required for RagasEvaluator. "
            "Install it with: pip install ragas datasets"
        ) from exc


class RagasEvaluator(BaseEvaluator):
    """使用 Ragas 框架进行 LLM-as-Judge 指标的评估器。

    Ragas 不需要基准真值标签，它使用 LLM 来评估
    生成答案与检索到的上下文的质量。

    支持的指标：
        - faithfulness: 衡量与上下文的事实一致性。
        - answer_relevancy: 衡量答案与查询的相关程度。
        - context_precision: 衡量检索到的片段的相关性/排序。

    示例::

        evaluator = RagasEvaluator(settings=settings)
        metrics = evaluator.evaluate(
            query="What is RAG?",
            retrieved_chunks=[{"id": "c1", "text": "RAG is ..."}],
            generated_answer="RAG stands for ...",
        )
        # metrics == {"faithfulness": 0.95, "answer_relevancy": 0.88, ...}    """

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 RagasEvaluator。

        参数：
            settings: 应用程序设置（用于配置 LLM 后端）。
            metrics: 要计算的指标名称。默认为所有支持的指标。
            **kwargs: 附加参数（保留）。

        抛出：
            ImportError: 如果未安装 ragas。
            ValueError: 如果请求的指标名称不受支持。
        """
        _import_ragas()

        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalised = [m.strip().lower() for m in (metrics or [])]
        if not normalised:
            normalised = sorted(SUPPORTED_METRICS)

        unsupported = [m for m in normalised if m not in SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported ragas metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}"
            )

        self._metric_names = normalised

    # ── 公共 API ────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """使用 Ragas LLM-as-Judge 指标评估 RAG 质量。

        参数：
            query: 用户查询字符串。
            retrieved_chunks: 检索到的片段（带有 'text' 键的字典或字符串）。
            generated_answer: 生成的答案文本。Ragas 需要此参数。
            ground_truth: Ragas 忽略（LLM-as-Judge 不需要）。
            trace: 可选的 TraceContext 用于可观测性。
            **kwargs: 附加参数。

        返回：
            将指标名称映射到浮点分数（0.0 – 1.0）的字典。

        抛出：
            ValueError: 如果查询/片段无效或缺少 generated_answer。
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        if not generated_answer or not generated_answer.strip():
            raise ValueError(
                "RagasEvaluator requires a non-empty 'generated_answer'. "
                "Ragas uses LLM-as-Judge and needs the answer text to evaluate."
            )

        contexts = self._extract_texts(retrieved_chunks)

        try:
            result = self._run_ragas(query, contexts, generated_answer)
        except Exception as exc:
            logger.error("Ragas evaluation failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Ragas evaluation failed: {exc}") from exc

        return result

    # ── 私有辅助方法 ───────────────────────────────────────────

    def _run_ragas(
        self,
        query: str,
        contexts: List[str],
        answer: str,
    ) -> Dict[str, float]:
        """执行 Ragas 指标集合并返回标准化分数。

        Ragas 0.4+ 指标使用每个指标的 ``score()`` 而不是
        传统的 ``evaluate()`` 管道。每个指标都有自己的签名：
        - Faithfulness / ContextPrecision: (user_input, response, retrieved_contexts)
        - AnswerRelevancy: (user_input, response)
        """
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
        )

        # 构建 LLM / Embedding wrappers from settings        llm, embeddings = self._build_wrappers()

        scores: Dict[str, float] = {}

        for metric_name in self._metric_names:
            if metric_name == FAITHFULNESS:
                m = Faithfulness(llm=llm)
                result = m.score(
                    user_input=query, response=answer, retrieved_contexts=contexts,
                )
            elif metric_name == ANSWER_RELEVANCY:
                m = AnswerRelevancy(llm=llm, embeddings=embeddings)
                result = m.score(user_input=query, response=answer)
            elif metric_name == CONTEXT_PRECISION:
                m = ContextPrecisionWithoutReference(llm=llm)
                result = m.score(
                    user_input=query, response=answer, retrieved_contexts=contexts,
                )
            else:
                continue

            scores[metric_name] = float(result.value) if result.value is not None else 0.0

        return scores

    def _build_wrappers(self) -> tuple:
        """从项目设置构建 Ragas LLM 和 Embedding 包装器。

        使用 Ragas 0.4+ 原生 API（InstructorLLM + OpenAIEmbeddings）
        而不是已弃用的 LangchainLLMWrapper。

        返回：
            (llm_wrapper, embeddings_wrapper) 元组。
        """
        from openai import AsyncAzureOpenAI, AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings import OpenAIEmbeddings

        if self.settings is None:
            raise ValueError("Settings required to create LLM for Ragas evaluation")

        # ── LLM ──
        llm_cfg = self.settings.llm
        provider = llm_cfg.provider.lower()
        llm_azure_endpoint = getattr(llm_cfg, "azure_endpoint", None)

        # Azure 兼容模式：如果配置了 azure_endpoint，即使 provider 是 "openai"，
        # 也使用 Azure 客户端（符合项目惯例）。
        use_azure_llm = (
            provider == "azure"
            or (provider == "openai" and llm_azure_endpoint)
        )

        if use_azure_llm:
            llm_client = AsyncAzureOpenAI(
                api_key=llm_cfg.api_key,
                azure_endpoint=llm_azure_endpoint or llm_cfg.azure_endpoint,
                api_version=getattr(llm_cfg, "api_version", None) or "2024-02-15-preview",
            )
        elif provider == "openai":
            llm_client = AsyncOpenAI(api_key=llm_cfg.api_key)
        else:
            raise ValueError(
                f"Unsupported LLM provider for Ragas: '{provider}'. "
                "Supported: azure, openai"
            )

        llm = llm_factory(llm_cfg.model, client=llm_client, max_tokens=8192)

        # ── Embeddings ──
        emb_cfg = self.settings.embedding
        emb_provider = emb_cfg.provider.lower()
        emb_azure_endpoint = getattr(emb_cfg, "azure_endpoint", None)

        # Embedding 使用相同的 Azure 兼容模式检测
        use_azure_emb = (
            emb_provider == "azure"
            or (emb_provider == "openai" and emb_azure_endpoint)
        )

        if use_azure_emb:
            emb_client = AsyncAzureOpenAI(
                api_key=emb_cfg.api_key,
                azure_endpoint=emb_azure_endpoint or emb_cfg.azure_endpoint,
                api_version=getattr(emb_cfg, "api_version", None) or "2024-02-15-preview",
            )
        elif emb_provider == "openai":
            emb_client = AsyncOpenAI(api_key=emb_cfg.api_key)
        else:
            raise ValueError(
                f"Unsupported embedding provider for Ragas: '{emb_provider}'. "
                "Supported: azure, openai"
            )

        embeddings = OpenAIEmbeddings(model=emb_cfg.model, client=emb_client)

        return llm, embeddings

    def _extract_texts(self, chunks: List[Any]) -> List[str]:
        """从各种片段表示中提取文本字符串。

        参数：
            chunks: 片段字典、字符串或带有 .text 的对象列表。

        返回：
            文本字符串列表。
        """
        texts: List[str] = []
        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
            elif isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or chunk.get("page_content", "")
                texts.append(str(text))
            elif hasattr(chunk, "text"):
                texts.append(str(getattr(chunk, "text")))
            else:
                texts.append(str(chunk))
        return texts

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """如果可用，从设置中提取指标列表。"""
        if settings is None:
            return []
        evaluation = getattr(settings, "evaluation", None)
        if evaluation is None:
            return []
        raw_metrics = getattr(evaluation, "metrics", None)
        if raw_metrics is None:
            return []
        # 过滤器 to only ragas-supported metrics        return [m for m in raw_metrics if m.lower() in SUPPORTED_METRICS]
