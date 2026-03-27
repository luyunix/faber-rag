"""核心层重排序器，用于编排 libs.reranker 后端，并支持故障回退。

本模块实现 CoreReranker 类，该类：
1. 通过 RerankerFactory 集成 libs.reranker（LLM、CrossEncoder、None）
2. 当后端失败或超时时提供优雅降级
3. 在 RetrievalResult 与重排序器输入/输出格式之间进行转换
4. 支持 TraceContext 用于可观测性

设计原则：
- 可插拔：使用 RerankerFactory 实例化配置的后端
- 配置驱动：从 settings.yaml 读取重排序设置
- 优雅降级：后端故障时返回原始顺序
- 可观测：集成 TraceContext 用于调试
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import BaseReranker, NoneReranker
from src.libs.reranker.reranker_factory import RerankerFactory

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


class RerankError(RuntimeError):
    """当重排序失败时抛出。"""


@dataclass
class RerankConfig:
    """CoreReranker 的配置。

    属性：
        enabled: 是否启用重排序
        top_k: 重排序后返回的结果数量
        timeout: 重排序器后端超时时间（秒）
        fallback_on_error: 发生错误时是否返回原始顺序
    """
    enabled: bool = True
    top_k: int = 5
    timeout: float = 30.0
    fallback_on_error: bool = True


@dataclass
class RerankResult:
    """重排序操作的结果。

    属性：
        results: 重排序后的 RetrievalResult 列表
        used_fallback: 是否因后端故障使用了降级
        fallback_reason: 降级原因（如适用）
        reranker_type: 使用的重排序器类型（'llm'、'cross_encoder'、'none'）
        original_order: 重排序前的原始结果（用于调试）
    """
    results: List[RetrievalResult] = field(default_factory=list)
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    reranker_type: str = "none"
    original_order: Optional[List[RetrievalResult]] = None


class CoreReranker:
    """支持降级的核心层重排序器。

    本类包装 libs.reranker 实现并提供：
    1. 在 RetrievalResult 与重排序器字典格式之间进行类型转换
    2. 后端故障时优雅降级
    3. 配置驱动的后端选择
    4. TraceContext 集成

    应用的设计原则：
    - 可插拔：通过 RerankerFactory 获取后端
    - 配置驱动：所有参数来自设置
    - 降级：故障时返回原始顺序
    - 可观测：支持 TraceContext

    示例：
        >>> from src.core.settings import load_settings
        >>> settings = load_settings("config/settings.yaml")
        >>> reranker = CoreReranker(settings)
        >>> results = [RetrievalResult(chunk_id="1", score=0.8, text="...", metadata={})]
        >>> reranked = reranker.rerank("query", results)
        >>> print(reranked.results)
    """

    def __init__(
        self,
        settings: Settings,
        reranker: Optional[BaseReranker] = None,
        config: Optional[RerankConfig] = None,
    ) -> None:
        """初始化 CoreReranker。

        参数：
            settings: 包含重排序配置的应用设置。
            reranker: 可选的重排序器后端。为 None 时通过 RerankerFactory 创建。
            config: 可选的 RerankConfig。为 None 时从设置中提取。
        """
        self.settings = settings

        # 从设置中提取配置或使用提供的配置
        if config is not None:
            self.config = config
        else:
            self.config = self._extract_config(settings)

        # 初始化重排序器后端
        if reranker is not None:
            self._reranker = reranker
        elif not self.config.enabled:
            self._reranker = NoneReranker(settings=settings)
        else:
            try:
                self._reranker = RerankerFactory.create(settings)
            except Exception as e:
                logger.warning(f"创建重排序器失败，使用 NoneReranker：{e}")
                self._reranker = NoneReranker(settings=settings)

        # 为结果报告确定重排序器类型
        self._reranker_type = self._get_reranker_type()

    def _extract_config(self, settings: Settings) -> RerankConfig:
        """从设置中提取 RerankConfig。

        参数：
            settings: 应用设置。

        返回：
            从设置中取值得到的 RerankConfig。
        """
        try:
            rerank_settings = settings.rerank
            return RerankConfig(
                enabled=bool(rerank_settings.enabled) if rerank_settings else False,
                top_k=int(rerank_settings.top_k) if rerank_settings and hasattr(rerank_settings, 'top_k') else 5,
                timeout=float(getattr(rerank_settings, 'timeout', 30.0)) if rerank_settings else 30.0,
                fallback_on_error=True,
            )
        except AttributeError:
            logger.warning("缺少重排序配置，使用默认值（禁用）")
            return RerankConfig(enabled=False)

    def _get_reranker_type(self) -> str:
        """获取当前重排序器后端的类型名称。

        返回：
            重排序器类型的字符串标识符。
        """
        class_name = self._reranker.__class__.__name__
        if "LLM" in class_name:
            return "llm"
        elif "CrossEncoder" in class_name:
            return "cross_encoder"
        elif "None" in class_name:
            return "none"
        else:
            return class_name.lower()

    def _results_to_candidates(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """将 RetrievalResults 转换为重排序器候选格式。

        参数：
            results: RetrievalResult 对象列表。

        返回：
            适合重排序器输入的字典列表。
        """
        candidates = []
        for result in results:
            candidates.append({
                "id": result.chunk_id,
                "text": result.text,
                "score": result.score,
                "metadata": result.metadata.copy(),
            })
        return candidates

    def _candidates_to_results(
        self,
        candidates: List[Dict[str, Any]],
        original_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """将重排序后的候选转换回 RetrievalResults。

        参数：
            candidates: 从重排序器获得的已重排序候选。
            original_results: 原始结果供参考。

        返回：
            按重排序排序的 RetrievalResult 列表。
        """
        # 从原始结果构建查找表
        id_to_original = {r.chunk_id: r for r in original_results}

        results = []
        for candidate in candidates:
            chunk_id = candidate["id"]

            # 获取原始结果或创建新结果
            if chunk_id in id_to_original:
                original = id_to_original[chunk_id]
                # 使用更新后的分数创建新结果
                rerank_score = candidate.get("rerank_score", candidate.get("score", 0.0))
                results.append(RetrievalResult(
                    chunk_id=original.chunk_id,
                    score=rerank_score,
                    text=original.text,
                    metadata={
                        **original.metadata,
                        "original_score": original.score,
                        "rerank_score": rerank_score,
                        "reranked": True,
                    },
                ))
            else:
                # 候选不在原始结果中 - 根据候选数据构建
                results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    score=candidate.get("rerank_score", candidate.get("score", 0.0)),
                    text=candidate.get("text", ""),
                    metadata=candidate.get("metadata", {}),
                ))

        return results

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> RerankResult:
        """使用配置的后端对检索结果进行重排序。

        参数：
            query: 用户查询字符串。
            results: 要重排序的 RetrievalResult 对象列表。
            top_k: 返回的结果数量。为 None 时使用 config.top_k。
            trace: 用于可观测性的可选 TraceContext。
            **kwargs: 传递给重排序器后端的额外参数。

        返回：
            包含重排序结果和元数据的 RerankResult。
        """
        effective_top_k = top_k if top_k is not None else self.config.top_k

        # 对空或单个结果提前返回
        if not results:
            return RerankResult(
                results=[],
                used_fallback=False,
                reranker_type=self._reranker_type,
            )

        if len(results) == 1:
            return RerankResult(
                results=results[:],
                used_fallback=False,
                reranker_type=self._reranker_type,
            )

        # 如果禁用重排序，返回原始顺序的前 top_k 个结果        if not self.config.enabled or isinstance(self._reranker, NoneReranker):
            return RerankResult(
                results=results[:effective_top_k],
                used_fallback=False,
                reranker_type="none",
                original_order=results[:],
            )

        # 转换为重排序器输入格式
        candidates = self._results_to_candidates(results)

        # 尝试重排序
        try:
            logger.debug(f"使用 {self._reranker_type} 对 {len(candidates)} 个候选进行重排序")
            _t0 = time.monotonic()
            reranked_candidates = self._reranker.rerank(
                query=query,
                candidates=candidates,
                trace=trace,
                **kwargs,
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0

            # 转换回 RetrievalResult
            reranked_results = self._candidates_to_results(reranked_candidates, results)
                        
            # 应用 top_k 限制
            final_results = reranked_results[:effective_top_k]

            logger.info(f"重排序完成：返回 {len(final_results)} 个结果")

            if trace is not None:
                trace.record_stage("rerank", {
                    "method": self._reranker_type,
                    "provider": self._reranker_type,
                    "input_count": len(candidates),
                    "output_count": len(final_results),
                    "chunks": [
                        {
                            "chunk_id": r.chunk_id,
                            "score": round(r.score, 4),
                            "text": r.text or "",
                            "source": r.metadata.get("source_path", r.metadata.get("source", "")),
                        }
                        for r in final_results
                    ],
                }, elapsed_ms=_elapsed)

            return RerankResult(
                results=final_results,
                used_fallback=False,
                reranker_type=self._reranker_type,
                original_order=results[:],
            )

        except Exception as e:
            logger.warning(f"重排序失败，使用降级：{e}")

            if self.config.fallback_on_error:
                # 返回原始顺序作为降级
                fallback_results = []
                for result in results[:effective_top_k]:
                    fallback_results.append(RetrievalResult(
                        chunk_id=result.chunk_id,
                        score=result.score,
                        text=result.text,
                        metadata={
                            **result.metadata,
                            "reranked": False,
                            "rerank_fallback": True,
                        },
                    ))

                return RerankResult(
                    results=fallback_results,
                    used_fallback=True,
                    fallback_reason=str(e),
                    reranker_type=self._reranker_type,
                    original_order=results[:],
                )
            else:
                raise RerankError(f"重排序失败且禁用降级：{e}") from e

    @property
    def reranker_type(self) -> str:
        """获取当前重排序器后端的类型。"""
        return self._reranker_type

    @property
    def is_enabled(self) -> bool:
        """检查重排序是否启用。"""
        return self.config.enabled and not isinstance(self._reranker, NoneReranker)


def create_core_reranker(
    settings: Settings,
    reranker: Optional[BaseReranker] = None,
) -> CoreReranker:
    """用于创建 CoreReranker 实例的工厂函数。

    参数：
        settings: 应用设置。
        reranker: 可选的重排序器后端覆盖。

    返回：
        配置好的 CoreReranker 实例。
    """
    return CoreReranker(settings=settings, reranker=reranker)
