"""混合搜索引擎，编排稠密 + 稀疏检索与 RRF 融合。

本模块实现 HybridSearch 类，该类组合：
1. QueryProcessor：预处理查询并提取关键词/过滤器
2. DenseRetriever：使用 embedding 进行语义搜索
3. SparseRetriever：使用 BM25 进行关键词搜索
4. RRFFusion：使用倒数排名融合（RRF）组合结果

设计原则：
- 优雅降级：如果一条检索路径失败，则回退到另一条
- 可插拔：所有组件通过构造函数注入以保证可测试性
- 可观测：集成 TraceContext 用于调试和监控
- 配置驱动：Top-k 和其他参数从设置中读取
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.core.types import ProcessedQuery, RetrievalResult

if TYPE_CHECKING:
    from src.core.query_engine.dense_retriever import DenseRetriever
    from src.core.query_engine.fusion import RRFFusion
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.sparse_retriever import SparseRetriever
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


def _snapshot_results(
    results: Optional[List[RetrievalResult]],
) -> List[Dict[str, Any]]:
    """创建用于追踪存储的检索结果的可序列化快照。

    参数：
        results: RetrievalResult 对象列表。

    返回：
        包含 chunk_id、score、完整文本、源的字典列表。
    """
    if not results:
        return []
    return [
        {
            "chunk_id": r.chunk_id,
            "score": round(r.score, 4),
            "text": r.text or "",
            "source": r.metadata.get("source_path", r.metadata.get("source", "")),
        }
        for r in results
    ]


@dataclass
class HybridSearchConfig:
    """HybridSearch 的配置。

    属性：
        dense_top_k: 稠密检索返回的结果数量
        sparse_top_k: 稀疏检索返回的结果数量
        fusion_top_k: 融合后最终返回的结果数量
        enable_dense: 是否使用稠密检索
        enable_sparse: 是否使用稀疏检索
        parallel_retrieval: 是否并行运行检索
        metadata_filter_post: 在融合后应用元数据过滤器（降级）
    """
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 10
    enable_dense: bool = True
    enable_sparse: bool = True
    parallel_retrieval: bool = True
    metadata_filter_post: bool = True


@dataclass
class HybridSearchResult:
    """混合搜索操作的结果。

    属性：
        results: 最终的 RetrievalResult 排名列表
        dense_results: 稠密检索的结果（用于调试）
        sparse_results: 稀疏检索的结果（用于调试）
        dense_error: 如果稠密检索失败的错误信息
        sparse_error: 如果稀疏检索失败的错误信息
        used_fallback: 是否使用了降级模式
        processed_query: 处理后的查询（用于调试）
    """
    results: List[RetrievalResult] = field(default_factory=list)
    dense_results: Optional[List[RetrievalResult]] = None
    sparse_results: Optional[List[RetrievalResult]] = None
    dense_error: Optional[str] = None
    sparse_error: Optional[str] = None
    used_fallback: bool = False
    processed_query: Optional[ProcessedQuery] = None


class HybridSearch:
    """组合稠密和稀疏检索的混合搜索引擎。

    本类编排完整的混合搜索流程：
    1. 查询处理：从原始查询提取关键词和过滤器
    2. 并行检索：并发运行稠密和稀疏检索器
    3. 融合：使用 RRF 算法组合结果
    4. 后过滤：如果指定则应用元数据过滤器

    应用的设计原则：
    - 优雅降级：如果一条路径失败，则使用另一条的结果
    - 可插拔：所有组件通过依赖注入
    - 可观测：支持 TraceContext 以便调试
    - 配置驱动：所有参数来自设置

    示例：
        >>> # 初始化组件
        >>> query_processor = QueryProcessor()
        >>> dense_retriever = DenseRetriever(settings, embedding_client, vector_store)
        >>> sparse_retriever = SparseRetriever(settings, bm25_indexer, vector_store)
        >>> fusion = RRFFusion(k=60)
        >>>
        >>> # 创建 HybridSearch
        >>> hybrid = HybridSearch(
        ...     settings=settings,
        ...     query_processor=query_processor,
        ...     dense_retriever=dense_retriever,
        ...     sparse_retriever=sparse_retriever,
        ...     fusion=fusion
        ... )
        >>>
        >>> # 搜索
        >>> results = hybrid.search("如何配置 Azure OpenAI？", top_k=10)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        query_processor: Optional[QueryProcessor] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        fusion: Optional[RRFFusion] = None,
        config: Optional[HybridSearchConfig] = None,
    ) -> None:
        """使用组件初始化 HybridSearch。

        参数：
            settings: 用于提取配置的应用设置。
            query_processor: 用于预处理查询的 QueryProcessor。
            dense_retriever: 用于语义搜索的 DenseRetriever。
            sparse_retriever: 用于关键词搜索的 SparseRetriever。
            fusion: 用于组合结果的 RRFFusion。
            config: 可选的 HybridSearchConfig。如果未提供，则从设置中提取。

        注意：
            至少必须提供 dense_retriever 或 sparse_retriever 中的一个
            搜索才能正常工作。如果其中一个不可用或失败，
            搜索将优雅降级。
        """
        self.query_processor = query_processor
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion = fusion

        # 从设置中提取配置或使用提供的/默认配置
        self.config = config or self._extract_config(settings)

        logger.info(
            f"HybridSearch 已初始化：dense={self.dense_retriever is not None}, "
            f"sparse={self.sparse_retriever is not None}, "
            f"config={self.config}"
        )

    def _extract_config(self, settings: Optional[Settings]) -> HybridSearchConfig:
        """从设置中提取 HybridSearchConfig。

        参数：
            settings: 应用设置对象。

        返回：
            从设置中取值或默认值的 HybridSearchConfig。
        """
        if settings is None:
            return HybridSearchConfig()

        retrieval_config = getattr(settings, 'retrieval', None)
        if retrieval_config is None:
            return HybridSearchConfig()

        return HybridSearchConfig(
            dense_top_k=getattr(retrieval_config, 'dense_top_k', 20),
            sparse_top_k=getattr(retrieval_config, 'sparse_top_k', 20),
            fusion_top_k=getattr(retrieval_config, 'fusion_top_k', 10),
            enable_dense=True,
            enable_sparse=True,
            parallel_retrieval=True,
            metadata_filter_post=True,
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        return_details: bool = False,
    ) -> List[RetrievalResult] | HybridSearchResult:
        """执行结合稠密和稀疏检索的混合搜索。

        参数：
            query: 搜索查询字符串。
            top_k: 返回的最大结果数量。为 None 时使用 config.fusion_top_k。
            filters: 可选的元数据过滤器（例如 {"collection": "docs"}）。
            trace: 用于可观测性的可选 TraceContext。
            return_details: 如果为 True，返回带调试信息的 HybridSearchResult。

        返回：
            如果 return_details=False：按相关性排序的 RetrievalResult 列表。
            如果 return_details=True：带完整详细信息的 HybridSearchResult。

        异常：
            ValueError: 如果查询为空或无效。
            RuntimeError: 如果两个检索器都失败或不可用。

        示例：
            >>> results = hybrid.search("Azure configuration", top_k=5)
            >>> for r in results:
            ...     print(f"[{r.score:.4f}] {r.chunk_id}: {r.text[:50]}...")
        """
        # 验证查询
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

        effective_top_k = top_k if top_k is not None else self.config.fusion_top_k

        logger.debug(f"HybridSearch：query='{query[:50]}...', top_k={effective_top_k}")

        # 步骤 1: 处理查询
        _t0 = time.monotonic()
        processed_query = self._process_query(query)
        _elapsed = (time.monotonic() - _t0) * 1000.0
        if trace is not None:
            trace.record_stage("query_processing", {
                "method": "query_processor",
                "original_query": query,
                "keywords": processed_query.keywords,
            }, elapsed_ms=_elapsed)

        # 合并显式过滤器与从查询中提取的过滤器
        merged_filters = self._merge_filters(processed_query.filters, filters)

        # 步骤 2: 运行检索
        dense_results, sparse_results, dense_error, sparse_error = self._run_retrievals(
            processed_query=processed_query,
            filters=merged_filters,
            trace=trace,
        )

        # 步骤 3: 处理降级场景
        used_fallback = False
        if dense_error and sparse_error:
            # 都失败 - 抛出异常
            raise RuntimeError(
                f"Both retrieval paths failed. "
                f"Dense error: {dense_error}. Sparse error: {sparse_error}"
            )
        elif dense_error:
            # 稠密失败，仅使用稀疏
            logger.warning(f"Dense retrieval failed, using sparse only: {dense_error}")
            used_fallback = True
            fused_results = sparse_results or []
        elif sparse_error:
            # 稀疏失败，仅使用稠密
            logger.warning(f"Sparse retrieval failed, using dense only: {sparse_error}")
            used_fallback = True
            fused_results = dense_results or []
        elif not dense_results and not sparse_results:
            # 都成功但返回空
            fused_results = []
        else:
            # 步骤 4: 融合结果
            fused_results = self._fuse_results(
                dense_results=dense_results or [],
                sparse_results=sparse_results or [],
                top_k=effective_top_k,
                trace=trace,
            )

        # 步骤 5: 应用融合后元数据过滤器（如有）
        if merged_filters and self.config.metadata_filter_post:
            fused_results = self._apply_metadata_filters(fused_results, merged_filters)

        # 步骤 6: 限制为 top_k
        final_results = fused_results[:effective_top_k]

        logger.debug(f"HybridSearch：返回 {len(final_results)} 个结果")

        if return_details:
            return HybridSearchResult(
                results=final_results,
                dense_results=dense_results,
                sparse_results=sparse_results,
                dense_error=dense_error,
                sparse_error=sparse_error,
                used_fallback=used_fallback,
                processed_query=processed_query,
            )

        return final_results

    def _process_query(self, query: str) -> ProcessedQuery:
        """使用 QueryProcessor 处理原始查询。

        参数：
            query: 原始查询字符串。

        返回：
            包含关键词和过滤器的 ProcessedQuery。
        """
        if self.query_processor is None:
            # 降级：创建基本的 ProcessedQuery
            logger.warning("No QueryProcessor configured, using basic tokenization")
            keywords = query.split()
            return ProcessedQuery(
                original_query=query,
                keywords=keywords,
                filters={},
            )

        return self.query_processor.process(query)

    def _merge_filters(
        self,
        query_filters: Dict[str, Any],
        explicit_filters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """合并从查询中提取的过滤器与显式过滤器。

        显式过滤器优先于从查询中提取的过滤器。

        参数：
            query_filters: QueryProcessor 从查询中提取的过滤器。
            explicit_filters: 显式传递给 search() 的过滤器。

        返回：
            合并后的过滤器字典。
        """
        merged = query_filters.copy() if query_filters else {}
        if explicit_filters:
            merged.update(explicit_filters)
        return merged

    def _run_retrievals(
        self,
        processed_query: ProcessedQuery,
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> Tuple[
        Optional[List[RetrievalResult]],
        Optional[List[RetrievalResult]],
        Optional[str],
        Optional[str],
    ]:
        """运行稠密和稀疏检索。

        如果配置则并行运行，否则顺序运行。

        参数：
            processed_query: 包含关键词的处理后查询。
            filters: 要应用的过滤器。
            trace: 可选的 TraceContext。

        返回：
            (dense_results, sparse_results, dense_error, sparse_error) 的元组。
        """
        dense_results: Optional[List[RetrievalResult]] = None
        sparse_results: Optional[List[RetrievalResult]] = None
        dense_error: Optional[str] = None
        sparse_error: Optional[str] = None

        # 确定要运行什么
        run_dense = (
            self.config.enable_dense
            and self.dense_retriever is not None
        )
        run_sparse = (
            self.config.enable_sparse
            and self.sparse_retriever is not None
            and processed_query.keywords  # 稀疏检索需要关键词
        )

        if not run_dense and not run_sparse:
            # 没有什么要运行的
            if self.dense_retriever is None and self.sparse_retriever is None:
                dense_error = "No retriever configured"
                sparse_error = "No retriever configured"
            return dense_results, sparse_results, dense_error, sparse_error

        if self.config.parallel_retrieval and run_dense and run_sparse:
            # 并行运行
            dense_results, sparse_results, dense_error, sparse_error = (
                self._run_parallel_retrievals(processed_query, filters, trace)
            )
        else:
            # 顺序运行
            if run_dense:
                dense_results, dense_error = self._run_dense_retrieval(
                    processed_query.original_query, filters, trace
                )

            if run_sparse:
                sparse_results, sparse_error = self._run_sparse_retrieval(
                    processed_query.keywords, filters, trace
                )

        return dense_results, sparse_results, dense_error, sparse_error

    def _run_parallel_retrievals(
        self,
        processed_query: ProcessedQuery,
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> Tuple[
        Optional[List[RetrievalResult]],
        Optional[List[RetrievalResult]],
        Optional[str],
        Optional[str],
    ]:
        """使用 ThreadPoolExecutor 并行运行稠密和稀疏检索。

        参数：
            processed_query: 处理后的查询。
            filters: 要应用的过滤器。
            trace: 可选的 TraceContext。

        返回：
            (dense_results, sparse_results, dense_error, sparse_error) 的元组。
        """
        dense_results: Optional[List[RetrievalResult]] = None
        sparse_results: Optional[List[RetrievalResult]] = None
        dense_error: Optional[str] = None
        sparse_error: Optional[str] = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            # 提交稠密检索
            futures['dense'] = executor.submit(
                self._run_dense_retrieval,
                processed_query.original_query,
                filters,
                trace,
            )

            # 提交稀疏检索
            futures['sparse'] = executor.submit(
                self._run_sparse_retrieval,
                processed_query.keywords,
                filters,
                trace,
            )

            # 收集结果
            for name, future in futures.items():
                try:
                    results, error = future.result(timeout=30)
                    if name == 'dense':
                        dense_results = results
                        dense_error = error
                    else:
                        sparse_results = results
                        sparse_error = error
                except Exception as e:
                    error_msg = f"{name} retrieval failed with exception: {e}"
                    logger.error(error_msg)
                    if name == 'dense':
                        dense_error = error_msg
                    else:
                        sparse_error = error_msg

        return dense_results, sparse_results, dense_error, sparse_error

    def _run_dense_retrieval(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> Tuple[Optional[List[RetrievalResult]], Optional[str]]:
        """运行带错误处理的稠密检索。

        参数：
            query: 原始查询字符串。
            filters: 要应用的过滤器。
            trace: 可选的 TraceContext。

        返回：
            (results, error) 的元组。成功时 error 为 None。
        """
        if self.dense_retriever is None:
            return None, "Dense retriever not configured"

        try:
            _t0 = time.monotonic()
            results = self.dense_retriever.retrieve(
                query=query,
                top_k=self.config.dense_top_k,
                filters=filters,
                trace=trace,
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0
            if trace is not None:
                trace.record_stage("dense_retrieval", {
                    "method": "dense",
                    "provider": getattr(self.dense_retriever, 'provider_name', 'unknown'),
                    "top_k": self.config.dense_top_k,
                    "result_count": len(results) if results else 0,
                    "chunks": _snapshot_results(results),
                }, elapsed_ms=_elapsed)
            return results, None
        except Exception as e:
            error_msg = f"Dense retrieval error: {e}"
            logger.error(error_msg)
            if trace is not None:
                trace.record_stage("dense_retrieval", {
                    "method": "dense",
                    "error": error_msg,
                    "result_count": 0,
                })
            return None, error_msg

    def _run_sparse_retrieval(
        self,
        keywords: List[str],
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> Tuple[Optional[List[RetrievalResult]], Optional[str]]:
        """运行带错误处理的稀疏检索。

        参数：
            keywords: QueryProcessor 提供的关键词列表。
            filters: 要应用的过滤器。
            trace: 可选的 TraceContext。

        返回：
            (results, error) 的元组。成功时 error 为 None。
        """
        if self.sparse_retriever is None:
            return None, "Sparse retriever not configured"

        if not keywords:
            return [], None  # 没有关键词，返回空（不是错误）

        try:
            # 如果过滤器中存在则提取集合
            collection = filters.get('collection') if filters else None

            _t0 = time.monotonic()
            results = self.sparse_retriever.retrieve(
                keywords=keywords,
                top_k=self.config.sparse_top_k,
                collection=collection,
                trace=trace,
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0
            if trace is not None:
                trace.record_stage("sparse_retrieval", {
                    "method": "bm25",
                    "keyword_count": len(keywords),
                    "top_k": self.config.sparse_top_k,
                    "result_count": len(results) if results else 0,
                    "chunks": _snapshot_results(results),
                }, elapsed_ms=_elapsed)
            return results, None
        except Exception as e:
            error_msg = f"Sparse retrieval error: {e}"
            logger.error(error_msg)
            return None, error_msg

    def _fuse_results(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int,
        trace: Optional[Any],
    ) -> List[RetrievalResult]:
        """使用 RRF 融合稠密和稀疏结果。

        参数：
            dense_results: 稠密检索的结果。
            sparse_results: 稀疏检索的结果。
            top_k: 融合后返回的结果数量。
            trace: 可选的 TraceContext。

        返回：
            融合并排序后的 RetrievalResult 列表。
        """
        if self.fusion is None:
            # 降级：交错结果（简单的 round-robin）
            logger.warning("No fusion configured, using simple interleave")
            return self._interleave_results(dense_results, sparse_results, top_k)

        # 为 RRF 构建排名列表
        ranking_lists = []
        if dense_results:
            ranking_lists.append(dense_results)
        if sparse_results:
            ranking_lists.append(sparse_results)

        if not ranking_lists:
            return []

        if len(ranking_lists) == 1:
            # 只有一个来源，无需融合
            return ranking_lists[0][:top_k]

        _t0 = time.monotonic()
        fused = self.fusion.fuse(
            ranking_lists=ranking_lists,
            top_k=top_k,
            trace=trace,
        )
        _elapsed = (time.monotonic() - _t0) * 1000.0
        if trace is not None:
            trace.record_stage("fusion", {
                "method": "rrf",
                "input_lists": len(ranking_lists),
                "top_k": top_k,
                "result_count": len(fused),
                "chunks": _snapshot_results(fused),
            }, elapsed_ms=_elapsed)
        return fused

    def _interleave_results(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """当未配置融合时的简单交错降级。

        参数：
            dense_results: 稠密检索的结果。
            sparse_results: 稀疏检索的结果。
            top_k: 返回的最大结果数量。

        返回：
            交错后按 chunk_id 去重的结果。
        """
        seen_ids = set()
        interleaved = []

        d_idx, s_idx = 0, 0
        while len(interleaved) < top_k and (d_idx < len(dense_results) or s_idx < len(sparse_results)):
            # 在稠密和稀疏之间交替
            if d_idx < len(dense_results):
                r = dense_results[d_idx]
                d_idx += 1
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    interleaved.append(r)

            if len(interleaved) >= top_k:
                break

            if s_idx < len(sparse_results):
                r = sparse_results[s_idx]
                s_idx += 1
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    interleaved.append(r)

        return interleaved

    def _apply_metadata_filters(
        self,
        results: List[RetrievalResult],
        filters: Dict[str, Any],
    ) -> List[RetrievalResult]:
        """对结果应用元数据过滤器（融合后降级）。

        这是一个备用过滤机制，用于底层存储不完全支持过滤器语法的场景。

        参数：
            results: 要过滤的结果。
            filters: 要应用的过滤条件。

        返回：
            过滤后的结果。
        """
        if not filters:
            return results

        filtered = []
        for result in results:
            if self._matches_filters(result.metadata, filters):
                filtered.append(result)

        return filtered

    def _matches_filters(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> bool:
        """检查元数据是否匹配所有过滤条件。

        参数：
            metadata: 结果元数据。
            filters: 过滤条件。

        返回：
            如果所有过滤器都匹配则返回 True，否则返回 False。
        """
        for key, value in filters.items():
            if key == "collection":
                # 集合可能在不同的元数据键中
                meta_collection = (
                    metadata.get("collection")
                    or metadata.get("source_collection")
                )
                if meta_collection != value:
                    return False
            elif key == "doc_type":
                if metadata.get("doc_type") != value:
                    return False
            elif key == "tags":
                # 标签是列表 - 检查交集
                meta_tags = metadata.get("tags", [])
                if not isinstance(value, list):
                    value = [value]
                if not set(meta_tags) & set(value):
                    return False
            elif key == "source_path":
                # 路径部分匹配
                source = metadata.get("source_path", "")
                if value not in source:
                    return False
            else:
                # 通用精确匹配
                if metadata.get(key) != value:
                    return False

        return True


def create_hybrid_search(
    settings: Optional[Settings] = None,
    query_processor: Optional[QueryProcessor] = None,
    dense_retriever: Optional[DenseRetriever] = None,
    sparse_retriever: Optional[SparseRetriever] = None,
    fusion: Optional[RRFFusion] = None,
) -> HybridSearch:
    """用于创建带默认组件的 HybridSearch 的工厂函数。

    这是一个方便函数，可在未提供时创建带默认 RRFFusion 的 HybridSearch。

    参数：
        settings: 应用设置。
        query_processor: QueryProcessor 实例。
        dense_retriever: DenseRetriever 实例。
        sparse_retriever: SparseRetriever 实例。
        fusion: RRFFusion 实例。为 None 时创建默认的 k=60。

    返回：
        配置好的 HybridSearch 实例。

    示例：
        >>> hybrid = create_hybrid_search(
        ...     settings=settings,
        ...     query_processor=QueryProcessor(),
        ...     dense_retriever=dense_retriever,
        ...     sparse_retriever=sparse_retriever,
        ... )
    """
    # 如果未提供则创建默认融合
    if fusion is None:
        from src.core.query_engine.fusion import RRFFusion
        rrf_k = 60
        if settings is not None:
            retrieval_config = getattr(settings, 'retrieval', None)
            if retrieval_config is not None:
                rrf_k = getattr(retrieval_config, 'rrf_k', 60)
        fusion = RRFFusion(k=rrf_k)

    return HybridSearch(
        settings=settings,
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion=fusion,
    )
