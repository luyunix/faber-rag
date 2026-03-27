"""使用 BM25 进行基于关键词搜索的稀疏检索器。

本模块实现 SparseRetriever 组件，使用 BM25 倒排索引执行基于关键词的搜索。
它在混合搜索引擎中形成稀疏路由，与 DenseRetriever 的语义搜索形成互补。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class SparseRetriever:
    """使用 BM25 基于关键词搜索的稀疏检索器。

    本类通过以下步骤执行基于关键词的检索：
    1. 使用关键词查询 BM25 索引以获取匹配的 chunk ID 和分数
    2. 使用 get_by_ids() 从向量存储中获取文本和元数据
    3. 返回标准化的 RetrievalResult 对象

    应用的设计原则：
    - 可插拔：通过依赖注入接受 bm25_indexer 和 vector_store。
    - 配置驱动：默认 top_k 和集合从设置中读取。
    - 可观测：接受可选的 TraceContext 用于可观测性集成。
    - 快速失败：尽早验证输入并给出清晰的错误信息。
    - 类型安全：返回标准化的 RetrievalResult 对象（与 DenseRetriever 相同）。

    属性：
        bm25_indexer: 用于关键词搜索的 BM25 索引器。
        vector_store: 用于获取文本和元数据的向量存储。
        default_top_k: 默认返回结果数量。
        default_collection: 要查询的默认 BM25 索引集合。

    示例：
        >>> from src.ingestion.storage.bm25_indexer import BM25Indexer
        >>> from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        >>>
        >>> settings = Settings.load('config/settings.yaml')
        >>> bm25_indexer = BM25Indexer(index_dir="data/db/bm25")
        >>> bm25_indexer.load("default")
        >>> vector_store = VectorStoreFactory.create(settings)
        >>>
        >>> retriever = SparseRetriever(
        ...     settings=settings,
        ...     bm25_indexer=bm25_indexer,
        ...     vector_store=vector_store
        ... )
        >>> results = retriever.retrieve(["RAG", "retrieval"], top_k=5)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        vector_store: Optional[BaseVectorStore] = None,
        default_top_k: int = 10,
        default_collection: str = "default",
    ) -> None:
        """使用依赖项初始化 SparseRetriever。

        参数：
            settings: 应用设置。用于在未提供时提取 default_top_k。
            bm25_indexer: 用于关键词搜索的 BM25 索引器。
                          实际的检索操作必需。
            vector_store: 用于获取文本和元数据的向量存储。
                          实际的检索操作必需。
            default_top_k: 默认返回结果数量（默认：10）。
                           可从 settings.retrieval.sparse_top_k 覆盖。
            default_collection: 默认 BM25 索引集合名称（默认："default"）。

        注意：
            依赖项可用于测试（使用 mock）或生产环境（使用来自工厂的真实实现）。
        """
        self.bm25_indexer = bm25_indexer
        self.vector_store = vector_store
        self.default_collection = default_collection

        # 从设置中提取 default_top_k（如果可用）
        self.default_top_k = default_top_k
        if settings is not None:
            retrieval_config = getattr(settings, 'retrieval', None)
            if retrieval_config is not None:
                self.default_top_k = getattr(
                    retrieval_config, 'sparse_top_k', default_top_k
                )

        logger.info(
            f"SparseRetriever 已初始化，default_top_k={self.default_top_k}, "
            f"default_collection='{self.default_collection}'"
        )

    def retrieve(
        self,
        keywords: List[str],
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """使用 BM25 检索与给定关键词匹配的块。

        参数：
            keywords: 要搜索的关键词列表（通常来自 QueryProcessor）。
            top_k: 返回的最大结果数量。为 None 时使用 default_top_k。
            collection: 要查询的 BM25 索引集合。为 None 时使用 default_collection。
            trace: 用于可观测性的可选 TraceContext（为 Stage F 预留）。

        返回：
            RetrievalResult 对象列表，按 BM25 分数降序排列。
            每个结果包含 chunk_id、score、text 和 metadata。

        异常：
            ValueError: 如果关键词列表为空。
            RuntimeError: 如果未配置 bm25_indexer 或 vector_store，
                          或检索操作失败。

        示例：
            >>> results = retriever.retrieve(["Azure", "OpenAI", "配置"])
            >>> for result in results:
            ...     print(f"[{result.score:.2f}] {result.chunk_id}: {result.text[:50]}...")
        """
        # 验证输入
        self._validate_keywords(keywords)
        self._validate_dependencies()

        # 如果未指定则使用默认值
        effective_top_k = top_k if top_k is not None else self.default_top_k
        effective_collection = collection if collection is not None else self.default_collection

        logger.debug(
            f"检索关键词={keywords[:5]}{'...' if len(keywords) > 5 else ''}, "
            f"top_k={effective_top_k}, collection='{effective_collection}'"
        )

        # 步骤 1: 确保索引已加载
        if not self._ensure_index_loaded(effective_collection):
            logger.warning(
                f"集合 '{effective_collection}' 的 BM25 索引不可用。"
                "返回空结果。"
            )
            return []

        # 步骤 2: 查询 BM25 索引
        try:
            bm25_results = self.bm25_indexer.query(
                query_terms=keywords,
                top_k=effective_top_k,
                trace=trace,
            )
        except Exception as e:
            raise RuntimeError(
                f"查询 BM25 索引失败: {e}。"
                "请检查索引可用性和查询词。"
            ) from e

        # 无匹配时提前返回
        if not bm25_results:
            logger.debug("BM25 查询未返回结果")
            return []

        # 步骤 3: 从向量存储获取文本和元数据
        chunk_ids = [r["chunk_id"] for r in bm25_results]
        try:
            records = self.vector_store.get_by_ids(chunk_ids, trace=trace)
        except Exception as e:
            raise RuntimeError(
                f"从向量存储获取记录失败: {e}。"
                "请检查向量存储配置和数据可用性。"
            ) from e

        # 步骤 4: 合并 BM25 分数与文本/元数据
        results = self._merge_results(bm25_results, records)

        logger.debug(f"检索到 {len(results)} 个关键词结果")
        return results

    def _validate_keywords(self, keywords: List[str]) -> None:
        """验证关键词列表。

        参数：
            keywords: 要验证的关键词列表。

        异常：
            ValueError: 如果 keywords 为空或不是列表。
        """
        if not isinstance(keywords, list):
            raise ValueError(
                f"Keywords must be a list, got {type(keywords).__name__}"
            )
        if not keywords:
            raise ValueError("Keywords list cannot be empty")
        # 过滤掉空字符串但允许调用继续
        # （空字符串不会匹配任何内容）

    def _validate_dependencies(self) -> None:
        """验证所需的依赖项是否已配置。

        异常：
            RuntimeError: 如果 bm25_indexer 或 vector_store 为 None。
        """
        if self.bm25_indexer is None:
            raise RuntimeError(
                "SparseRetriever requires a bm25_indexer. "
                "Provide one during initialization or via setter."
            )
        if self.vector_store is None:
            raise RuntimeError(
                "SparseRetriever requires a vector_store. "
                "Provide one during initialization or via setter."
            )

    def _ensure_index_loaded(self, collection: str) -> bool:
        """确保给定集合的 BM25 索引已加载。

        始终从磁盘重新加载，因为索引可能已被另一个进程更新
        （例如仪表板摄取）。加载速度很快（单个 JSON 文件读取）
        相对于整体查询而言。

        参数：
            collection: 要加载的集合名称。

        返回：
            如果索引已加载并准备好则返回 True，否则返回 False。
        """
        try:
            loaded = self.bm25_indexer.load(collection=collection)
            return loaded
        except Exception as e:
            logger.warning(f"加载集合 '{collection}' 的 BM25 索引失败: {e}")
            return False

    def _merge_results(
        self,
        bm25_results: List[Dict[str, Any]],
        records: List[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """将 BM25 分数与向量存储中的文本和元数据合并。

        参数：
            bm25_results: BM25 查询的结果，每个包含 'chunk_id' 和 'score'。
            records: 来自向量存储的记录，每个包含 'id'、'text'、'metadata'。

        返回：
            包含完整信息的 RetrievalResult 对象列表。
        """
        results = []

        for bm25_result, record in zip(bm25_results, records):
            chunk_id = bm25_result["chunk_id"]
            score = bm25_result["score"]

            # 处理记录未找到的情况
            if not record:
                logger.warning(
                    f"在向量存储中未找到 chunk_id='{chunk_id}' 的记录。"
                    "跳过此结果。"
                )
                continue

            # 验证记录包含预期字段
            text = record.get('text', '')
            metadata = record.get('metadata', {})

            try:
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    score=float(score),
                    text=str(text),
                    metadata=metadata,
                )
                results.append(result)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"为 chunk_id='{chunk_id}' 创建 RetrievalResult 失败: {e}。"
                    "跳过此结果。"
                )
                continue

        return results


def create_sparse_retriever(
    settings: Settings,
    bm25_indexer: Optional[BM25Indexer] = None,
    vector_store: Optional[BaseVectorStore] = None,
    index_dir: str = "data/db/bm25",
) -> SparseRetriever:
    """用于创建支持可选依赖注入的 SparseRetriever 的工厂函数。

    本函数通过自动从工厂创建依赖项（如果未提供）来简化 SparseRetriever 的创建。

    参数：
        settings: 应用设置。
        bm25_indexer: 可选的预配置 BM25 索引器。
                      为 None 时以默认 index_dir 创建。
        vector_store: 可选的预配置向量存储。
                      为 None 时从 VectorStoreFactory 创建。
        index_dir: BM25 索引文件目录（默认："data/db/bm25"）。

    返回：
        配置好的 SparseRetriever 实例。

    示例：
        >>> settings = Settings.load('config/settings.yaml')
        >>> retriever = create_sparse_retriever(settings)
    """
    # 延迟导入以避免循环依赖
    if bm25_indexer is None:
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        bm25_indexer = BM25Indexer(index_dir=index_dir)

    if vector_store is None:
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        vector_store = VectorStoreFactory.create(settings)

    return SparseRetriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
