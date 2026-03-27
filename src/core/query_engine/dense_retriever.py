"""Dense Retriever for semantic search using vector embeddings.

本模块实现了 DenseRetriever 组件，通过 embedding 查询并从向量存储中检索相似 chunk 来执行语义搜索。
它是混合搜索引擎中的 Dense 路由。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.libs.embedding.base_embedding import BaseEmbedding
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """使用 embedding 进行语义搜索的 Dense 检索器。

    本类通过以下步骤执行语义检索：
    1. 使用配置的 embedding 客户端对查询进行 embedding
    2. 在向量存储中查询相似向量
    3. 返回标准化的 RetrievalResult 对象

    应用的设计原则：
    - 可插拔：通过依赖注入接受 embedding_client 和 vector_store
    - 配置驱动：默认 top_k 从 settings.retrieval.dense_top_k 读取
    - 可观察：支持可选的 TraceContext 用于可观测性集成
    - 快速失败：尽早验证输入并给出清晰的错误信息
    - 类型安全：返回标准化的 RetrievalResult 对象

    属性：
        embedding_client: 用于查询向量化的 embedding 提供方
        vector_store: 用于相似性搜索的向量存储
        default_top_k: 默认返回结果数量

    示例：
        >>> from src.libs.embedding.embedding_factory import EmbeddingFactory
        >>> from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        >>>
        >>> settings = Settings.load('config/settings.yaml')
        >>> embedding_client = EmbeddingFactory.create(settings)
        >>> vector_store = VectorStoreFactory.create(settings)
        >>>
        >>> retriever = DenseRetriever(
        ...     settings=settings,
        ...     embedding_client=embedding_client,
        ...     vector_store=vector_store
        ... )
        >>> results = retriever.retrieve("What is RAG?", top_k=5)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_client: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None,
        default_top_k: int = 10,
    ) -> None:
        """使用依赖项初始化 DenseRetriever。

        参数：
            settings: 应用配置，用于在未提供时提取 default_top_k
            embedding_client: 用于查询向量化的 embedding 提供方
                              实际的检索操作必需
            vector_store: 用于相似性搜索的向量存储
                          实际的检索操作必需
            default_top_k: 默认返回结果数量（默认：10）
                           可从 settings.retrieval.dense_top_k 覆盖

        异常：
            ValueError: 当需要时 embedding_client 或 vector_store 为 None

        注意：
            依赖项可用于测试（使用 mock）或生产环境（使用工厂的真实实现）
        """
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        
        # 如果配置中有，则从 settings 提取 default_top_k
        self.default_top_k = default_top_k
        if settings is not None:
            retrieval_config = getattr(settings, 'retrieval', None)
            if retrieval_config is not None:
                self.default_top_k = getattr(
                    retrieval_config, 'dense_top_k', default_top_k
                )

        logger.info(
            f"DenseRetriever 已初始化，default_top_k={self.default_top_k}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """检索与查询语义相似的 chunk。

        参数：
            query: 搜索查询字符串，不能为空
            top_k: 返回的最大结果数量，为 None 时使用 default_top_k
            filters: 可选的元数据过滤器（如 {"collection": "api-docs"}）
            trace: 可选的 TraceContext 用于可观测性（为 Stage F 预留）

        返回：
            RetrievalResult 对象列表，按相似度降序排列
            每个结果包含 chunk_id、score、text 和 metadata

        异常：
            ValueError: 查询为空或无效
            RuntimeError: embedding_client 或 vector_store 未配置，
                          或检索操作失败

        示例：
            >>> results = retriever.retrieve("How to configure Azure OpenAI?")
            >>> for result in results:
            ...     print(f"[{result.score:.2f}] {result.chunk_id}: {result.text[:50]}...")
        """
        # 验证输入
        self._validate_query(query)
        self._validate_dependencies()

        # 如果未指定则使用默认 top_k
        effective_top_k = top_k if top_k is not None else self.default_top_k

        logger.debug(f"检索查询='{query[:50]}...', top_k={effective_top_k}")

        # 步骤 1: 对查询进行 embedding
        try:
            query_vectors = self.embedding_client.embed([query], trace=trace)
            query_vector = query_vectors[0]
        except Exception as e:
            raise RuntimeError(
                f"查询 embedding 失败: {e}。"
                "请检查 embedding 客户端配置和连接。"
            ) from e

        # 步骤 2: 查询向量存储
        try:
            raw_results = self.vector_store.query(
                vector=query_vector,
                top_k=effective_top_k,
                filters=filters,
                trace=trace,
            )
        except Exception as e:
            raise RuntimeError(
                f"向量存储查询失败: {e}。"
                "请检查向量存储配置和数据可用性。"
            ) from e

        # 步骤 3: 转换为 RetrievalResult 对象
        results = self._transform_results(raw_results)

        logger.debug(f"查询返回 {len(results)} 个结果")
        return results
    
    def _validate_query(self, query: str) -> None:
        """验证查询字符串。

        参数：
            query: 要验证的查询字符串

        异常：
            ValueError: 查询为空或不是字符串
        """
        if not isinstance(query, str):
            raise ValueError(
                f"Query must be a string, got {type(query).__name__}"
            )
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
    
    def _validate_dependencies(self) -> None:
        """验证所需的依赖项是否已配置。

        异常：
            RuntimeError: embedding_client 或 vector_store 为 None
        """
        if self.embedding_client is None:
            raise RuntimeError(
                "DenseRetriever requires an embedding_client. "
                "Provide one during initialization or via setter."
            )
        if self.vector_store is None:
            raise RuntimeError(
                "DenseRetriever requires a vector_store. "
                "Provide one during initialization or via setter."
            )
    
    def _transform_results(
        self,
        raw_results: List[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """将原始向量存储结果转换为 RetrievalResult 对象。

        参数：
            raw_results: 向量存储查询的原始结果
                         每个结果应包含：id、score、text、metadata

        返回：
            RetrievalResult 对象列表
        """
        results = []
        for raw in raw_results:
            try:
                result = RetrievalResult(
                    chunk_id=str(raw.get('id', '')),
                    score=float(raw.get('score', 0.0)),
                    text=str(raw.get('text', '')),
                    metadata=raw.get('metadata', {}),
                )
                results.append(result)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"转换结果 {raw.get('id', 'unknown')} 失败: {e}。"
                    "跳过此结果。"
                )
                continue

        return results


def create_dense_retriever(
    settings: Settings,
    embedding_client: Optional[BaseEmbedding] = None,
    vector_store: Optional[BaseVectorStore] = None,
) -> DenseRetriever:
    """工厂函数，用于创建具有可选依赖注入的 DenseRetriever。

    本函数通过自动从工厂创建依赖项（如果未提供）来简化 DenseRetriever 的创建。

    参数：
        settings: 应用配置
        embedding_client: 可选的预配置 embedding 客户端
                          为 None 时从 EmbeddingFactory 创建
        vector_store: 可选的预配置向量存储
                      为 None 时从 VectorStoreFactory 创建

    返回：
        配置好的 DenseRetriever 实例

    示例：
        >>> settings = Settings.load('config/settings.yaml')
        >>> retriever = create_dense_retriever(settings)
    """
    # 延迟导入以避免循环依赖
    if embedding_client is None:
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        embedding_client = EmbeddingFactory.create(settings)

    if vector_store is None:
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        vector_store = VectorStoreFactory.create(settings)

    return DenseRetriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
