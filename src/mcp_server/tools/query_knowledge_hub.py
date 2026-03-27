"""MCP 工具: query_knowledge_hub

该工具通过 MCP 协议提供知识检索能力。
它结合混合搜索（稠密 + 稀疏 + RRF 融合）和可选的重排序
来查找相关文档，并返回带引用的格式化结果。

使用方法:
    工具名: query_knowledge_hub
    输入模式:
        - query (string, required): 搜索查询
        - top_k (integer, optional): 返回结果数 (默认: 5)
        - collection (string, optional): 限制在特定集合中搜索
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp import types

from src.core.response.response_builder import ResponseBuilder, MCPToolResponse
from src.core.settings import load_settings, resolve_path, Settings
from src.core.trace import TraceContext, TraceCollector
from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.reranker import CoreReranker

logger = logging.getLogger(__name__)


# 工具元数据
TOOL_NAME = "query_knowledge_hub"
TOOL_DESCRIPTION = """搜索知识库以查找相关文档。

该工具使用混合搜索（语义 + 关键词）找到与你的查询最相关的文档。
结果包含供参考的来源引用。

参数:
- query: 你的搜索问题或关键词
- top_k: 最大结果数（默认: 5）
- collection: 限制在特定文档集合中搜索
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "要为其查找相关文档的搜索查询或问题。",
        },
        "top_k": {
            "type": "integer",
            "description": "返回的最大结果数。",
            "default": 5,
            "minimum": 1,
            "maximum": 20,
        },
        "collection": {
            "type": "string",
            "description": "可选的集合名称，用于限制搜索范围。",
        },
    },
    "required": ["query"],
}


@dataclass
class QueryKnowledgeHubConfig:
    """query_knowledge_hub 工具的配置。

    属性:
        default_top_k: 未指定时的默认返回结果数
        max_top_k: 最大的 top_k 值
        default_collection: 未指定时的默认集合
        enable_rerank: 是否应用重排序
    """
    default_top_k: int = 5
    max_top_k: int = 20
    default_collection: str = "default"
    enable_rerank: bool = True


class QueryKnowledgeHubTool:
    """用于知识库查询的 MCP 工具。

    此类封装了 query_knowledge_hub 工具的逻辑，
    协调混合搜索（HybridSearch）和重排序器（Reranker）以生成格式化结果。

    设计原则:
    - 延迟初始化: 组件在首次使用时创建
    - 错误恢复能力: 优雅处理搜索/重排序失败
    - 可配置性: 所有参数来自 settings.yaml

    示例:
        >>> tool = QueryKnowledgeHubTool(settings)
        >>> result = await tool.execute(query="Azure 配置", top_k=5)
        >>> print(result.content)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[QueryKnowledgeHubConfig] = None,
        hybrid_search: Optional[HybridSearch] = None,
        reranker: Optional[CoreReranker] = None,
        response_builder: Optional[ResponseBuilder] = None,
    ) -> None:
        """初始化 QueryKnowledgeHubTool。

        参数:
            settings: 应用程序设置。如果为 None，从默认路径加载。
            config: 工具配置。如果为 None，使用默认值。
            hybrid_search: 可选的预配置 HybridSearch 实例。
            reranker: 可选的预配置 CoreReranker 实例。
            response_builder: 可选的预配置 ResponseBuilder 实例。
        """
        self._settings = settings
        self.config = config or QueryKnowledgeHubConfig()
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._embedding_client = None
        self._response_builder = response_builder or ResponseBuilder()
        
        # Track initialization state        self._initialized = False
        self._current_collection: Optional[str] = None
    
    @property
    def settings(self) -> Settings:
        """获取设置，如有必要则加载。"""
        if self._settings is None:
            self._settings = load_settings()
        return self._settings
    
    def _ensure_initialized(self, collection: str) -> None:
        """确保为给定集合初始化搜索组件。

        缓存策略（平衡速度与新鲜度）:
        - **完全缓存**（无状态，永不过时）: 嵌入客户端、
          重排序器、查询处理器、设置。
        - **缓存直到集合变更**: 向量存储（ChromaDB
          PersistentClient 从 SQLite 读取——能看到其他
          进程写入的数据）、稠密检索器、混合搜索。
        - **每次查询自动刷新**: BM25 稀疏索引——
          ``SparseRetriever._ensure_index_loaded()`` 总是从
          磁盘重新加载，所以缓存的 SparseRetriever 对象是可以的。

        仅当 *collection* 变更时我们才会销毁并重建。

        参数:
            collection: 目标集合名称。
        """
        # 始终重建 vector_store 和检索器组件，以便其他进程（例如 Dashboard）
        # 导入的数据立即可见，无需重启 MCP 服务器。
        logger.info(f"Initializing query components for collection: {collection}")
        
        # 导入 here to avoid circular imports and allow lazy loading
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.core.query_engine.reranker import create_core_reranker
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        
        # === Fully cached components (stateless, never go stale) ===
        if self._embedding_client is None:
            self._embedding_client = EmbeddingFactory.create(self.settings)
        
        if self._reranker is None:
            self._reranker = create_core_reranker(settings=self.settings)
        
        # === Rebuild for new collection ===
        # ChromaDB PersistentClient uses SQLite under the hood —
        # concurrent readers see committed writes from other processes
        # (dashboard ingestion), so caching the client is safe.
        vector_store = VectorStoreFactory.create(
            self.settings,
            collection_name=collection,
        )
        
        dense_retriever = create_dense_retriever(
            settings=self.settings,
            embedding_client=self._embedding_client,
            vector_store=vector_store,
        )
        
        # BM25Indexer 仅保存索引目录路径；SparseRetriever
        # 在每次搜索时调用 _ensure_index_loaded()，该函数总是
        # 从磁盘重新加载——所以它能获取 dashboard 写入的数据。
        bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
        sparse_retriever = create_sparse_retriever(
            settings=self.settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection
        
        query_processor = QueryProcessor()
        self._hybrid_search = create_hybrid_search(
            settings=self.settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )
        
        self._current_collection = collection
        self._initialized = True
        logger.info(f"Query components initialized for collection: {collection}")
    
    async def execute(
        self,
        query: str,
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
    ) -> MCPToolResponse:
        """执行 query_knowledge_hub 工具。

        参数:
            query: 搜索查询字符串。
            top_k: 返回的最大结果数。
            collection: 目标集合名称。

        返回:
            MCPToolResponse，包含格式化的内容和引用。

        抛出异常:
            ValueError: 如果查询为空或无效。
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Apply defaults
        effective_top_k = min(
            top_k or self.config.default_top_k,
            self.config.max_top_k
        )
        effective_collection = collection or self.config.default_collection
        
        logger.info(
            f"Executing query_knowledge_hub: query='{query[:50]}...', "
            f"top_k={effective_top_k}, collection={effective_collection}"
        )
        
        trace = TraceContext(trace_type="query")
        trace.metadata["query"] = query[:200]
        trace.metadata["top_k"] = effective_top_k
        trace.metadata["collection"] = effective_collection
        trace.metadata["source"] = "mcp"

        try:
            # 初始化 components for collection
            # 运行 blocking I/O (embedding API, ChromaDB, BM25) in a thread
            # to avoid blocking the async event loop / MCP stdio transport
            import time as _time
            _init_t0 = _time.monotonic()
            await asyncio.to_thread(self._ensure_initialized, effective_collection)
            _init_elapsed = (_time.monotonic() - _init_t0) * 1000.0
            trace.record_stage("initialization", {
                "collection": effective_collection,
                "cold_start": _init_elapsed > 500,  # >500ms ≈ cold
            }, elapsed_ms=_init_elapsed)
            
            # Perform hybrid search (blocking: embedding API + DB queries)
            results = await asyncio.to_thread(
                self._perform_search, query, effective_top_k, trace,
            )
            
            # Apply reranking if enabled (may call LLM API)
            if self.config.enable_rerank and results:
                results = await asyncio.to_thread(
                    self._apply_rerank, query, results, effective_top_k, trace,
                )
            
            # 构建 response
            response = self._response_builder.build(
                results=results,
                query=query,
                collection=effective_collection,
            )
            
            # 存储 final results in trace for dashboard display
            trace.metadata["final_results"] = [
                {
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "text": r.text or "",
                    "source": r.metadata.get("source_path", r.metadata.get("source", "")),
                    "title": r.metadata.get("title", ""),
                }
                for r in results
            ]

            logger.info(
                f"query_knowledge_hub completed: {len(results)} results, "
                f"is_empty={response.is_empty}"
            )
            
            TraceCollector().collect(trace)
            return response
            
        except Exception as e:
            import traceback
            error_stack = traceback.format_exc()
            logger.error(f"query_knowledge_hub failed: {e}")
            logger.error(f"Full stack trace:\n{error_stack}")
            TraceCollector().collect(trace)
            # 返回 error response
            return self._build_error_response(query, effective_collection, str(e))
    
    def _perform_search(
        self,
        query: str,
        top_k: int,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """执行混合搜索。

        参数:
            query: 搜索查询。
            top_k: 最大结果数。
            trace: 可选的 TraceContext，用于可观测性。

        返回:
            RetrievalResult 列表。
        """
        if self._hybrid_search is None:
            raise RuntimeError("HybridSearch not initialized")
        
        # 为重排序使用更大的初始检索数量
        initial_top_k = top_k * 2 if self.config.enable_rerank else top_k
        
        try:
            results = self._hybrid_search.search(
                query=query,
                top_k=initial_top_k,
                filters=None,
                trace=trace,
                return_details=False,
            )
            return results if isinstance(results, list) else results.results
        except Exception as e:
            logger.warning(f"Hybrid search failed: {e}")
            return []
    
    def _apply_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """对搜索结果应用重排序。

        参数:
            query: 原始查询。
            results: 要重排序的搜索结果。
            top_k: 最终结果数。
            trace: 可选的 TraceContext，用于可观测性。

        返回:
            重排序后的结果（如果重排序失败则返回原始顺序）。
        """
        if self._reranker is None or not self._reranker.is_enabled:
            return results[:top_k]
        
        try:
            rerank_result = self._reranker.rerank(
                query=query,
                results=results,
                top_k=top_k,
                trace=trace,
            )
            
            if rerank_result.used_fallback:
                logger.warning(
                    f"Reranker fallback: {rerank_result.fallback_reason}"
                )
            
            return rerank_result.results
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return results[:top_k]
    
    def _build_error_response(
        self,
        query: str,
        collection: str,
        error_message: str,
    ) -> MCPToolResponse:
        """构建错误响应。

        参数:
            query: 原始查询。
            collection: 目标集合。
            error_message: 错误描述。

        返回:
            指示错误的 MCPToolResponse。
        """
        content = f"## 查询失败\n\n"
        content += f"查询: **{query}**\n"
        content += f"集合: `{collection}`\n\n"
        content += f"**错误信息:** {error_message}\n\n"
        content += "请检查:\n"
        content += "- 数据库连接是否正常\n"
        content += "- 集合是否已创建并包含数据\n"
        content += "- 配置文件是否正确\n"
        
        return MCPToolResponse(
            content=content,
            citations=[],
            metadata={
                "query": query,
                "collection": collection,
                "error": error_message,
            },
            is_empty=True,
        )


# 模块级工具实例（延迟初始化）
_tool_instance: Optional[QueryKnowledgeHubTool] = None


def get_tool_instance(settings: Optional[Settings] = None) -> QueryKnowledgeHubTool:
    """获取或创建工具实例。

    参数:
        settings: 可选的初始化设置。

    返回:
        QueryKnowledgeHubTool 实例。
    """
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = QueryKnowledgeHubTool(settings=settings)
    return _tool_instance


async def query_knowledge_hub_handler(
    query: str,
    top_k: int = 5,
    collection: Optional[str] = None,
) -> types.CallToolResult:
    """MCP 工具注册的处理器函数。

    此函数向 ProtocolHandler 注册，并在 MCP 客户端
    调用 query_knowledge_hub 工具时被调用。

    支持多模态响应 - 如果搜索结果包含图像，
    响应将包含 ImageContent 块以及 TextContent。

    参数:
        query: 搜索查询字符串。
        top_k: 最大结果数。
        collection: 可选的集合名称。

    返回:
        MCP CallToolResult，包含内容块（文本和可选图像）。
    """
    tool = get_tool_instance()
    
    try:
        response = await tool.execute(
            query=query,
            top_k=top_k,
            collection=collection,
        )
        
        # Use to_mcp_content() which handles multimodal (text + images)
        content_blocks = response.to_mcp_content()
        
        return types.CallToolResult(
            content=content_blocks,
            isError=response.is_empty and "error" in response.metadata,
        )
        
    except ValueError as e:
        # 无效的参数
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"参数错误: {e}",
                )
            ],
            isError=True,
        )
    except Exception as e:
        # 内部 error - 记录详细堆栈
        import traceback
        error_stack = traceback.format_exc()
        logger.exception(f"query_knowledge_hub handler error: {e}")
        logger.error(f"Full stack trace:\n{error_stack}")
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"内部错误：查询处理失败\n\n错误详情：{str(e)}\n\n堆栈跟踪:\n```\n{error_stack}\n```",
                )
            ],
            isError=True,
        )


def register_tool(protocol_handler) -> None:
    """向协议处理器注册 query_knowledge_hub 工具。

    参数:
        protocol_handler: 要注册到的 ProtocolHandler 实例。
    """
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=query_knowledge_hub_handler,
    )
    logger.info(f"Registered MCP tool: {TOOL_NAME}")
