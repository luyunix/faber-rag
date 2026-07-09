"""Reusable query service for REST API endpoints.

This module keeps the FastAPI layer thin and centralizes construction of the
hybrid search stack. It intentionally mirrors the MCP query tool's behavior
while returning API-friendly objects.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.response.response_builder import MCPToolResponse, ResponseBuilder
from src.core.settings import Settings, load_settings, resolve_path
from src.core.trace import TraceCollector, TraceContext
from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class QueryExecutionResult:
    """Result returned by :class:`QueryService`."""

    response: MCPToolResponse
    results: List[RetrievalResult]
    trace: TraceContext


class QueryService:
    """Construct and reuse query components for API requests.

    Embedding clients and rerankers are cached because they are stateless and
    expensive to initialize. Hybrid search is cached per collection so repeated
    UI queries do not rebuild Chroma/BM25/retriever objects every time.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings
        self._embedding_client: Any = None
        self._reranker: Any = None
        self._hybrid_search_by_collection: Dict[str, Any] = {}
        self._response_builder = ResponseBuilder()

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    def reload(self) -> None:
        """Drop cached settings and components."""
        self._settings = None
        self._embedding_client = None
        self._reranker = None
        self._hybrid_search_by_collection.clear()

    def get_hybrid_search(self, collection: str) -> Any:
        """Return a cached HybridSearch instance for *collection*."""
        collection = collection or "default"
        cached = self._hybrid_search_by_collection.get(collection)
        if cached is not None:
            return cached

        logger.info("Initializing API query components for collection: %s", collection)

        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        if self._embedding_client is None:
            self._embedding_client = EmbeddingFactory.create(self.settings)

        vector_store = VectorStoreFactory.create(
            self.settings,
            collection_name=collection,
        )
        dense_retriever = create_dense_retriever(
            settings=self.settings,
            embedding_client=self._embedding_client,
            vector_store=vector_store,
        )
        bm25_indexer = BM25Indexer(
            index_dir=str(resolve_path(f"data/db/bm25/{collection}"))
        )
        sparse_retriever = create_sparse_retriever(
            settings=self.settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection

        hybrid_search = create_hybrid_search(
            settings=self.settings,
            query_processor=QueryProcessor(),
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )
        self._hybrid_search_by_collection[collection] = hybrid_search
        return hybrid_search

    def get_reranker(self) -> Any:
        """Return the cached core reranker."""
        if self._reranker is None:
            from src.core.query_engine.reranker import create_core_reranker

            self._reranker = create_core_reranker(settings=self.settings)
        return self._reranker

    def search(
        self,
        query: str,
        top_k: int,
        collection: str = "default",
        trace: Optional[TraceContext] = None,
        apply_rerank: bool = True,
    ) -> List[RetrievalResult]:
        """Run hybrid search and optional rerank."""
        hybrid_search = self.get_hybrid_search(collection)
        reranker = self.get_reranker() if apply_rerank else None
        has_reranker = bool(
            reranker and getattr(reranker, "is_enabled", False)
        )
        initial_top_k = top_k * 2 if has_reranker else top_k

        t0 = time.monotonic()
        results = hybrid_search.search(
            query=query,
            top_k=initial_top_k,
            filters=None,
            trace=trace,
            return_details=False,
        )
        results = results if isinstance(results, list) else results.results
        search_elapsed = (time.monotonic() - t0) * 1000.0

        if trace is not None:
            trace.record_stage(
                "search",
                {
                    "result_count": len(results),
                    "initial_top_k": initial_top_k,
                },
                elapsed_ms=search_elapsed,
            )

        if has_reranker and results:
            t0 = time.monotonic()
            rerank_result = reranker.rerank(
                query=query,
                results=results,
                top_k=top_k,
                trace=trace,
            )
            results = rerank_result.results
            if trace is not None:
                trace.record_stage(
                    "rerank",
                    {
                        "used_fallback": rerank_result.used_fallback,
                        "fallback_reason": (
                            rerank_result.fallback_reason
                            if rerank_result.used_fallback
                            else None
                        ),
                    },
                    elapsed_ms=(time.monotonic() - t0) * 1000.0,
                )
        else:
            results = results[:top_k]

        return results

    def execute_query(
        self,
        query: str,
        top_k: int = 10,
        collection: str = "default",
        source: str = "direct_api",
    ) -> QueryExecutionResult:
        """Run a full query and build an API/MCP compatible response."""
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")

        trace = TraceContext(trace_type="query")
        trace.metadata["query"] = query[:200]
        trace.metadata["top_k"] = top_k
        trace.metadata["collection"] = collection
        trace.metadata["source"] = source

        t0 = time.monotonic()
        self.get_hybrid_search(collection)
        init_elapsed = (time.monotonic() - t0) * 1000.0
        trace.record_stage(
            "initialization",
            {
                "collection": collection,
                "cold_start": init_elapsed > 500,
            },
            elapsed_ms=init_elapsed,
        )

        results = self.search(
            query=query,
            top_k=top_k,
            collection=collection,
            trace=trace,
            apply_rerank=True,
        )

        t0 = time.monotonic()
        response = self._response_builder.build(
            results=results,
            query=query,
            collection=collection,
        )
        trace.record_stage(
            "build_response",
            {
                "is_empty": response.is_empty,
                "citation_count": len(response.citations),
            },
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
        )

        trace.metadata["final_results"] = [
            {
                "chunk_id": r.chunk_id,
                "score": round(r.score, 4),
                "text_preview": _preview(r.text or ""),
                "text_length": len(r.text or ""),
                "source": r.metadata.get("source_path", r.metadata.get("source", "")),
                "title": r.metadata.get("title", ""),
            }
            for r in results
        ]
        TraceCollector().collect(trace)

        return QueryExecutionResult(response=response, results=results, trace=trace)


def _preview(value: str, limit: int = 500) -> str:
    """Return a compact single-line preview for trace metadata."""
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"
