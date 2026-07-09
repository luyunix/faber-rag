"""FastAPI 服务器 - 为 Faber RAG UI 前端提供 REST API。

提供以下端点：
- 配置服务：获取组件配置、集合统计
- 数据服务：文档/片段/图片的 CRUD 操作
- 追踪服务：查询追踪历史
- 评估服务：运行评估
- MCP 服务：MCP 查询代理
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.core.settings import load_settings, resolve_path, Settings, SettingsError
from src.api.services.data_service import DataService
from src.api.services.trace_service import TraceService
from src.api.services.config_service import ConfigService
from src.api.services.query_service import QueryService

logger = logging.getLogger(__name__)


# ── Pydantic Models ─────────────────────────────────────────────

class ComponentInfo(BaseModel):
    name: str
    provider: str
    model: str
    extra: Dict[str, Any]


class CollectionStats(BaseModel):
    collection: str
    chunk_count: int


class Document(BaseModel):
    source_path: str
    source_hash: str
    collection: str
    chunk_count: int
    image_count: int
    processed_at: str


class Chunk(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]


class Image(BaseModel):
    image_id: str
    file_path: str
    page: int
    doc_hash: str


class DeleteResult(BaseModel):
    success: bool
    chunks_deleted: int
    images_deleted: int
    errors: List[str]


class Stage(BaseModel):
    stage: str
    timestamp: str
    elapsed_ms: float
    data: Dict[str, Any]


class Trace(BaseModel):
    trace_id: str
    trace_type: str
    started_at: str
    elapsed_ms: Optional[float]
    metadata: Dict[str, Any]
    stages: List[Stage]


class QueryResult(BaseModel):
    query: str
    elapsed_ms: float
    metrics: Dict[str, float]
    retrieved_chunk_ids: List[str]
    generated_answer: Optional[str]


class EvaluationReport(BaseModel):
    evaluator_name: str
    query_count: int
    total_elapsed_ms: float
    aggregate_metrics: Dict[str, float]
    query_results: List[QueryResult]


class MCPQueryRequest(BaseModel):
    query: str
    top_k: int = 10
    collection: str = "default"
    server_url: str = "http://localhost:8080"


class DirectQueryRequest(BaseModel):
    query: str
    top_k: int = 10
    collection: str = "default"


class EvaluationRunRequest(BaseModel):
    backend: str = "ragas"
    golden_set_path: str = ""
    top_k: int = 10
    collection: Optional[str] = None
    user_answers: Optional[Dict[str, Any]] = None
    test_set_content: Optional[str] = None


class EvaluateTraceRequest(BaseModel):
    query: str
    meta: Dict[str, Any]
    user_answer: str


# ── FastAPI App Lifecycle ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    logger.info("Starting Faber RAG API Server...")
    yield
    logger.info("Shutting down Faber RAG API Server...")


app = FastAPI(
    title="Faber RAG API",
    description="REST API for Faber RAG Dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Services ─────────────────────────────────────────────

_data_service: Optional[DataService] = None
_trace_service: Optional[TraceService] = None
_config_service: Optional[ConfigService] = None
_query_service: Optional[QueryService] = None


def get_data_service() -> DataService:
    global _data_service
    if _data_service is None:
        _data_service = DataService()
    return _data_service


def get_trace_service() -> TraceService:
    global _trace_service
    if _trace_service is None:
        _trace_service = TraceService()
    return _trace_service


def get_config_service() -> ConfigService:
    global _config_service
    if _config_service is None:
        _config_service = ConfigService()
    return _config_service


def get_query_service() -> QueryService:
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service


# ── Health Check ─────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """健康检查端点。"""
    return {"status": "ok", "service": "faber-rag-api"}


# ── Config Endpoints ─────────────────────────────────────────────

@app.get("/api/config/components", response_model=List[ComponentInfo])
async def get_components():
    """获取组件配置列表。"""
    try:
        config = get_config_service()
        cards = config.get_component_cards()
        return [
            ComponentInfo(
                name=card.name,
                provider=card.provider,
                model=card.model,
                extra=card.extra
            )
            for card in cards
        ]
    except Exception as e:
        logger.exception("Failed to get components")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config/collections/stats", response_model=List[CollectionStats])
async def get_collection_stats():
    """获取集合统计信息。"""
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        settings = load_settings()
        persist_dir = str(resolve_path(settings.vector_store.persist_directory))
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )

        stats = []
        for col in client.list_collections():
            name = col if isinstance(col, str) else col.name
            collection = client.get_collection(name)
            stats.append(CollectionStats(
                collection=name,
                chunk_count=collection.count()
            ))

        return stats
    except Exception as e:
        logger.exception("Failed to get collection stats")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config/settings")
async def get_raw_settings():
    """获取原始 settings.yaml 内容（保留环境变量引用格式）。"""
    try:
        import yaml
        settings_path = resolve_path("config/settings.yaml")
        with settings_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except Exception as e:
        logger.exception("Failed to get raw settings")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/config/settings")
async def update_raw_settings(request_body: Dict[str, Any]):
    """更新 settings.yaml 内容。

    接收完整的 settings dict，验证后写回文件。
    部分配置（如 retrieval 参数）会在下次查询时自动生效，
    LLM/Embedding 等Provider变更需要重启服务。
    """
    try:
        import yaml

        # 验证配置格式
        try:
            Settings.from_dict(request_body)
        except SettingsError as e:
            raise HTTPException(status_code=400, detail=f"配置验证失败: {e}")

        # 写回 yaml 文件
        settings_path = resolve_path("config/settings.yaml")
        with settings_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                request_body,
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
                indent=2,
            )

        # 清除 ConfigService 缓存，使新配置尽快生效
        get_config_service().reload()
        get_query_service().reload()

        return {"success": True, "message": "配置已保存"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update settings")
        raise HTTPException(status_code=500, detail=str(e))


# ── Data Endpoints ───────────────────────────────────────────────

@app.get("/api/data/collections", response_model=List[str])
async def list_collections():
    """获取所有集合名称。"""
    try:
        service = get_data_service()
        return service.list_collections()
    except Exception as e:
        logger.exception("Failed to list collections")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/documents", response_model=List[Document])
async def list_documents(collection: Optional[str] = Query(None)):
    """获取文档列表。"""
    try:
        service = get_data_service()
        docs = service.list_documents(collection)
        return [
            Document(
                source_path=d["source_path"],
                source_hash=d["source_hash"],
                collection=d.get("collection", ""),
                chunk_count=d["chunk_count"],
                image_count=d["image_count"],
                processed_at=d.get("processed_at", "")
            )
            for d in docs
        ]
    except Exception as e:
        logger.exception("Failed to list documents")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/documents/{source_hash}/chunks", response_model=List[Chunk])
async def get_chunks(source_hash: str, collection: Optional[str] = Query(None)):
    """获取文档的片段列表。"""
    try:
        service = get_data_service()
        chunks = service.get_chunks(source_hash, collection)
        return [
            Chunk(
                id=c["id"],
                text=c["text"],
                metadata=c.get("metadata", {})
            )
            for c in chunks
        ]
    except Exception as e:
        logger.exception("Failed to get chunks")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/documents/{source_hash}/images", response_model=List[Image])
async def get_images(source_hash: str, collection: Optional[str] = Query(None)):
    """获取文档的图片列表。"""
    try:
        service = get_data_service()
        imgs = service.get_images(source_hash, collection)
        return [
            Image(
                image_id=img["image_id"],
                file_path=img["file_path"],
                page=img.get("page", 0),
                doc_hash=img.get("doc_hash", "")
            )
            for img in imgs
        ]
    except Exception as e:
        logger.exception("Failed to get images")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/data/documents", response_model=DeleteResult)
async def delete_document(
    source_path: str = Query(...),
    collection: str = Query("default"),
    source_hash: Optional[str] = Query(None)
):
    """删除文档。"""
    try:
        service = get_data_service()
        result = service.delete_document(source_path, collection, source_hash)
        return DeleteResult(
            success=result.success,
            chunks_deleted=result.chunks_deleted,
            images_deleted=result.images_deleted,
            errors=result.errors
        )
    except Exception as e:
        logger.exception("Failed to delete document")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/data/reset")
async def reset_all():
    """清空所有数据。"""
    try:
        service = get_data_service()
        result = service.reset_all()
        return {
            "collections_deleted": result["collections_deleted"],
            "bm25_cleared": result["bm25_cleared"],
            "images_cleared": result["images_cleared"],
            "integrity_cleared": result["integrity_cleared"],
            "traces_cleared": result["traces_cleared"],
            "errors": result["errors"]
        }
    except Exception as e:
        logger.exception("Failed to reset all data")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingestion/upload")
async def upload_and_ingest(
    file: UploadFile = File(...),
    collection: str = Form("default")
):
    """上传并处理文档。"""
    from src.ingestion.pipeline import IngestionPipeline
    from src.core.trace import TraceContext, TraceCollector

    tmp_path: Optional[str] = None
    trace: Optional[TraceContext] = None
    try:
        settings = load_settings()

        # 保存上传的文件到临时位置
        suffix = Path(file.filename or "upload").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)

        # 运行处理流程
        trace = TraceContext(trace_type="ingestion")
        trace.metadata["source_path"] = file.filename or ""
        trace.metadata["collection"] = collection
        trace.metadata["source"] = "dashboard"

        def _run_pipeline():
            pipeline = IngestionPipeline(settings, collection=collection)
            try:
                return pipeline.run(
                    file_path=tmp_path,
                    trace=trace,
                    on_progress=lambda stage, current, total: None,  # 简化，暂不报告进度
                    source_path=file.filename or tmp_path,
                )
            finally:
                pipeline.close()

        result = await asyncio.to_thread(_run_pipeline)

        # 记录追踪
        try:
            TraceCollector().collect(trace)
            logger.info("Ingestion trace collected: %s", trace.trace_id)
        except Exception as e:
            logger.exception("Failed to collect ingestion trace: %s", e)

        return {
            "doc_id": result.doc_id if result else "",
            "success": bool(result and result.success),
            "error": None if result and result.success else (result.error if result else "empty result"),
        }
    except Exception as e:
        logger.exception("Failed to upload and ingest")
        if trace is not None:
            try:
                TraceCollector().collect(trace)
            except Exception:
                logger.exception("Failed to collect failed ingestion trace")
        return {
            "doc_id": "",
            "success": False,
            "error": str(e)
        }
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ── Trace Endpoints ─────────────────────────────────────────────

@app.get("/api/traces", response_model=List[Trace])
async def list_traces(
    trace_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """获取追踪列表。"""
    try:
        service = get_trace_service()
        traces = service.list_traces(trace_type, limit)
        return [
            Trace(
                trace_id=t["trace_id"],
                trace_type=t.get("trace_type", ""),
                started_at=t["started_at"],
                elapsed_ms=t.get("elapsed_ms"),
                metadata=t.get("metadata", {}),
                stages=[
                    Stage(
                        stage=s["stage"],
                        timestamp=s["timestamp"],
                        elapsed_ms=s.get("elapsed_ms", 0),
                        data=s.get("data", {})
                    )
                    for s in t.get("stages", [])
                ]
            )
            for t in traces
        ]
    except Exception as e:
        logger.exception("Failed to list traces")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/traces/{trace_id}", response_model=Trace)
async def get_trace(trace_id: str):
    """获取单个追踪详情。"""
    try:
        service = get_trace_service()
        trace = service.get_trace(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="Trace not found")

        return Trace(
            trace_id=trace["trace_id"],
            trace_type=trace.get("trace_type", ""),
            started_at=trace["started_at"],
            elapsed_ms=trace.get("elapsed_ms"),
            metadata=trace.get("metadata", {}),
            stages=[
                Stage(
                    stage=s["stage"],
                    timestamp=s["timestamp"],
                    elapsed_ms=s.get("elapsed_ms", 0),
                    data=s.get("data", {})
                )
                for s in trace.get("stages", [])
            ]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get trace")
        raise HTTPException(status_code=500, detail=str(e))


# ── Evaluation Endpoints ─────────────────────────────────────────

@app.post("/api/evaluation/run", response_model=EvaluationReport)
async def run_evaluation(request: EvaluationRunRequest):
    """运行评估。"""
    tmp_path: Optional[str] = None
    try:
        import json
        from dataclasses import replace as dc_replace
        from pathlib import Path as PathlibPath

        from src.core.settings import EvaluationSettings
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory
        from src.observability.evaluation.eval_runner import EvalRunner

        settings = load_settings()

        # 解析用户答案
        answers_dict = {
            int(k): v
            for k, v in (request.user_answers or {}).items()
            if str(k).isdigit()
        }

        # 覆盖评估设置
        eval_settings = EvaluationSettings(
            enabled=True,
            provider=request.backend,
            metrics=[],
        )
        settings_with_override = dc_replace(settings, evaluation=eval_settings)

        evaluator = EvaluatorFactory.create(settings_with_override)

        # 加载测试集
        golden_path = None
        if request.test_set_content:
            # 前端直接传入 JSON 内容，写入临时文件
            import tempfile
            tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
            tmp_file.write(request.test_set_content)
            tmp_file.flush()
            tmp_path = tmp_file.name
            golden_path = PathlibPath(tmp_path)
        elif request.golden_set_path:
            golden_path = PathlibPath(request.golden_set_path)
            if not golden_path.exists():
                raise HTTPException(status_code=400, detail="Golden test set file not found")
        else:
            raise HTTPException(status_code=400, detail="请提供 Golden Test Set 路径或直接上传 JSON 内容")

        target_collection = request.collection or "default"
        query_service = get_query_service()
        hybrid_search = query_service.get_hybrid_search(target_collection)
        reranker = query_service.get_reranker()
        if not reranker.is_enabled:
            reranker = None

        # 运行评估
        runner = EvalRunner(
            settings=settings,
            hybrid_search=hybrid_search,
            evaluator=evaluator,
            answer_overrides=answers_dict if answers_dict else None,
            reranker=reranker,
        )

        report = await asyncio.to_thread(
            runner.run,
            test_set_path=golden_path,
            top_k=request.top_k,
            collection=target_collection,
        )

        result = EvaluationReport(
            evaluator_name=report.evaluator_name,
            query_count=len(report.query_results),
            total_elapsed_ms=report.total_elapsed_ms,
            aggregate_metrics=report.aggregate_metrics,
            query_results=[
                QueryResult(
                    query=qr.query,
                    elapsed_ms=qr.elapsed_ms,
                    metrics=qr.metrics,
                    retrieved_chunk_ids=qr.retrieved_chunk_ids,
                    generated_answer=qr.generated_answer
                )
                for qr in report.query_results
            ]
        )
        return result
    except HTTPException:
        raise
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to run evaluation")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file if created from test_set_content
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/api/evaluation/evaluate-trace")
async def evaluate_single_trace(request: EvaluateTraceRequest):
    """评估单个查询追踪。"""
    try:
        from dataclasses import replace as dc_replace
        from src.core.settings import EvaluationSettings
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory

        settings = load_settings()

        # 覆盖评估设置为 Ragas
        ragas_eval = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=["faithfulness", "answer_relevancy", "context_precision"],
        )
        settings = dc_replace(settings, evaluation=ragas_eval)
        evaluator = EvaluatorFactory.create(settings)

        collection = request.meta.get("collection", "default")
        top_k = request.meta.get("top_k", 10)

        def _search_and_evaluate():
            chunks = get_query_service().search(
                query=request.query,
                top_k=top_k,
                collection=collection,
                apply_rerank=True,
            )
            if not chunks:
                return {"error": "未检索到片段"}
            metrics = evaluator.evaluate(
                query=request.query,
                retrieved_chunks=chunks,
                generated_answer=request.user_answer,
            )
            return {
                "metrics": metrics,
                "answer_used": request.user_answer,
            }

        return await asyncio.to_thread(_search_and_evaluate)
    except Exception as e:
        logger.exception("Failed to evaluate trace")
        return {"error": str(e)}


@app.get("/api/evaluation/history", response_model=List[EvaluationReport])
async def get_evaluation_history(
    limit: int = Query(10, ge=1, le=100)
):
    """获取评估历史。"""
    try:
        history_path = resolve_path("logs/eval_history.jsonl")
        if not history_path.exists():
            return []

        reports = []
        with history_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        reports.append(EvaluationReport(
                            evaluator_name=data.get("evaluator_name", ""),
                            query_count=data.get("query_count", 0),
                            total_elapsed_ms=data.get("total_elapsed_ms", 0),
                            aggregate_metrics=data.get("aggregate_metrics", {}),
                            query_results=[
                                QueryResult(
                                    query=qr.get("query", ""),
                                    elapsed_ms=qr.get("elapsed_ms", 0),
                                    metrics=qr.get("metrics", {}),
                                    retrieved_chunk_ids=qr.get("retrieved_chunk_ids", []),
                                    generated_answer=qr.get("generated_answer")
                                )
                                for qr in data.get("query_results", [])
                            ]
                        ))
                    except json.JSONDecodeError:
                        continue

        return reports[-limit:] if len(reports) > limit else reports
    except Exception as e:
        logger.exception("Failed to get evaluation history")
        raise HTTPException(status_code=500, detail=str(e))


# ── Direct Query Endpoint ───────────────────────────────────────

@app.post("/api/query")
async def direct_query(request: DirectQueryRequest):
    """直接执行查询，绕过 MCP 协议。"""
    try:
        collection = request.collection or "default"
        top_k = request.top_k or 10
        query = request.query.strip()

        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        result = await asyncio.to_thread(
            get_query_service().execute_query,
            query=query,
            top_k=top_k,
            collection=collection,
            source="direct_api",
        )
        response = result.response

        # Return format matching frontend expectations
        return {
            "content": [
                {"type": "text", "text": response.content}
            ],
            "is_error": False,
            "text": response.content,
            "raw": {
                "citations": [c.to_dict() for c in response.citations],
                "metadata": response.metadata,
                "is_empty": response.is_empty,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Direct query failed")
        return {
            "content": [{"type": "text", "text": f"查询失败: {str(e)}"}],
            "is_error": True,
            "text": f"查询失败: {str(e)}",
            "raw": {"error": str(e)}
        }


# ── MCP Endpoints ───────────────────────────────────────────────

@app.post("/api/mcp/query")
async def mcp_query(request: MCPQueryRequest):
    """通过 MCP 协议执行查询。"""
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "query_knowledge_hub",
                    "arguments": {
                        "query": request.query,
                        "top_k": request.top_k,
                        "collection": request.collection,
                    }
                }
            }

            async with session.post(
                f"{request.server_url}/call",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                data = await response.json()

                if "error" in data:
                    return {
                        "content": [],
                        "is_error": True,
                        "text": data["error"].get("message", "Unknown error"),
                        "raw": data
                    }

                result = data.get("result", {})
                content = result.get("content", [])
                text = result.get("text", "")

                return {
                    "content": content,
                    "is_error": False,
                    "text": text,
                    "raw": data
                }
    except Exception as e:
        logger.exception("MCP query failed")
        return {
            "content": [],
            "is_error": True,
            "text": str(e),
            "raw": {}
        }


@app.get("/api/mcp/health")
async def mcp_health_check(server_url: str = Query("http://localhost:8080")):
    """检查 MCP 服务器健康状态。"""
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{server_url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    return {"healthy": True}
                else:
                    return {"healthy": False, "status": response.status}
    except Exception as e:
        logger.exception(f"MCP health check failed: {e}")
        return {"healthy": False, "error": str(e)}


# ── Main ───────────────────────────────────────────────────────

def main():
    """启动 FastAPI 服务器。"""
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
