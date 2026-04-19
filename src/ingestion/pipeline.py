"""Faber RAG 的摄取 Pipeline 编排器。

本模块实现主要的 Pipeline，编排完整的文档摄取流程：
    1. 文件完整性检查（SHA256 跳过检查）
    2. 文档加载（PDF → Document）
    3. 分块（Document → Chunks）
    4. 转换（优化 + 丰富 + 描述）
    5. 编码（稠密 + 稀疏向量）
    6. 存储（VectorStore + BM25 索引 + ImageStorage）

设计原则：
- 配置驱动：所有组件通过 settings.yaml 配置
- 可观测：记录进度和阶段完成情况
- 优雅降级：LLM 失败不会阻塞 Pipeline
- 幂等性：基于 SHA256 对未更改文件进行跳过
"""

from pathlib import Path
from typing import Callable, List, Optional, Dict, Any
import time

from src.core.settings import Settings, load_settings, resolve_path
from src.core.types import Document, Chunk
from src.core.trace.trace_context import TraceContext
from src.observability.logger import get_logger

# Libs layer imports
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

# Ingestion layer imports
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.storage.image_storage import ImageStorage

logger = get_logger(__name__)


class PipelineResult:
    """Pipeline 执行结果及详细统计信息。

    属性：
        success: Pipeline 是否成功完成
        file_path: 已处理文件的路径
        doc_id: 文档 ID (SHA256 哈希)
        chunk_count: 生成的分块数
        image_count: 处理的图片数
        vector_ids: 存储的向量 ID 列表
        error: Pipeline 失败时的错误信息
        stages: 各阶段名称及其单独结果的字典
    """

    def __init__(
        self,
        success: bool,
        file_path: str,
        doc_id: Optional[str] = None,
        chunk_count: int = 0,
        image_count: int = 0,
        vector_ids: Optional[List[str]] = None,
        error: Optional[str] = None,
        stages: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.file_path = file_path
        self.doc_id = doc_id
        self.chunk_count = chunk_count
        self.image_count = image_count
        self.vector_ids = vector_ids or []
        self.error = error
        self.stages = stages or {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便序列化。"""
        return {
            "success": self.success,
            "file_path": self.file_path,
            "doc_id": self.doc_id,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
            "vector_ids_count": len(self.vector_ids),
            "error": self.error,
            "stages": self.stages
        }


class IngestionPipeline:
    """文档摄取的主 Pipeline 编排器。

    此类协调摄取过程的所有阶段：
    - 文件完整性检查用于增量处理
    - 文档加载（带图片提取的 PDF）
    - 文本分块使用可配置的切分器
    - 分块优化（基于规则 + LLM）
    - 元数据丰富（基于规则 + LLM）
    - 图片描述（视觉 LLM）
    - 稠密嵌入（Azure text-embedding-ada-002）
    - 稀疏编码（BM25 词项统计）
    - 向量存储（ChromaDB）
    - BM25 索引构建

    示例：
        >>> from src.core.settings import load_settings
        >>> settings = load_settings("config/settings.yaml")
        >>> pipeline = IngestionPipeline(settings)
        >>> result = pipeline.run("documents/report.pdf", collection="contracts")
        >>> print(f"处理了 {result.chunk_count} 个分块")
    """

    def __init__(
        self,
        settings: Settings,
        collection: str = "default",
        force: bool = False
    ):
        """使用所有组件初始化 Pipeline。

        参数：
            settings: 来自 settings.yaml 的应用程序设置
            collection: 用于组织文档的集合名称
            force: 如果为 True，即使文件之前已处理也重新处理
        """
        self.settings = settings
        self.collection = collection
        self.force = force

        # 初始化所有组件
        logger.info("初始化摄取 Pipeline 组件...")

        # 阶段 1：文件完整性
        self.integrity_checker = SQLiteIntegrityChecker(db_path=str(resolve_path("data/db/ingestion_history.db")))
        logger.info("  ✓ FileIntegrityChecker 已初始化")

        # 阶段 2：加载器
        self.loader = PdfLoader(
            extract_images=True,
            image_storage_dir=str(resolve_path(f"data/images/{collection}"))
        )
        logger.info("  ✓ PdfLoader 已初始化")

        # 阶段 3：分块器
        self.chunker = DocumentChunker(settings)
        logger.info("  ✓ DocumentChunker 已初始化")

        # 阶段 4：转换
        self.chunk_refiner = ChunkRefiner(settings)
        logger.info(f"  ✓ ChunkRefiner 已初始化 (use_llm={self.chunk_refiner.use_llm})")

        self.metadata_enricher = MetadataEnricher(settings)
        logger.info(f"  ✓ MetadataEnricher 已初始化 (use_llm={self.metadata_enricher.use_llm})")

        self.image_captioner = ImageCaptioner(settings)
        has_vision = self.image_captioner.llm is not None
        logger.info(f"  ✓ ImageCaptioner 已初始化 (vision_enabled={has_vision})")

        # 阶段 5：编码器
        embedding = EmbeddingFactory.create(settings)
        batch_size = settings.ingestion.batch_size if settings.ingestion else 100
        self.dense_encoder = DenseEncoder(embedding, batch_size=batch_size)
        logger.info(f"  ✓ DenseEncoder 已初始化 (provider={settings.embedding.provider})")

        self.sparse_encoder = SparseEncoder()
        logger.info("  ✓ SparseEncoder 已初始化")

        self.batch_processor = BatchProcessor(
            dense_encoder=self.dense_encoder,
            sparse_encoder=self.sparse_encoder,
            batch_size=batch_size
        )
        logger.info(f"  ✓ BatchProcessor 已初始化 (batch_size={batch_size})")

        # 阶段 6：存储
        self.vector_upserter = VectorUpserter(settings, collection_name=collection)
        logger.info(f"  ✓ VectorUpserter 已初始化 (provider={settings.vector_store.provider}, collection={collection})")

        self.bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
        logger.info("  ✓ BM25Indexer 已初始化")

        self.image_storage = ImageStorage(
            db_path=str(resolve_path("data/db/image_index.db")),
            images_root=str(resolve_path("data/images"))
        )
        logger.info("  ✓ ImageStorage 已初始化")

        logger.info("Pipeline 初始化完成！")
    
    def run(
        self,
        file_path: str,
        trace: Optional[TraceContext] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        source_path: Optional[str] = None,
    ) -> PipelineResult:
        """在文件上执行完整的摄取 Pipeline。

        参数：
            file_path: 要处理的文件路径（如 PDF）
            trace: 用于可观察性的可选跟踪上下文
            on_progress: 可选的回调 ``(stage_name, current, total)``
                在每个 Pipeline 阶段完成时调用。*current* 是
                已完成阶段的基于 1 的索引；*total* 是
                阶段总数（当前为 6）。
            source_path: 原始文件路径（用于记录到数据库）。
                当 file_path 是临时文件时，传入原始文件名以保持可追溯性。

        返回：
            PipelineResult，包含成功状态和统计信息
        """
        file_path = Path(file_path)
        _source_path = source_path if source_path else str(file_path)
        stages: Dict[str, Any] = {}
        _total_stages = 6

        def _notify(stage_name: str, step: int) -> None:
            if on_progress is not None:
                on_progress(stage_name, step, _total_stages)

        logger.info(f"=" * 60)
        logger.info(f"🚀 启动 Pipeline: {file_path}")
        logger.info(f"   集合：{self.collection}")
        logger.info(f"   强制模式：{self.force}")
        logger.info("=" * 60)
        
        # 记录开始时间以计算总耗时
        _pipeline_start = time.monotonic()
        
        try:
            # ─────────────────────────────────────────────────────────────
            # 阶段 1：文件完整性检查
            # ─────────────────────────────────────────────────────────────
            logger.info("\n📋 阶段 1：文件完整性检查")
            _notify("integrity", 1)

            file_hash = self.integrity_checker.compute_sha256(str(file_path))
            logger.info(f"  文件哈希：{file_hash[:16]}...")

            if not self.force and self.integrity_checker.should_skip(file_hash):
                logger.info(f"  ⏭️  文件已处理，跳过（使用 force=True 重新处理）")
                return PipelineResult(
                    success=True,
                    file_path=str(file_path),
                    doc_id=file_hash,
                    stages={"integrity": {"skipped": True, "reason": "already_processed"}}
                )

            stages["integrity"] = {"file_hash": file_hash, "skipped": False}
            logger.info("  ✓ 文件需要处理")

            # ─────────────────────────────────────────────────────────────
            # 阶段 2：文档加载
            # ─────────────────────────────────────────────────────────────
            logger.info("\n📄 阶段 2：文档加载")
            _notify("load", 2)

            _t0 = time.monotonic()
            document = self.loader.load(str(file_path))
            _elapsed = (time.monotonic() - _t0) * 1000.0

            text_preview = document.text[:200].replace('\n', ' ') + "..." if len(document.text) > 200 else document.text
            image_count = len(document.metadata.get("images", []))

            logger.info(f"  文档 ID：{document.id}")
            logger.info(f"  文本长度：{len(document.text)} 字符")
            logger.info(f"  提取的图片：{image_count}")
            logger.info(f"  预览：{text_preview[:100]}...")

            stages["loading"] = {
                "doc_id": document.id,
                "text_length": len(document.text),
                "image_count": image_count
            }
            if trace is not None:
                trace.record_stage("load", {
                    "method": "markitdown",
                    "doc_id": document.id,
                    "text_length": len(document.text),
                    "image_count": image_count,
                    "text_preview": document.text,
                }, elapsed_ms=_elapsed)

            # ─────────────────────────────────────────────────────────────
            # 阶段 3：分块
            # ─────────────────────────────────────────────────────────────
            logger.info("\n✂️  阶段 3：文档分块")
            _notify("split", 3)

            logger.info(f"  输入：文档 ID={document.id}, 文本长度={len(document.text)}")
            _t0 = time.monotonic()
            chunks = self.chunker.split_document(document)
            _elapsed = (time.monotonic() - _t0) * 1000.0

            logger.info(f"  生成分块：{len(chunks)}")
            if chunks:
                logger.info(f"  第一个分块详情:")
                logger.info(f"    - ID: {chunks[0].id}")
                logger.info(f"    - 文本预览：{chunks[0].text[:100]}...")
                logger.info(f"    - 长度：{len(chunks[0].text)} 字符")
                logger.info(f"    - 元数据键：{list(chunks[0].metadata.keys())}")
                
                # 记录所有分块的 ID，方便调试
                for i, chunk in enumerate(chunks[:5]):  # 只显示前 5 个
                    logger.info(f"    - Chunk {i}: id={chunk.id}, len={len(chunk.text)}")
                if len(chunks) > 5:
                    logger.info(f"    ... 还有 {len(chunks) - 5} 个分块")

            stages["chunking"] = {
                "chunk_count": len(chunks),
                "avg_chunk_size": sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0
            }
            if trace is not None:
                trace.record_stage("split", {
                    "method": "recursive",
                    "chunk_count": len(chunks),
                    "avg_chunk_size": sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0,
                    "chunks": [
                        {
                            "chunk_id": c.id,
                            "text": c.text,
                            "char_len": len(c.text),
                            "chunk_index": c.metadata.get("chunk_index", i),
                        }
                        for i, c in enumerate(chunks)
                    ],
                }, elapsed_ms=_elapsed)

            # ─────────────────────────────────────────────────────────────
            # 阶段 4：转换 Pipeline
            # ─────────────────────────────────────────────────────────────
            logger.info("\n🔄 阶段 4：转换 Pipeline")
            _notify("transform", 4)

            # 4a：分块优化
            logger.info("  4a. 分块优化...")
            logger.info(f"      输入分块数：{len(chunks)}")
            _t0_transform = time.monotonic()
            # 优化前快照
            _pre_refine_texts = {c.id: c.text for c in chunks}
            chunks = self.chunk_refiner.transform(chunks, trace)
            refined_by_llm = sum(1 for c in chunks if c.metadata.get("refined_by") == "llm")
            refined_by_rule = sum(1 for c in chunks if c.metadata.get("refined_by") == "rule")
            logger.info(f"      LLM 优化：{refined_by_llm}, 规则优化：{refined_by_rule}")
            logger.info(f"      输出分块数：{len(chunks)}")

            # 4b：元数据丰富
            logger.info("  4b. 元数据丰富...")
            logger.info(f"      输入分块数：{len(chunks)}")
            chunks = self.metadata_enricher.transform(chunks, trace)
            enriched_by_llm = sum(1 for c in chunks if c.metadata.get("enriched_by") == "llm")
            enriched_by_rule = sum(1 for c in chunks if c.metadata.get("enriched_by") == "rule")
            logger.info(f"      LLM 丰富：{enriched_by_llm}, 规则丰富：{enriched_by_rule}")
            logger.info(f"      输出分块数：{len(chunks)}")

            # 4c：图片描述
            logger.info("  4c. 图片描述...")
            logger.info(f"      输入分块数：{len(chunks)}")
            chunks = self.image_captioner.transform(chunks, trace)
            captioned = sum(1 for c in chunks if c.metadata.get("image_captions"))
            logger.info(f"      有描述的分块：{captioned}")
            logger.info(f"      输出分块数：{len(chunks)}")

            stages["transform"] = {
                "chunk_refiner": {"llm": refined_by_llm, "rule": refined_by_rule},
                "metadata_enricher": {"llm": enriched_by_llm, "rule": enriched_by_rule},
                "image_captioner": {"captioned_chunks": captioned}
            }
            _elapsed_transform = (time.monotonic() - _t0_transform) * 1000.0
            if trace is not None:
                trace.record_stage("transform", {
                    "method": "refine+enrich+caption",
                    "refined_by_llm": refined_by_llm,
                    "refined_by_rule": refined_by_rule,
                    "enriched_by_llm": enriched_by_llm,
                    "enriched_by_rule": enriched_by_rule,
                    "captioned_chunks": captioned,
                    "chunks": [
                        {
                            "chunk_id": c.id,
                            "text_before": _pre_refine_texts.get(c.id, ""),
                            "text_after": c.text,
                            "char_len": len(c.text),
                            "refined_by": c.metadata.get("refined_by", ""),
                            "enriched_by": c.metadata.get("enriched_by", ""),
                            "title": c.metadata.get("title", ""),
                            "tags": c.metadata.get("tags", []),
                            "summary": c.metadata.get("summary", ""),
                        }
                        for c in chunks
                    ],
                }, elapsed_ms=_elapsed_transform)

            # ─────────────────────────────────────────────────────────────
            # 阶段 5：编码
            # ─────────────────────────────────────────────────────────────
            logger.info("\n🔢 阶段 5：编码")
            _notify("embed", 5)

            # 通过 BatchProcessor 处理
            logger.info(f"  准备编码 {len(chunks)} 个分块...")
            _t0 = time.monotonic()
            
            try:
                batch_result = self.batch_processor.process(chunks, trace)
                _elapsed = (time.monotonic() - _t0) * 1000.0

                dense_vectors = batch_result.dense_vectors
                sparse_stats = batch_result.sparse_stats
                successful_chunks = batch_result.successful_chunks
                failed_chunks = batch_result.failed_chunks

                logger.info(f"  ✅ 编码完成：成功={successful_chunks}, 失败={failed_chunks}")
                logger.info(f"  稠密向量：{len(dense_vectors)} (dim={len(dense_vectors[0]) if dense_vectors else 0})")
                logger.info(f"  稀疏统计：{len(sparse_stats)} 文档")
                
                # 如果有失败的批次，记录警告
                if failed_chunks > 0:
                    logger.warning(f"  ⚠️  有 {failed_chunks} 个分块编码失败，但继续处理")
                
                # 记录阶段数据到 stages（用于 Dashboard 显示）
                stages["encoding"] = {
                    "dense_vector_count": len(dense_vectors),
                    "dense_dimension": len(dense_vectors[0]) if dense_vectors else 0,
                    "sparse_doc_count": len(sparse_stats),
                    "successful_chunks": successful_chunks,
                    "failed_chunks": failed_chunks,
                }
                
                # 即使部分失败，也要记录 embed 阶段到 trace
                if trace is not None:
                    # 构建每个分块的编码详情（稠密和稀疏）
                    chunk_details = []
                    for idx, c in enumerate(chunks):
                        detail: dict = {
                            "chunk_id": c.id,
                            "char_len": len(c.text),
                        }
                        # 稠密：向量维度（对所有分块都一样，但每个确认）
                        if idx < len(dense_vectors):
                            detail["dense_dim"] = len(dense_vectors[idx])
                        # 稀疏：BM25 词项统计
                        # 注意：如果批次处理失败，sparse_stats 可能少于 chunks
                        if idx < len(sparse_stats):
                            ss = sparse_stats[idx]
                            detail["doc_length"] = ss.get("doc_length", 0)
                            detail["unique_terms"] = ss.get("unique_terms", 0)
                            # 按频率显示前 10 个词项以供检查
                            tf = ss.get("term_frequencies", {})
                            top_terms = sorted(tf.items(), key=lambda x: x[1], reverse=True)[:10]
                            detail["top_terms"] = [{"term": t, "freq": f} for t, f in top_terms]
                        else:
                            # 批次处理失败时的占位符
                            detail["doc_length"] = 0
                            detail["unique_terms"] = 0
                            detail["top_terms"] = []
                        chunk_details.append(detail)

                    trace.record_stage("embed", {
                        "method": "batch_processor",
                        "dense_vector_count": len(dense_vectors),
                        "dense_dimension": len(dense_vectors[0]) if dense_vectors else 0,
                        "sparse_doc_count": len(sparse_stats),
                        "successful_chunks": successful_chunks,
                        "failed_chunks": failed_chunks,
                        "chunks": chunk_details,
                    }, elapsed_ms=_elapsed)
                    
            except Exception as e:
                logger.error(f"  ❌ 编码阶段失败：{e}", exc_info=True)
                # 记录错误信息到 stages
                stages["encoding"] = {"error": str(e), "type": type(e).__name__}
                # 即使失败也要记录阶段到 trace
                if trace is not None:
                    trace.record_stage("embed", {
                        "method": "batch_processor",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "chunks_processed": 0,
                    }, elapsed_ms=(time.monotonic() - _t0) * 1000.0)
                raise
            
            # ─────────────────────────────────────────────────────────────
            # 阶段 6：存储
            # ─────────────────────────────────────────────────────────────
            logger.info("\n💾 阶段 6：存储")
            _notify("upsert", 6)

            try:
                # 6a：向量插入/更新
                logger.info("  6a. 向量存储 (ChromaDB)...")
                logger.info(f"      准备存储 {len(dense_vectors)} 个向量")
                _t0_storage = time.monotonic()
                vector_ids = self.vector_upserter.upsert(chunks, dense_vectors, trace)
                _elapsed_storage = (time.monotonic() - _t0_storage) * 1000.0
                logger.info(f"      ✅ 存储了 {len(vector_ids)} 个向量")
                if vector_ids:
                    logger.info(f"      第一个向量 ID: {vector_ids[0]}")
                
                # 记录成功信息到 stages
                stages["storage"] = {
                    "vector_count": len(vector_ids),
                    "bm25_docs": len(sparse_stats),
                    "images_indexed": len(document.metadata.get("images", [])),
                }
                
            except Exception as e:
                logger.error(f"  ❌ 向量存储失败：{e}", exc_info=True)
                # 记录错误信息到 stages
                stages["storage"] = {"error": str(e), "type": type(e).__name__}
                # 即使失败也要记录阶段到 trace
                if trace is not None:
                    trace.record_stage("upsert", {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "vectors_stored": 0,
                    }, elapsed_ms=(time.monotonic() - _t0_storage) * 1000.0)
                raise
            
            # 将 BM25 分块 ID 与 Chroma 向量 ID 对齐，以便 SparseRetriever
            # 可以在检索后在向量存储中查找 BM25 命中结果。
            logger.info(f"      关联 {len(sparse_stats)} 个 BM25 统计到向量 ID...")
            for stat, vid in zip(sparse_stats, vector_ids):
                stat["chunk_id"] = vid

            # 6b：BM25 索引
            logger.info("  6b. BM25 索引...")
            logger.info(f"      准备为 {len(sparse_stats)} 个文档构建索引")
            self.bm25_indexer.add_documents(
                sparse_stats,
                collection=self.collection,
                doc_id=document.id,
                trace=trace,
            )
            logger.info(f"      ✅ 完成索引构建")

            # 6c：在图片存储索引中注册图片
            # 注意：图片已由 PdfLoader 保存，我们只需要索引它们
            logger.info("  6c. 图片存储索引...")
            images = document.metadata.get("images", [])
            logger.info(f"      发现 {len(images)} 张图片")
            for img in images:
                img_path = Path(img["path"])
                if img_path.exists():
                    self.image_storage.register_image(
                        image_id=img["id"],
                        file_path=img_path,
                        collection=self.collection,
                        doc_hash=file_hash,
                        page_num=img.get("page", 0)
                    )
            logger.info(f"      ✅ 索引了 {len(images)} 张图片")
            
            # 更新 stages 中的详细信息
            stages["storage"].update({
                "vector_count": len(vector_ids),
                "bm25_docs": len(sparse_stats),
                "images_indexed": len(images)
            })
            
            _elapsed_storage = (time.monotonic() - _t0_storage) * 1000.0
            if trace is not None:
                # 每个分块的存储映射：chunk_id → vector_id
                chunk_storage = [
                    {
                        "chunk_id": c.id,
                        "vector_id": vector_ids[i] if i < len(vector_ids) else "—",
                        "collection": self.collection,
                        "store": "ChromaDB",
                    }
                    for i, c in enumerate(chunks)
                ]
                # 图片存储详情
                image_storage_details = [
                    {
                        "image_id": img["id"],
                        "file_path": str(img["path"]),
                        "page": img.get("page", 0),
                        "doc_hash": file_hash,
                    }
                    for img in images
                ]
                trace.record_stage("upsert", {
                    "dense_store": {
                        "backend": "ChromaDB",
                        "collection": self.collection,
                        "count": len(vector_ids),
                        "path": "data/db/chroma/",
                    },
                    "sparse_store": {
                        "backend": "BM25",
                        "collection": self.collection,
                        "count": len(sparse_stats),
                        "path": f"data/db/bm25/{self.collection}/",
                    },
                    "image_store": {
                        "backend": "ImageStorage (JSON index)",
                        "count": len(images),
                        "images": image_storage_details,
                    },
                    "chunk_mapping": chunk_storage,
                }, elapsed_ms=_elapsed_storage)

            # ─────────────────────────────────────────────────────────────
            # 标记成功
            # ─────────────────────────────────────────────────────────────
            self.integrity_checker.mark_success(file_hash, _source_path, self.collection)

            logger.info("\n" + "=" * 60)
            logger.info("✅ Pipeline 成功完成！")
            logger.info(f"   分块：{len(chunks)}")
            logger.info(f"   向量：{len(vector_ids)}")
            logger.info(f"   图片：{len(images)}")
            _total_time = (time.monotonic() - _pipeline_start) * 1000.0
            logger.info(f"   总耗时：{_total_time:.2f}ms")
            logger.info("=" * 60)

            return PipelineResult(
                success=True,
                file_path=str(file_path),
                doc_id=file_hash,
                chunk_count=len(chunks),
                image_count=len(images),
                vector_ids=vector_ids,
                stages=stages
            )

        except Exception as e:
            logger.error(f"❌ Pipeline 失败：{e}", exc_info=True)
            self.integrity_checker.mark_failed(file_hash, str(file_path), str(e))

            return PipelineResult(
                success=False,
                file_path=str(file_path),
                doc_id=file_hash if 'file_hash' in locals() else None,
                error=str(e),
                stages=stages
            )

    def close(self) -> None:
        """清理资源。"""
        self.image_storage.close()


def run_pipeline(
    file_path: str,
    settings_path: Optional[str] = None,
    collection: str = "default",
    force: bool = False
) -> PipelineResult:
    """运行 Pipeline 的便捷函数。

    参数：
        file_path: 要处理的文件路径
        settings_path: settings.yaml 的路径（默认：<repo>/config/settings.yaml）
        collection: 集合名称
        force: 强制重新处理

    返回：
        PipelineResult，包含执行详情
    """
    settings = load_settings(settings_path)
    pipeline = IngestionPipeline(settings, collection=collection, force=force)

    try:
        return pipeline.run(file_path)
    finally:
        pipeline.close()
