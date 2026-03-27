"""跨存储的文档生命周期管理。

本模块提供用于列出、检查和删除文档的统一入口点，
涵盖所有四个存储后端（ChromaDB、BM25、
ImageStorage、FileIntegrityChecker）。

设计原则：
- 协调一致：一次调用可级联到所有相关存储。
- 故障安全：报告部分失败，但不会中止剩余存储的操作。
- 只读安全：列出/统计/详细信息方法不会修改数据。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 结果数据类
# ---------------------------------------------------------------------------

@dataclass
class DocumentInfo:
    """已摄取文档的摘要信息。"""

    source_path: str
    source_hash: str
    collection: Optional[str] = None
    chunk_count: int = 0
    image_count: int = 0
    processed_at: Optional[str] = None


@dataclass
class DocumentDetail(DocumentInfo):
    """扩展的文档信息，包含分块 ID 和图片 ID。"""

    chunk_ids: List[str] = field(default_factory=list)
    image_ids: List[str] = field(default_factory=list)


@dataclass
class DeleteResult:
    """delete_document 操作的结果。"""

    success: bool
    chunks_deleted: int = 0
    bm25_removed: bool = False
    images_deleted: int = 0
    integrity_removed: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class CollectionStats:
    """集合的聚合统计信息。"""

    collection: Optional[str] = None
    document_count: int = 0
    chunk_count: int = 0
    image_count: int = 0


# ---------------------------------------------------------------------------
# DocumentManager# ---------------------------------------------------------------------------

class DocumentManager:
    """跨所有存储后端协调文档生命周期。

    参数：
        chroma_store: ChromaStore 实例（向量存储）。
        bm25_indexer: BM25Indexer 实例（稀疏索引）。
        image_storage: ImageStorage 实例（图片文件 + SQLite 索引）。
        file_integrity: SQLiteIntegrityChecker 实例（摄取历史）。
    """

    def __init__(
        self,
        chroma_store: Any,
        bm25_indexer: Any,
        image_storage: Any,
        file_integrity: Any,
    ) -> None:
        self.chroma = chroma_store
        self.bm25 = bm25_indexer
        self.images = image_storage
        self.integrity = file_integrity

    # ------------------------------------------------------------------
    # list_documents    # ------------------------------------------------------------------

    def list_documents(
        self, collection: Optional[str] = None
    ) -> List[DocumentInfo]:
        """返回已摄取文档的列表。

        将完整性检查器的信息（source_path、
        hash、processed_at）与 ChromaDB 和 ImageStorage 的计数合并。

        参数：
            collection: 可选的集合过滤器。

        返回：
            ``DocumentInfo`` 对象列表。
        """
        records = self.integrity.list_processed(collection)

        docs: List[DocumentInfo] = []
        for rec in records:
            source_hash = rec["file_hash"]
            source_path = rec["file_path"]
            coll = rec.get("collection")

            # 统计 Chroma 中的分块数
            chunk_count = self._count_chunks(source_hash)

            # 统计图片数
            image_count = self._count_images(source_hash)

            docs.append(
                DocumentInfo(
                    source_path=source_path,
                    source_hash=source_hash,
                    collection=coll,
                    chunk_count=chunk_count,
                    image_count=image_count,
                    processed_at=rec.get("processed_at"),
                )
            )

        return docs

    # ------------------------------------------------------------------
    # get_document_detail    # ------------------------------------------------------------------

    def get_document_detail(self, doc_id: str) -> Optional[DocumentDetail]:
        """获取单个文档的详细信息。

        *doc_id* 与完整性检查器中存储的 ``source_hash`` 匹配。

        参数：
            doc_id: 文档的 source_hash。

        返回：
            ``DocumentDetail``，包含分块/图片 ID，如果未找到则返回 *None*。
        """
        # 查找完整性记录
        all_records = self.integrity.list_processed()
        record = None
        for rec in all_records:
            if rec["file_hash"] == doc_id:
                record = rec
                break

        if record is None:
            return None

        source_hash = record["file_hash"]

        # 从 Chroma 收集分块 ID
        chunk_ids = self._get_chunk_ids(source_hash)

        # 收集图片 ID
        image_ids = self._get_image_ids(source_hash)

        return DocumentDetail(
            source_path=record["file_path"],
            source_hash=source_hash,
            collection=record.get("collection"),
            chunk_count=len(chunk_ids),
            image_count=len(image_ids),
            processed_at=record.get("processed_at"),
            chunk_ids=chunk_ids,
            image_ids=image_ids,
        )

    # ------------------------------------------------------------------
    # delete_document    # ------------------------------------------------------------------

    def delete_document(
        self,
        source_path: str,
        collection: str = "default",
        source_hash: Optional[str] = None,
    ) -> DeleteResult:
        """从所有存储后端删除文档。

        在 ChromaDB、BM25、ImageStorage 和
        FileIntegrity 之间协调删除。部分失败被捕获在
        ``DeleteResult.errors`` 中，但不会阻止清理剩余的存储。

        文档通过其 *source_hash* 识别。未提供哈希值时，
        该方法尝试从文件计算；如果文件不存在，
        则回退到通过路径从完整性记录中查找哈希。

        参数：
            source_path: 文档的原始文件系统路径。
            collection: 文档所属的集合。
            source_hash: 预计算的 SHA-256 哈希。提供时，
                该方法不会尝试读取源文件。

        返回：
            ``DeleteResult``，汇总清理的内容。
        """
        result = DeleteResult(success=True)

        # 解析哈希 - 优先使用调用方提供的，然后尝试文件，最后查数据库
        if source_hash is None:
            try:
                source_hash = self.integrity.compute_sha256(source_path)
            except Exception as e:
                source_hash = self._hash_from_path(source_path)
                if source_hash is None:
                    result.success = False
                    result.errors.append(f"无法识别文档: {e}")
                    return result

        # 1. ChromaDB - 删除匹配 source_hash 的分块
        try:
            count = self.chroma.delete_by_metadata(
                {"doc_hash": source_hash}
            )
            result.chunks_deleted = count
        except Exception as e:
            result.errors.append(f"ChromaDB 删除失败: {e}")

        # 2. BM25 - 移除此文档的索引项
        try:
            result.bm25_removed = self.bm25.remove_document(
                source_hash, collection
            )
        except Exception as e:
            result.errors.append(f"BM25 移除失败: {e}")

        # 3. ImageStorage - 按 doc_hash 删除图片
        try:
            images = self.images.list_images(doc_hash=source_hash)
            deleted_imgs = 0
            for img in images:
                if self.images.delete_image(img["image_id"]):
                    deleted_imgs += 1
            result.images_deleted = deleted_imgs
        except Exception as e:
            result.errors.append(f"ImageStorage 删除失败: {e}")

        # 4. FileIntegrity - 移除摄取记录
        try:
            result.integrity_removed = self.integrity.remove_record(
                source_hash
            )
        except Exception as e:
            result.errors.append(f"FileIntegrity 移除失败: {e}")

        if result.errors:
            result.success = False

        return result

    # ------------------------------------------------------------------
    # get_collection_stats    # ------------------------------------------------------------------

    def get_collection_stats(
        self, collection: Optional[str] = None
    ) -> CollectionStats:
        """返回集合的聚合统计信息。

        参数：
            collection: 集合名称。为 *None* 时，统计跨越
                所有集合。

        返回：
            ``CollectionStats`` 数据类。
        """
        docs = self.list_documents(collection)
        chunk_total = sum(d.chunk_count for d in docs)
        image_total = sum(d.image_count for d in docs)

        return CollectionStats(
            collection=collection,
            document_count=len(docs),
            chunk_count=chunk_total,
            image_count=image_total,
        )

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    def _count_chunks(self, source_hash: str) -> int:
        """统计 Chroma 中属于 *source_hash* 的分块数。"""
        try:
            results = self.chroma.collection.get(
                where={"doc_hash": source_hash}, include=[]
            )
            return len(results.get("ids", []))
        except Exception:
            return 0

    def _get_chunk_ids(self, source_hash: str) -> List[str]:
        """返回 Chroma 中匹配 *source_hash* 的分块 ID。"""
        try:
            results = self.chroma.collection.get(
                where={"doc_hash": source_hash}, include=[]
            )
            return results.get("ids", [])
        except Exception:
            return []

    def _count_images(self, source_hash: str) -> int:
        """统计属于 *source_hash* 的图片数。"""
        try:
            return len(self.images.list_images(doc_hash=source_hash))
        except Exception:
            return 0

    def _get_image_ids(self, source_hash: str) -> List[str]:
        """返回属于 *source_hash* 的图片 ID。"""
        try:
            imgs = self.images.list_images(doc_hash=source_hash)
            return [img["image_id"] for img in imgs]
        except Exception:
            return []

    def _hash_from_path(self, source_path: str) -> Optional[str]:
        """尝试通过路径从完整性记录中查找 source_hash。"""
        try:
            for rec in self.integrity.list_processed():
                if rec["file_path"] == source_path:
                    return rec["file_hash"]
        except Exception:
            pass
        return None
