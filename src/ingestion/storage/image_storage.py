"""基于 SQLite 索引的多模态 RAG 图像存储。

本模块提供基于 SQLite 索引的图像存储功能，
支持高效的图像检索和管理。

设计原则：
- 持久化：图像存储在文件系统中，元数据存储在 SQLite
- 并发性：WAL 模式支持并发读写操作
- 幂等性：重复保存相同 image_id 会安全更新元数据
- 组织性：图像按集合分组，实现命名空间隔离
"""

import sqlite3
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Union


class ImageStorage:
    """基于 SQLite 的图像存储管理器。
    
    将图像文件存储在组织化的目录结构中，并维护
    SQLite 索引以进行高效查找和查询。
    
    目录结构：
        data/images/{collection}/{image_id}.png
    
    数据库模式：
        image_index (
            image_id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            collection TEXT,
            doc_hash TEXT,
            page_num INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        
        INDEX idx_collection ON image_index(collection)
        INDEX idx_doc_hash ON image_index(doc_hash)
    
    参数：
        db_path: SQLite 数据库文件路径（默认：data/db/image_index.db）。
        images_root: 图像存储根目录（默认：data/images）。
    
    示例：
        >>> storage = ImageStorage()
        >>> 
        >>> # 保存图像
        >>> with open("sample.png", "rb") as f:
        >>>     image_data = f.read()
        >>> path = storage.save_image(
        ...     image_id="doc123_p1_img0",
        ...     image_data=image_data,
        ...     collection="contracts",
        ...     doc_hash="abc123",
        ...     page_num=1
        ... )
        >>> print(path)  # data/images/contracts/doc123_p1_img0.png
        >>> 
        >>> # 获取图像路径
        >>> path = storage.get_image_path("doc123_p1_img0")
        >>> print(path)  # data/images/contracts/doc123_p1_img0.png
        >>> 
        >>> # 列出集合中的图像
        >>> images = storage.list_images("contracts")
        >>> print(len(images))  # 1
    """
    
    def __init__(
        self,
        db_path: str = "data/db/image_index.db",
        images_root: str = "data/images"
    ):
        """初始化图像存储并创建数据库（如需要）。
        
        参数：
            db_path: SQLite 数据库文件路径。
            images_root: 存储图像文件的根目录。
        """
        self.db_path = db_path
        self.images_root = Path(images_root)
        self._conn = None
        self._ensure_database()
    
    def close(self) -> None:
        """关闭数据库连接（如果已打开）。"""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __del__(self):
        """清理：删除时关闭连接。"""
        self.close()
    
    def _ensure_database(self) -> None:
        """创建数据库文件和模式（如不存在）。"""
        # 创建 parent directories
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建 images root directory
        self.images_root.mkdir(parents=True, exist_ok=True)
        
        # Connect and initialize schema
        conn = sqlite3.connect(self.db_path)
        try:
            # Enable WAL mode for concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            
            # 创建表（如不存在）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT,
                    doc_hash TEXT,
                    page_num INTEGER,
                    created_at TEXT NOT NULL
                )
            """)
            
            # 创建索引以提高查询效率
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_collection 
                ON image_index(collection)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_hash 
                ON image_index(doc_hash)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def save_image(
        self,
        image_id: str,
        image_data: Union[bytes, Path, str],
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None,
        page_num: Optional[int] = None,
        extension: str = "png"
    ) -> str:
        """保存图像到文件系统并注册到数据库。
        
        此操作是幂等的 - 使用相同的 image_id 重新保存
        将更新元数据并覆盖文件。
        
        参数：
            image_id: 图像的唯一标识符。
            image_data: 图像数据（字节）或源图像文件路径。
            collection: 可选的集合/命名空间用于组织。
            doc_hash: 可选的文档哈希用于追溯。
            page_num: 可选的页码（如果来自分页文档）。
            extension: 文件扩展名，不带点（默认："png"）。
        
        返回：
            图像保存的相对路径。
            
        异常：
            ValueError: 如果 image_id 为空或无效。
            IOError: 如果图像文件无法保存。
            RuntimeError: 如果数据库操作失败。
            
        示例：
            >>> # 从字节保存
            >>> path = storage.save_image("img1", b"PNG_DATA", "docs")
            >>> 
            >>> # 从文件保存
            >>> path = storage.save_image("img2", Path("source.png"), "docs")
        """
        if not image_id or not image_id.strip():
            raise ValueError("image_id cannot be empty")
        
        # 确定集合目录
        if collection:
            collection_dir = self.images_root / collection
        else:
            collection_dir = self.images_root / "default"

        collection_dir.mkdir(parents=True, exist_ok=True)

        # 构建 image file path
        image_filename = f"{image_id}.{extension}"
        image_path = collection_dir / image_filename

        # 保存图像文件
        try:
            if isinstance(image_data, bytes):
                # 写入 bytes directly
                image_path.write_bytes(image_data)
            elif isinstance(image_data, (Path, str)):
                # 从源文件复制
                source_path = Path(image_data)
                if not source_path.exists():
                    raise FileNotFoundError(f"Source image not found: {source_path}")
                shutil.copy2(source_path, image_path)
            else:
                raise ValueError(f"Unsupported image_data type: {type(image_data)}")
        except Exception as e:
            raise IOError(f"Failed to save image {image_id}: {e}")

        # 存储 absolute path for reliable retrieval
        # (relative paths would fail with temp directories in tests)
        stored_path = str(image_path.resolve())

        # Register in database
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self.db_path)
        try:
            # Use INSERT OR REPLACE for idempotent operation
            conn.execute("""
                INSERT OR REPLACE INTO image_index 
                (image_id, file_path, collection, doc_hash, page_num, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (image_id, stored_path, collection, doc_hash, page_num, now))

            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to register image {image_id}: {e}")
        finally:
            conn.close()

        return stored_path

    def register_image(
        self,
        image_id: str,
        file_path: Union[Path, str],
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None,
        page_num: Optional[int] = None
    ) -> str:
        """在数据库索引中注册现有图像文件。

        与 save_image() 不同，此方法不复制或移动文件。
        它仅创建数据库条目指向现有文件。
        当图像已被其他组件（如 PdfLoader）保存，
        您只需要索引它时使用此方法。

        参数：
            image_id: 图像的唯一标识符。
            file_path: 现有图像文件的路径。
            collection: 可选的集合/命名空间用于组织。
            doc_hash: 可选的文档哈希用于追溯。
            page_num: 可选的页码（如果来自分页文档）。

        返回：
            已注册图像的绝对路径。

        异常：
            ValueError: 如果 image_id 为空或无效。
            FileNotFoundError: 如果图像文件不存在。
            RuntimeError: 如果数据库操作失败。

        示例：
            >>> # 注册由 PdfLoader 保存的图像
            >>> path = storage.register_image(
            ...     image_id="doc123_p1_img0",
            ...     file_path="data/images/tech_docs/abc123/doc123_p1_img0.png",
            ...     collection="tech_docs",
            ...     doc_hash="abc123",
            ...     page_num=1
            ... )
        """
        if not image_id or not image_id.strip():
            raise ValueError("image_id cannot be empty")

        # 验证文件存在
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # 存储 absolute path for reliable retrieval
        stored_path = str(path.resolve())

        # Register in database
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self.db_path)
        try:
            # Use INSERT OR REPLACE for idempotent operation
            conn.execute("""
                INSERT OR REPLACE INTO image_index 
                (image_id, file_path, collection, doc_hash, page_num, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (image_id, stored_path, collection, doc_hash, page_num, now))

            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to register image {image_id}: {e}")
        finally:
            conn.close()

        return stored_path

    def get_image_path(self, image_id: str) -> Optional[str]:
        """通过 ID 获取图像的文件系统路径。

        参数：
            image_id: 图像的唯一标识符。

        返回：
            如果图像存在则返回相对文件路径，否则返回 None。

        示例：
            >>> path = storage.get_image_path("img1")
            >>> if path:
            ...     with open(path, "rb") as f:
            ...         image_data = f.read()
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (image_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            conn.close()

    def image_exists(self, image_id: str) -> bool:
        """检查图像是否存在于数据库中。

        参数：
            image_id: 图像的唯一标识符。

        返回：
            如果图像已注册则返回 True，否则返回 False。
        """
        return self.get_image_path(image_id) is not None

    def list_images(
        self,
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """列出图像，支持可选过滤。

        参数：
            collection: 可选的集合过滤条件。
            doc_hash: 可选的文档哈希过滤条件。

        返回：
            图像元数据字典列表，包含以下键：
            - image_id: 图像标识符
            - file_path: 文件系统路径
            - collection: 集合名称
            - doc_hash: 文档哈希
            - page_num: 页码（如适用）
            - created_at: 创建时间戳

        示例：
            >>> # 列出集合中的所有图像
            >>> images = storage.list_images(collection="contracts")
            >>> for img in images:
            ...     print(img["image_id"], img["file_path"])

            >>> # 列出特定文档的图像
            >>> images = storage.list_images(doc_hash="abc123")
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access

        try:
            # 构建 query with optional filters
            query = "SELECT * FROM image_index WHERE 1=1"
            params = []

            if collection is not None:
                query += " AND collection = ?"
                params.append(collection)

            if doc_hash is not None:
                query += " AND doc_hash = ?"
                params.append(doc_hash)

            query += " ORDER BY created_at ASC"

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            # 将行转换为字典
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def delete_image(self, image_id: str, remove_file: bool = True) -> bool:
        """从数据库删除图像，可选择是否同时从文件系统删除。

        参数：
            image_id: 图像的唯一标识符。
            remove_file: 如果为 True，则同时删除图像文件（默认：True）。

        返回：
            如果图像被删除则返回 True，否则返回 False。

        示例：
            >>> # 删除图像和文件
            >>> deleted = storage.delete_image("img1")
            >>>
            >>> # 仅从数据库删除，保留文件
            >>> deleted = storage.delete_image("img2", remove_file=False)
        """
        # 在删除前获取文件路径
        file_path = self.get_image_path(image_id)

        if file_path is None:
            return False

        # 删除 from database
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM image_index WHERE image_id = ?",
                (image_id,)
            )
            conn.commit()
            deleted = cursor.rowcount > 0
        except sqlite3.Error:
            return False
        finally:
            conn.close()

        # 可选删除文件
        if remove_file and deleted:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                # 日志 but don't fail if file deletion fails
                pass

        return deleted

    def get_collection_stats(self, collection: str) -> Dict[str, any]:
        """获取集合的统计信息。

        参数：
            collection: 集合名称。

        返回：
            包含统计信息的字典：
            - total_images: 集合中的图像数量
            - total_size_bytes: 总文件大小（如果文件存在）

        示例：
            >>> stats = storage.get_collection_stats("contracts")
            >>> print(f"总图像数：{stats['total_images']}")
        """
        images = self.list_images(collection=collection)

        total_size = 0
        for img in images:
            try:
                file_path = Path(img["file_path"])
                if file_path.exists():
                    total_size += file_path.stat().st_size
            except Exception:
                pass

        return {
            "total_images": len(images),
            "total_size_bytes": total_size
        }
