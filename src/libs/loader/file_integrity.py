"""用于增量摄入的文件完整性检查器。

本模块提供基于 SHA256 的文件完整性跟踪功能，支持增量摄入。
已成功处理的文件在后续摄入运行时可以跳过。

设计原则：
- 幂等性：同一文件的多次摄入运行是安全的
- 持久化：SQLite 后端存储在进程重启后依然有效
- 并发性：WAL 模式支持并发读写操作
- 优雅性：失败的摄入会被跟踪但不会阻止重试
"""

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class FileIntegrityChecker(ABC):
    """文件完整性检查的抽象基类。
    
    实现类跟踪哪些文件已被成功处理，
    以支持增量摄入。
    """
    
    @abstractmethod
    def compute_sha256(self, file_path: str) -> str:
        """计算文件的 SHA256 哈希值。
        
        参数：
            file_path: 要哈希的文件路径。
            
        返回：
            十六进制 SHA256 哈希字符串（64 个字符）。
            
        异常：
            FileNotFoundError: 如果文件不存在。
            IOError: 如果路径不是文件或无法读取。
        """
        pass
    
    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """根据哈希值检查是否应跳过文件。
        
        参数：
            file_hash: 文件的 SHA256 哈希值。
            
        返回：
            如果文件之前已被成功处理则返回 True，否则返回 False。
        """
        pass
    
    @abstractmethod
    def mark_success(
        self, 
        file_hash: str, 
        file_path: str, 
        collection: Optional[str] = None
    ) -> None:
        """标记文件为已成功处理。
        
        参数：
            file_hash: 文件的 SHA256 哈希值。
            file_path: 原始文件路径（用于跟踪）。
            collection: 可选的集合/命名空间标识符。
            
        异常：
            RuntimeError: 如果数据库操作失败。
        """
        pass
    
    @abstractmethod
    def mark_failed(
        self, 
        file_hash: str, 
        file_path: str, 
        error_msg: str
    ) -> None:
        """标记文件处理为失败。
        
        失败的文件会被跟踪但不会在后续运行中跳过，
        允许重试。
        
        参数：
            file_hash: 文件的 SHA256 哈希值。
            file_path: 原始文件路径（用于跟踪）。
            error_msg: 描述失败的错误消息。
            
        异常：
            RuntimeError: 如果数据库操作失败。
        """
        pass

    @abstractmethod
    def remove_record(self, file_hash: str) -> bool:
        """Remove an ingestion record by its file hash.

        Args:
            file_hash: SHA256 hash identifying the record.

        Returns:
            True if a record was deleted, False if not found.
        """
        pass

    @abstractmethod
    def list_processed(
        self, collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """列出已成功处理的文件。

        参数：
            collection: 可选的集合过滤器。当为 *None* 时返回所有
                成功记录。

        返回：
            字典列表，包含键：file_hash、file_path、collection、
            processed_at、updated_at。
        """
        pass


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """基于 SQLite 的文件完整性检查器。
    
    将摄入历史存储在 SQLite 数据库中，支持 WAL 模式以
    进行并发访问。
    
    数据库模式：
        ingestion_history (
            file_hash TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            status TEXT NOT NULL,  -- 'success' 或 'failed'
            collection TEXT,
            error_msg TEXT,
            processed_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    
    参数：
        db_path: SQLite 数据库文件路径（如需要则创建）。
    
    异常：
        sqlite3.DatabaseError: 如果数据库文件损坏。
    """
    
    def __init__(self, db_path: str):
        """初始化检查器并创建数据库（如需要）。
        
        参数：
            db_path: SQLite 数据库文件路径。
        """
        self.db_path = db_path
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
        # 创建 parent directories if needed
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect and initialize schema
        conn = sqlite3.connect(self.db_path)
        try:
            # Enable WAL mode for concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            
            # 创建 table if not exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    collection TEXT,
                    error_msg TEXT,
                    processed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # 创建 index on status for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status 
                ON ingestion_history(status)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def compute_sha256(self, file_path: str) -> str:
        """Compute SHA256 hash of file using chunked reading.
        
        Uses 64KB chunks to handle large files without loading entire
        file into memory.
        
        Args:
            file_path: Path to the file to hash.
            
        Returns:
            Hexadecimal SHA256 hash string (64 characters).
            
        Raises:
            FileNotFoundError: If file does not exist.
            IOError: If path is not a file or cannot be read.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise IOError(f"Path is not a file: {file_path}")
        
        # Compute hash using chunked reading
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                # 读取 in 64KB chunks
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(chunk)
        except Exception as e:
            raise IOError(f"Failed to read file {file_path}: {e}")
        
        return sha256_hash.hexdigest()
    
    def should_skip(self, file_hash: str) -> bool:
        """Check if file should be skipped.
        
        Only files with status='success' are skipped. Failed files
        can be retried.
        
        Args:
            file_hash: SHA256 hash of the file.
            
        Returns:
            True if file has status='success', False otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            result = cursor.fetchone()
            
            if result is None:
                return False
            
            return result[0] == "success"
        finally:
            conn.close()
    
    def mark_success(
        self, 
        file_hash: str, 
        file_path: str, 
        collection: Optional[str] = None
    ) -> None:
        """Mark file as successfully processed.
        
        Uses INSERT OR REPLACE for idempotent operation.
        
        Args:
            file_hash: SHA256 hash of the file.
            file_path: Original file path (for tracking).
            collection: Optional collection/namespace identifier.
            
        Raises:
            RuntimeError: If database operation fails.
        """
        now = datetime.now(timezone.utc).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        try:
            # 检查 if record exists to preserve processed_at
            cursor = conn.execute(
                "SELECT processed_at FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            result = cursor.fetchone()
            
            if result:
                # 更新 existing record
                conn.execute("""
                    UPDATE ingestion_history 
                    SET file_path = ?,
                        status = 'success',
                        collection = ?,
                        error_msg = NULL,
                        updated_at = ?
                    WHERE file_hash = ?
                """, (file_path, collection, now, file_hash))
            else:
                # Insert new record
                conn.execute("""
                    INSERT INTO ingestion_history 
                    (file_hash, file_path, status, collection, error_msg, processed_at, updated_at)
                    VALUES (?, ?, 'success', ?, NULL, ?, ?)
                """, (file_hash, file_path, collection, now, now))
            
            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to mark success for {file_path}: {e}")
        finally:
            conn.close()
    
    def mark_failed(
        self, 
        file_hash: str, 
        file_path: str, 
        error_msg: str
    ) -> None:
        """Mark file processing as failed.
        
        Failed files are not skipped, allowing retries.
        
        Args:
            file_hash: SHA256 hash of the file.
            file_path: Original file path (for tracking).
            error_msg: Error message describing the failure.
            
        Raises:
            RuntimeError: If database operation fails.
        """
        now = datetime.now(timezone.utc).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        try:
            # 检查 if record exists to preserve processed_at
            cursor = conn.execute(
                "SELECT processed_at FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            result = cursor.fetchone()
            
            if result:
                # 更新 existing record
                conn.execute("""
                    UPDATE ingestion_history 
                    SET file_path = ?,
                        status = 'failed',
                        error_msg = ?,
                        updated_at = ?
                    WHERE file_hash = ?
                """, (file_path, error_msg, now, file_hash))
            else:
                # Insert new record
                conn.execute("""
                    INSERT INTO ingestion_history 
                    (file_hash, file_path, status, collection, error_msg, processed_at, updated_at)
                    VALUES (?, ?, 'failed', NULL, ?, ?, ?)
                """, (file_hash, file_path, error_msg, now, now))
            
            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to mark failure for {file_path}: {e}")
        finally:
            conn.close()

    def remove_record(self, file_hash: str) -> bool:
        """Remove an ingestion record by its file hash.

        Args:
            file_hash: SHA256 hash identifying the record.

        Returns:
            True if a record was deleted, False if not found.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),
            )
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to remove record {file_hash}: {e}")
        finally:
            conn.close()

    def list_processed(
        self, collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List successfully processed files.

        Args:
            collection: Optional collection filter.

        Returns:
            List of dicts with keys: file_hash, file_path, collection,
            processed_at, updated_at.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            query = (
                "SELECT file_hash, file_path, collection, processed_at, updated_at "
                "FROM ingestion_history WHERE status = 'success'"
            )
            params: list[str] = []
            if collection is not None:
                query += " AND collection = ?"
                params.append(collection)
            query += " ORDER BY processed_at ASC"

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
