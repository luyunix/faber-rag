"""
核心层 - 核心业务逻辑。

此包包含核心业务逻辑，包括：
- 配置管理 (settings.py)
- 核心数据类型 (types.py) - 所有管道阶段的共享契约
- 查询引擎
- 响应构建
- 追踪收集
"""

from src.core.types import Document, Chunk, ChunkRecord, Metadata, Vector, SparseVector

__all__ = [
    "Document",
    "Chunk", 
    "ChunkRecord",
    "Metadata",
    "Vector",
    "SparseVector"
]
