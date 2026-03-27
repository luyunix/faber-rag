"""分块模块 - 文档切分的适配层。

本模块提供文本切分的业务适配器，将 Document 对象转换为
Chunk 对象，并保留适当的元数据和可追溯性。
"""

from src.ingestion.chunking.document_chunker import DocumentChunker

__all__ = ["DocumentChunker"]
