"""
Embedding 模块。

本包包含嵌入组件：
- 稠密编码器
- 稀疏编码器 (BM25)
- 批处理器
"""

from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.embedding.batch_processor import BatchProcessor, BatchResult

__all__ = ["DenseEncoder", "SparseEncoder", "BatchProcessor", "BatchResult"]
