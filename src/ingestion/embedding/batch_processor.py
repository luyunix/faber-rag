"""用于协调稠密和稀疏编码的批处理器。

本模块实现了 ingestion pipeline 的批处理器组件，
负责协调编码工作流程并管理批处理操作。

设计原则：
- 编排性：在统一工作流程中协调 DenseEncoder 和 SparseEncoder
- 配置驱动：批处理大小来自配置，而非硬编码
- 可观测性：通过 TraceContext 记录批处理时间和统计信息
- 错误处理：单个批处理失败不会导致整个 pipeline 崩溃
- 确定性：相同的输入产生相同的批处理和结果
"""

from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass
import logging

from src.core.types import Chunk
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """批处理操作的结果。
    
    属性：
        dense_vectors: 稠密嵌入列表（每个 chunk 一个）
        sparse_stats: 术语统计列表（每个 chunk 一个）
        batch_count: 处理的批次数
        total_time: 总处理时间（秒）
        successful_chunks: 成功处理的 chunk 数量
        failed_chunks: 处理失败的 chunk 数量
    """
    dense_vectors: List[List[float]]
    sparse_stats: List[Dict[str, Any]]
    batch_count: int
    total_time: float
    successful_chunks: int
    failed_chunks: int


class BatchProcessor:
    """编排通过编码 pipeline 的 chunk 批处理。
    
    此处理器管理将 chunk 转换为稠密和稀疏表示的工作流程。
    它将 chunk 分成批次，驱动编码器，并收集时间指标。
    
    设计：
    - 无状态：process() 调用之间不维护状态
    - 并行编码：稠密和稀疏编码独立进行
    - 指标收集：记录批处理级别的时间用于可观测性
    - 顺序保持：输出顺序与输入 chunk 顺序匹配
    
    示例：
        >>> from src.libs.embedding.embedding_factory import EmbeddingFactory
        >>> from src.core.settings import load_settings
        >>> 
        >>> settings = load_settings("config/settings.yaml")
        >>> embedding = EmbeddingFactory.create(settings)
        >>> dense_encoder = DenseEncoder(embedding, batch_size=2)
        >>> sparse_encoder = SparseEncoder()
        >>> 
        >>> processor = BatchProcessor(
        ...     dense_encoder=dense_encoder,
        ...     sparse_encoder=sparse_encoder,
        ...     batch_size=2
        ... )
        >>> 
        >>> chunks = [
        ...     Chunk(id="1", text="Hello", metadata={}),
        ...     Chunk(id="2", text="World", metadata={})
        ... ]
        >>> result = processor.process(chunks)
        >>> len(result.dense_vectors) == len(chunks)  # True
        >>> len(result.sparse_stats) == len(chunks)  # True
    """
    
    def __init__(
        self,
        dense_encoder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        batch_size: int = 100,
    ):
        """初始化批处理器。
        
        参数：
            dense_encoder: 用于嵌入生成的 DenseEncoder 实例
            sparse_encoder: 用于术语统计的 SparseEncoder 实例
            batch_size: 每批处理的 chunk 数量（默认：100）
        
        异常：
            ValueError: 如果 batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.batch_size = batch_size
    
    def process(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> BatchResult:
        """通过稠密和稀疏编码 pipeline 处理 chunk。
        
        工作流程：
        1. 验证输入
        2. 从 chunk 创建批次
        3. 通过两个编码器处理每个批次
        4. 收集结果和时间指标
        5. 如果可用则记录到 TraceContext
        
        参数：
            chunks: 要处理的 Chunk 对象列表
            trace: 用于可观测性的可选 TraceContext
        
        返回：
            包含向量、统计信息和指标的 BatchResult
        
        异常：
            ValueError: 如果 chunk 列表为空
            RuntimeError: 如果两个编码器都完全失败
        
        示例：
            >>> chunks = [Chunk(id=f"{i}", text=f"Text {i}", metadata={}) 
            ...           for i in range(5)]
            >>> result = processor.process(chunks)
            >>> result.batch_count  # 3（batch_size=2 时）
            >>> result.successful_chunks  # 5
        """
        if not chunks:
            raise ValueError("Cannot process empty chunks list")
        
        start_time = time.time()
        
        # 创建批次
        batches = self._create_batches(chunks)
        batch_count = len(batches)
        
        # 处理所有批次
        dense_vectors: List[List[float]] = []
        sparse_stats: List[Dict[str, Any]] = []
        successful_chunks = 0
        failed_chunks = 0
        
        for batch_idx, batch in enumerate(batches):
            batch_start = time.time()
            
            # 记录批次中每个 chunk 的详细信息
            logger.info(f"BatchProcessor: Batch {batch_idx} contains {len(batch)} chunks:")
            for idx, chunk in enumerate(batch):
                logger.info(f"  Chunk {idx}: id={chunk.id}, type={type(chunk).__name__}, text_len={len(chunk.text) if hasattr(chunk, 'text') and chunk.text else 0}")
            
            try:
                # 稠密编码
                logger.info(f"BatchProcessor: Encoding dense vectors for batch {batch_idx} (size={len(batch)})")
                batch_dense = self.dense_encoder.encode(batch, trace=trace)
                logger.info(f"BatchProcessor: Dense encoder returned {len(batch_dense)} vectors for batch {batch_idx}")
                dense_vectors.extend(batch_dense)
                
                # 稀疏编码
                logger.info(f"BatchProcessor: Encoding sparse stats for batch {batch_idx} (size={len(batch)})")
                batch_sparse = self.sparse_encoder.encode(batch, trace=trace)
                logger.info(f"BatchProcessor: Sparse encoder returned {len(batch_sparse)} stats for batch {batch_idx}")
                sparse_stats.extend(batch_sparse)
                
                successful_chunks += len(batch)
                logger.info(f"BatchProcessor: Batch {batch_idx} completed successfully")
                
            except Exception as e:
                # 记录失败但继续处理剩余批次
                logger.error(f"BatchProcessor: Batch {batch_idx} failed: {e}", exc_info=True)
                failed_chunks += len(batch)
                if trace:
                    trace.record_stage(
                        f"batch_{batch_idx}_error",
                        {"error": str(e), "batch_size": len(batch)}
                    )
            
            batch_duration = time.time() - batch_start
            
            # 如果 trace 可用，记录批次时间
            if trace:
                trace.record_stage(
                    f"batch_{batch_idx}",
                    {
                        "batch_size": len(batch),
                        "duration_seconds": batch_duration,
                        "chunks_processed": len(batch)
                    }
                )
        
        total_time = time.time() - start_time
        
        # 记录整体处理统计
        if trace:
            trace.record_stage(
                "batch_processing",
                {
                    "total_chunks": len(chunks),
                    "batch_count": batch_count,
                    "batch_size": self.batch_size,
                    "successful_chunks": successful_chunks,
                    "failed_chunks": failed_chunks,
                    "total_time_seconds": total_time
                }
            )
        
        return BatchResult(
            dense_vectors=dense_vectors,
            sparse_stats=sparse_stats,
            batch_count=batch_count,
            total_time=total_time,
            successful_chunks=successful_chunks,
            failed_chunks=failed_chunks
        )
    
    def _create_batches(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """将 chunk 分成指定大小的批次。
        
        参数：
            chunks: 要分批的 chunk 列表
        
        返回：
            批次列表，每个批次是一个 chunk 列表。
            顺序保持不变：第一个批次包含 chunks[0:batch_size]，
            第二个批次包含 chunks[batch_size:2*batch_size]，依此类推。
        
        示例：
            >>> chunks = [Chunk(id=f"{i}", text="", metadata={}) for i in range(5)]
            >>> batches = processor._create_batches(chunks)
            >>> len(batches)  # 3（batch_size=2 时）
            >>> [len(b) for b in batches]  # [2, 2, 1]
        """
        batches = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def get_batch_count(self, total_chunks: int) -> int:
        """计算给定 chunk 数量的批次数。
        
        用于规划和测试的工具方法。
        
        参数：
            total_chunks: 要处理的总 chunk 数量
        
        返回：
            将创建的批次数
        
        示例：
            >>> processor.get_batch_count(5)  # 3（batch_size=2 时）
            >>> processor.get_batch_count(4)  # 2
            >>> processor.get_batch_count(0)  # 0
        """
        if total_chunks <= 0:
            return 0
        return (total_chunks + self.batch_size - 1) // self.batch_size
