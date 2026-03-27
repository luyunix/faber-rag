"""用于从文本分块生成嵌入向量的稠密编码器。

本模块实现摄取 Pipeline 的稠密编码器组件，
负责使用可配置的嵌入提供商将文本分块转换为稠密向量表示。

设计原则：
- 配置驱动：使用工厂模式从设置中获取嵌入提供商
- 批处理：通过批处理优化 API 调用
- 可观测：为未来可观测性集成接受 TraceContext
- 错误处理：单个失败不应导致整个批次崩溃
- 确定性：相同输入产生相同输出
"""

from typing import List, Optional, Any
from src.core.types import Chunk
from src.libs.embedding.base_embedding import BaseEmbedding
from src.observability.logger import get_logger

logger = get_logger(__name__)


class DenseEncoder:
    """使用 BaseEmbedding 提供者将文本分块编码为稠密向量。
    
    此编码器作为摄取管道和可插拔嵌入层之间的桥梁。
    它处理批处理、错误恢复，并维护输入分块和输出向量之间的对齐。
    
    设计原则：
    - 依赖注入：接收 BaseEmbedding 实例（不直接调用工厂）
    - 批处理优先：以可配置的批量大小处理所有分块
    - 无状态：encode() 调用之间没有内部状态
    
    示例：
        >>> from src.libs.embedding.embedding_factory import EmbeddingFactory
        >>> from src.core.settings import load_settings
        >>> 
        >>> settings = load_settings("config/settings.yaml")
        >>> embedding = EmbeddingFactory.create(settings)
        >>> encoder = DenseEncoder(embedding, batch_size=32)
        >>> 
        >>> chunks = [Chunk(id="1", text="Hello world", metadata={})]
        >>> vectors = encoder.encode(chunks)
        >>> print(len(vectors))  # 1
        >>> print(len(vectors[0]))  # dimension (e.g., 1536)
    """
    
    def __init__(
        self,
        embedding: BaseEmbedding,
        batch_size: int = 100,
    ):
        """初始化 DenseEncoder。
        
        Args:
            embedding: Embedding 提供者实例（来自 EmbeddingFactory）
            batch_size: 每次 API 调用处理的块数量（默认：100）
        
        Raises:
            ValueError: 如果 batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        self.embedding = embedding
        self.batch_size = batch_size
    
    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[List[float]]:
        """将分块编码为稠密向量。
        
        此方法：
        1. 从每个分块提取文本
        2. 根据 batch_size 分批文本
        3. 为每批调用 embedding.embed()
        4. 连接结果并保持分块顺序
        
        Args:
            chunks: 要编码的 Chunk 对象列表
            trace: 可选的 TraceContext 用于可观测性（保留用于 Stage F）
        
        Returns:
            稠密向量列表（每个分块一个，顺序相同）。
            每个向量是一个浮点数列表，维度与嵌入模型匹配。
        
        Raises:
            ValueError: 如果分块列表为空
            RuntimeError: 如果嵌入提供者在所有批次中都失败
        
        Example:
            >>> chunks = [
            ...     Chunk(id="1", text="First chunk", metadata={}),
            ...     Chunk(id="2", text="Second chunk", metadata={})
            ... ]
            >>> vectors = encoder.encode(chunks)
            >>> len(vectors) == len(chunks)  # True
        """
        if not chunks:
            raise ValueError("Cannot encode empty chunks list")
        
        # 从分块中提取文本
        texts = [chunk.text for chunk in chunks]
        
        # 验证所有文本非空
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(
                    f"Chunk at index {i} (id={chunks[i].id}) has empty or whitespace-only text"
                )
        
        # 分批处理
        all_vectors: List[List[float]] = []
        
        for batch_start in range(0, len(texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            
            try:
                # 调用嵌入提供者
                batch_vectors = self.embedding.embed(
                    texts=batch_texts,
                    trace=trace,
                )
                
                # 验证输出形状
                if len(batch_vectors) != len(batch_texts):
                    raise RuntimeError(
                        f"Embedding provider returned {len(batch_vectors)} vectors "
                        f"for {len(batch_texts)} texts in batch {batch_start}-{batch_end}"
                    )
                
                all_vectors.extend(batch_vectors)
                
            except Exception as e:
                # 重新抛出异常，带上批次失败上下文
                raise RuntimeError(
                    f"Failed to encode batch {batch_start}-{batch_end}: {str(e)}"
                ) from e
        
        # 最终验证
        if len(all_vectors) != len(chunks):
            raise RuntimeError(
                f"Vector count mismatch: got {len(all_vectors)} vectors "
                f"for {len(chunks)} chunks"
            )
        
        # 验证向量维度一致
        if all_vectors:
            expected_dim = len(all_vectors[0])
            for i, vec in enumerate(all_vectors):
                if len(vec) != expected_dim:
                    raise RuntimeError(
                        f"Inconsistent vector dimensions: vector {i} has "
                        f"{len(vec)} dimensions, expected {expected_dim}"
                    )
        
        return all_vectors
    
    def get_batch_count(self, num_chunks: int) -> int:
        """计算给定分块数所需的批次数。
        
        用于日志记录/进度跟踪的工具方法。
        
        Args:
            num_chunks: 要编码的分块数
        
        Returns:
            所需批次数
        """
        if num_chunks <= 0:
            return 0
        return (num_chunks + self.batch_size - 1) // self.batch_size
