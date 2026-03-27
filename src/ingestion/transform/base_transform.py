"""用于 chunk 转换操作的基类。"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext


class BaseTransform(ABC):
    """用于 chunk 转换操作的抽象基类。
    
    转换操作用于处理 chunk，以增强其质量、添加元数据，
    或为下游处理（嵌入、索引）做准备。
    
    设计原则：
        - 单一职责：每个转换只做一种类型的增强
        - 原子操作：一个 chunk 的失败不会影响其他 chunk
        - 可观测性：在 TraceContext 中记录处理信息
        - 优雅降级：遇到不可恢复的错误时返回原始 chunk
    """
    
    @abstractmethod
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """转换 chunk 列表。
        
        参数：
            chunks: 要转换的 chunk 列表
            trace: 用于可观测性的可选跟踪上下文
            
        返回：
            已转换的 chunk 列表（与输入长度相同）
            
        异常：
            ValueError: 如果输入验证失败
        """
        pass
