"""嵌入提供商的抽象基类。

本模块定义了所有嵌入提供商必须实现的接口，
确保不同提供商之间具有一致的行为。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseEmbedding(ABC):
    """所有嵌入提供商的抽象基类。
    
    所有嵌入实现必须继承此类并实现 embed() 方法。
    这确保了不同提供商之间具有统一的接口。
    """
    
    @abstractmethod
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """为一批文本生成嵌入向量。
        
        参数：
            texts: 要嵌入的文本字符串列表。
            trace: 可选的 TraceContext 用于可观测性。
            **kwargs: 提供商特定的参数。
        
        返回：
            嵌入向量列表，每个向量是浮点数列表。
        
        异常：
            ValueError: 如果输入无效。
            RuntimeError: 如果嵌入失败。
        """
        pass
    
    def validate_texts(self, texts: List[str]) -> None:
        """验证文本列表。
        
        参数：
            texts: 要验证的文本列表。
        
        异常：
            ValueError: 如果列表为空或包含非字符串项。
        """
        if not isinstance(texts, list):
            raise ValueError("Texts must be a list")
        if not texts:
            raise ValueError("Texts list cannot be empty")
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(
                    f"Text at index {i} is not a string (type: {type(text).__name__})"
                )
            if not text.strip():
                raise ValueError(
                    f"Text at index {i} is empty or whitespace-only"
                )
    
    @abstractmethod
    def get_dimension(self) -> Optional[int]:
        """获取嵌入向量的维度。
        
        返回：
            嵌入维度，如果不确定则返回 None。
        """
        pass
