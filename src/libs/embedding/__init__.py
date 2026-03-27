"""
嵌入模块。

该包包含嵌入服务的抽象和实现：
- 基础嵌入类
- 嵌入工厂
- 提供商实现（OpenAI、Qwen）
"""

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.openai_embedding import OpenAIEmbedding
from src.libs.embedding.qwen_embedding import QwenEmbedding

__all__ = [
    "BaseEmbedding",
    "EmbeddingFactory",
    "OpenAIEmbedding",
    "QwenEmbedding",
]
