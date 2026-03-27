"""通过 DashScope OpenAI 兼容 API 实现 Qwen 嵌入。

本模块通过 DashScope OpenAI 兼容 API 端点
提供与阿里云文本嵌入模型的集成。

支持的模型：
- text-embedding-v1: 第一代嵌入模型
- text-embedding-v2: 性能和准确性改进
- text-embedding-v3: 最新模型，支持可配置维度（1024/768/512）

特性：
- OpenAI API 兼容接口
- 可配置嵌入维度（v3）
- 批量嵌入支持
- 中英文语言支持

用法：
    from src.libs.embedding.qwen_embedding import QwenEmbedding
    from src.core.settings import load_settings

    settings = load_settings()
    embedding = QwenEmbedding(settings)
    vectors = embedding.embed(["文本一", "文本二"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from src.libs.embedding.openai_embedding import OpenAIEmbedding

if TYPE_CHECKING:
    from src.core.settings import Settings


class QwenEmbeddingError(Exception):
    """QwenEmbedding 引发的异常。"""
    pass


class QwenEmbedding(OpenAIEmbedding):
    """使用 OpenAI 兼容 API 的 Qwen 嵌入提供商。

    此类继承自 OpenAIEmbedding，因为 Qwen 的 DashScope API
    与 OpenAI 的嵌入 API 格式完全兼容。主要区别是：
    - 基本 URL：https://dashscope.aliyuncs.com/compatible-mode/v1
    - API 密钥：DashScope API 密钥（以 'sk-' 开头）
    - 模型：text-embedding-v1、text-embedding-v2、text-embedding-v3
    - 维度：v3 可配置（1024/768/512）

    所有嵌入功能均从 OpenAIEmbedding 继承。
    """

    def __init__(self, settings: Settings, **override_kwargs: Any) -> None:
        """初始化 Qwen 嵌入。

        参数：
            settings: 包含嵌入配置的应用设置。
            **override_kwargs: 配置值的可选覆盖。
        """
        # Qwen 使用 OpenAI 兼容 API，因此我们可以复用 OpenAIEmbedding
        # 需要确保 settings.embedding.api_key 已设置
        super().__init__(settings, **override_kwargs)

    def __repr__(self) -> str:
        """返回 Qwen 嵌入的字符串表示。"""
        return f"QwenEmbedding(model={self.model}, dimensions={self.dimensions})"
