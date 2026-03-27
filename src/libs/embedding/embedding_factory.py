"""用于创建嵌入提供商实例的工厂。

本模块实现工厂模式，根据配置实例化合适的
嵌入提供商，支持无需更改代码即可配置驱动地选择
不同的后端。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from src.core.settings import Settings


class EmbeddingFactory:
    """用于创建嵌入提供商实例的工厂。

    此工厂从设置中读取提供商配置，并实例化
    相应的嵌入实现。支持的提供商：OpenAI、Azure、
    Ollama、Qwen，以及更多将在后续任务中添加的提供商。

    应用的设计原则：
    - 工厂模式：集中对象创建逻辑。
    - 配置驱动：基于 settings.yaml 选择提供商。
    - 快速失败：对未知提供商抛出明确的错误。
    """

    # 支持的提供商注册表
    _PROVIDERS: dict[str, type[BaseEmbedding]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseEmbedding]) -> None:
        """注册一个新的嵌入提供商实现。

        此方法允许提供商实现向工厂注册自身，
        支持可扩展性。

        参数：
            name: 提供商标识符（例如，'openai'、'azure'、'local'）。
            provider_class: 实现该提供商的 BaseEmbedding 子类。

        异常：
            ValueError: 如果 provider_class 没有继承 BaseEmbedding。
        """
        if not issubclass(provider_class, BaseEmbedding):
            raise ValueError(
                f"提供商类 {provider_class.__name__} 必须继承自 BaseEmbedding"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseEmbedding:
        """基于配置创建嵌入实例。

        参数：
            settings: 包含嵌入配置的应用设置。
            **override_kwargs: 可选参数以覆盖配置值。

        返回：
            已配置嵌入提供商的实例。

        异常：
            ValueError: 如果配置的提供商不受支持。
            AttributeError: 如果缺少必需的配置字段。

        示例：
            >>> settings = Settings.load('config/settings.yaml')
            >>> embedding = EmbeddingFactory.create(settings)
            >>> vectors = embedding.embed(["hello world", "test"])
        """
        # 从设置中提取提供商名称
        try:
            provider_name = settings.embedding.provider.lower()
        except AttributeError as e:
            raise ValueError(
                "缺少必需的配置：settings.embedding.provider。"
                "请确保在 settings.yaml 中指定了 'embedding.provider'"
            ) from e

        # 在注册表中查找提供商类
        provider_class = cls._PROVIDERS.get(provider_name)

        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "无"
            raise ValueError(
                f"不支持的嵌入提供商：'{provider_name}'。"
                f"可用提供商：{available}"
            )

        # 实例化提供商
        # 提供商类应接受设置和可选的 kwargs
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"无法实例化嵌入提供商 '{provider_name}'：{e}"
            ) from e

    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的提供商名称。

        返回：
            可用提供商标识符的排序列表。
        """
        return sorted(cls._PROVIDERS.keys())


# 模块导入时自动注册提供商
def _register_builtin_providers() -> None:
    """向工厂注册内置嵌入提供商。"""
    try:
        from src.libs.embedding.openai_embedding import OpenAIEmbedding
        EmbeddingFactory.register_provider("openai", OpenAIEmbedding)
    except ImportError:
        pass  # OpenAI 提供商不可用

    try:
        from src.libs.embedding.azure_embedding import AzureEmbedding
        EmbeddingFactory.register_provider("azure", AzureEmbedding)
    except ImportError:
        pass  # Azure 提供商不可用

    try:
        from src.libs.embedding.ollama_embedding import OllamaEmbedding
        EmbeddingFactory.register_provider("ollama", OllamaEmbedding)
    except ImportError:
        pass  # Ollama 提供商不可用

    try:
        from src.libs.embedding.qwen_embedding import QwenEmbedding
        EmbeddingFactory.register_provider("qwen", QwenEmbedding)
    except ImportError:
        pass  # Qwen 提供商不可用


# 导入模块时注册提供商
_register_builtin_providers()
