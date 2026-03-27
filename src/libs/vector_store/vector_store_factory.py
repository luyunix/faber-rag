"""用于创建 VectorStore 提供者实例的工厂。

本模块实现了工厂模式，根据配置实例化适当的
VectorStore 提供者，实现无需代码更改即可切换不同后端。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from src.core.settings import Settings


class VectorStoreFactory:
    """用于创建 VectorStore 提供者实例的工厂。
    
    此工厂从设置中读取提供者配置并实例化
    相应的 VectorStore 实现。支持的提供者将在
    后续任务中添加（B7.6 及以后）。
    
    应用的设计原则：
    - 工厂模式：集中对象创建逻辑
    - 配置驱动：基于 settings.yaml 的提供者选择
    - 快速失败：为未知提供者抛出清晰错误
    """
    
    # 支持提供者的注册表（将在 B7.x 任务中填充）
    _PROVIDERS: dict[str, type[BaseVectorStore]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseVectorStore]) -> None:
        """注册新的 VectorStore 提供者实现。
        
        此方法允许提供者实现向工厂注册自己，
        支持扩展性。
        
        参数:
            name: 提供者标识符（例如 'chroma', 'qdrant', 'milvus'）
            provider_class: 实现提供者的 BaseVectorStore 子类
        
        异常:
            ValueError: 如果 provider_class 不继承自 BaseVectorStore
        """
        if not issubclass(provider_class, BaseVectorStore):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseVectorStore"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseVectorStore:
        """根据配置创建 VectorStore 实例。
        
        参数:
            settings: 包含 VectorStore 配置的应用设置
            **override_kwargs: 可选参数，用于覆盖配置值
        
        返回:
            配置好的 VectorStore 提供者实例
        
        异常:
            ValueError: 如果配置的提供者不支持
            AttributeError: 如果缺少必需的配置字段
        
        示例:
            >>> settings = Settings.load('config/settings.yaml')
            >>> vector_store = VectorStoreFactory.create(settings)
            >>> vector_store.upsert([{'id': 'doc1', 'vector': [0.1, 0.2]}])
        """
        # 从设置中提取提供者名称
        try:
            provider_name = settings.vector_store.provider.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vector_store.provider. "
                "Please ensure 'vector_store.provider' is specified in settings.yaml"
            ) from e
        
        # 在注册表中查找提供者类
        provider_class = cls._PROVIDERS.get(provider_name)
        
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported VectorStore provider: '{provider_name}'. "
                f"Available providers: {available}. "
                f"Provider implementations will be added in task B7.6 and beyond."
            )
        
        # 实例化提供者
        # 提供者类应该接受 settings 和可选参数
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate VectorStore provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有注册的提供者名称。
        
        返回:
            提供者名称的排序列表
        
        示例:
            >>> VectorStoreFactory.list_providers()
            ['chroma', 'milvus', 'qdrant']
        """
        return sorted(cls._PROVIDERS.keys())
