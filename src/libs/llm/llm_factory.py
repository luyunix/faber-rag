"""用于创建 LLM 提供商实例的工厂。

本模块实现工厂模式，根据配置实例化合适的
LLM 提供商，支持无需更改代码即可配置驱动地选择
不同的后端。

支持纯文本 LLM 和视觉 LLM（多模态）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.base_vision_llm import BaseVisionLLM

if TYPE_CHECKING:
    from src.core.settings import Settings


# 在模块加载时导入并注册视觉 LLM 提供商
def _register_vision_providers() -> None:
    """注册所有视觉 LLM 提供商实现。

    此函数在模块导入时被调用，以填充视觉 LLM
    提供商注册表。当新的提供商实现时，请在此处添加。
    """
    # 确保类属性已初始化
    if not hasattr(LLMFactory, '_VISION_PROVIDERS'):
        LLMFactory._VISION_PROVIDERS = {}
    
    try:
        from src.libs.llm.azure_vision_llm import AzureVisionLLM
        LLMFactory.register_vision_provider("azure", AzureVisionLLM)
    except ImportError:
        # 提供商尚未实现，跳过注册
        pass

    try:
        from src.libs.llm.openai_vision_llm import OpenAIVisionLLM
        LLMFactory.register_vision_provider("openai", OpenAIVisionLLM)
    except ImportError:
        pass

    try:
        from src.libs.llm.qwen_vision_llm import QwenVisionLLM
        LLMFactory.register_vision_provider("qwen", QwenVisionLLM)
    except ImportError:
        pass


class LLMFactory:
    """用于创建 LLM 提供商实例的工厂。

    此工厂从设置中读取提供商配置，并实例化
    相应的 LLM 实现。支持纯文本 LLM 和
    视觉 LLM（多模态）。

    应用的设计原则：
    - 工厂模式：集中对象创建逻辑。
    - 配置驱动：基于 settings.yaml 选择提供商。
    - 快速失败：对未知提供商抛出明确的错误。
    - 分离：文本和视觉 LLM 注册表是分开的。
    """

    # 支持的纯文本 LLM 提供商注册表（将在 B7.x 任务中填充）    _PROVIDERS: dict[str, type[BaseLLM]] = {}

    # 支持的视觉 LLM 提供商注册表（将在 B9+ 任务中填充）    _VISION_PROVIDERS: dict[str, type[BaseVisionLLM]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLLM]) -> None:
        """注册一个新的 LLM 提供商实现。

        此方法允许提供商实现向工厂注册自身，
        支持可扩展性。

        参数：
            name: 提供商标识符（例如，'openai'、'azure'、'ollama'）。
            provider_class: 实现该提供商的 BaseLLM 子类。

        异常：
            ValueError: 如果 provider_class 没有继承 BaseLLM。
        """
        if not issubclass(provider_class, BaseLLM):
            raise ValueError(
                f"提供商类 {provider_class.__name__} 必须继承自 BaseLLM"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseLLM:
        """基于配置创建 LLM 实例。

        参数：
            settings: 包含 LLM 配置的应用设置。
            **override_kwargs: 可选参数以覆盖配置值。

        返回：
            已配置 LLM 提供商的实例。

        异常：
            ValueError: 如果配置的提供商不受支持。
            AttributeError: 如果缺少必需的配置字段。

        示例：
            >>> settings = Settings.load('config/settings.yaml')
            >>> llm = LLMFactory.create(settings)
            >>> response = llm.chat([Message(role='user', content='Hello')])
        """
        # 从设置中提取提供商名称
        try:
            provider_name = settings.llm.provider.lower()
        except AttributeError as e:
            raise ValueError(
                "缺少必需的配置：settings.llm.provider。"
                "请确保在 settings.yaml 中指定了 'llm.provider'"
            ) from e

        # 在注册表中查找提供商类
        provider_class = cls._PROVIDERS.get(provider_name)

        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "无"
            raise ValueError(
                f"不支持的 LLM 提供商：'{provider_name}'。"
                f"可用提供商：{available}。"
                f"提供商实现将在任务 B7.1-B7.2 中添加。"
            )

        # 实例化提供商
        # 提供商类应接受设置和可选的 kwargs
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"无法实例化 LLM 提供商 '{provider_name}'：{e}"
            ) from e

    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的提供商名称。

        返回：
            可用提供商标识符的排序列表。
        """
        return sorted(cls._PROVIDERS.keys())

    @classmethod
    def register_vision_provider(
        cls,
        name: str,
        provider_class: type[BaseVisionLLM]
    ) -> None:
        """注册一个新的视觉 LLM 提供商实现。

        此方法允许视觉 LLM 提供商实现向工厂注册
        自身，支持可扩展性。

        参数：
            name: 提供商标识符（例如，'azure'、'ollama'）。
            provider_class: 实现该提供商的 BaseVisionLLM 子类。

        异常：
            ValueError: 如果 provider_class 没有继承 BaseVisionLLM。
        """
        if not issubclass(provider_class, BaseVisionLLM):
            raise ValueError(
                f"提供商类 {provider_class.__name__} 必须继承自 BaseVisionLLM"
            )
        cls._VISION_PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create_vision_llm(
        cls,
        settings: Settings,
        **override_kwargs: Any
    ) -> BaseVisionLLM:
        """基于配置创建视觉 LLM 实例。

        视觉 LLM 支持多模态输入（文本 + 图像），用于
        图像描述、视觉问答和含嵌入式图像的文档理解等任务。

        参数：
            settings: 包含视觉 LLM 配置的应用设置。
            **override_kwargs: 可选参数以覆盖配置值。

        返回：
            已配置视觉 LLM 提供商的实例。

        异常：
            ValueError: 如果配置的提供商不受支持或配置缺失。
            RuntimeError: 如果提供商实例化失败。

        示例：
            >>> settings = Settings.load('config/settings.yaml')
            >>> vision_llm = LLMFactory.create_vision_llm(settings)
            >>> image = ImageInput(path="diagram.png")
            >>> response = vision_llm.chat_with_image("描述此图", image)
        """
        # 从设置中提取提供商名称
        # 视觉 LLM 配置可能嵌套在 settings.vision_llm 或 settings.llm 下
        try:
            # 首先尝试 vision_llm 部分
            if hasattr(settings, 'vision_llm') and hasattr(settings.vision_llm, 'provider'):
                provider_name = settings.vision_llm.provider.lower()
            # 回退到 llm.provider（某些提供商同时支持文本和视觉）
            elif hasattr(settings, 'llm') and hasattr(settings.llm, 'provider'):
                provider_name = settings.llm.provider.lower()
            else:
                raise AttributeError("未找到 vision_llm 或 llm 提供商配置")
        except AttributeError as e:
            raise ValueError(
                "缺少必需的配置：settings.vision_llm.provider 或 settings.llm.provider。"
                "请确保在 settings.yaml 中指定了 'vision_llm.provider' 或 'llm.provider'"
            ) from e

        # 在视觉注册表中查找提供商类
        provider_class = cls._VISION_PROVIDERS.get(provider_name)

        if provider_class is None:
            available = ", ".join(sorted(cls._VISION_PROVIDERS.keys())) if cls._VISION_PROVIDERS else "无"
            raise ValueError(
                f"不支持的视觉 LLM 提供商：'{provider_name}'。"
                f"可用视觉 LLM 提供商：{available}。"
                f"视觉 LLM 实现将在 B9+ 任务中添加。"
            )

        # 实例化提供商
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"无法实例化视觉 LLM 提供商 '{provider_name}'：{e}"
            ) from e

    @classmethod
    def list_vision_providers(cls) -> list[str]:
        """列出所有已注册的视觉 LLM 提供商名称。

        返回：
            可用视觉 LLM 提供商标识符的排序列表。
        """
        return sorted(cls._VISION_PROVIDERS.keys())


# 在模块加载时注册视觉 LLM 提供商
_register_vision_providers()
