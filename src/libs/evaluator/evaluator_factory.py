"""用于创建评估器提供商实例的工厂。

本模块实现工厂模式，根据配置实例化合适的
评估器提供商。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.evaluator.base_evaluator import BaseEvaluator, NoneEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator

if TYPE_CHECKING:
    from src.core.settings import Settings


def _get_ragas_evaluator() -> type[BaseEvaluator]:
    """延迟导入 RagasEvaluator 以避免对 ragas 的硬依赖。"""
    from src.observability.evaluation.ragas_evaluator import RagasEvaluator
    return RagasEvaluator


def _get_composite_evaluator() -> type[BaseEvaluator]:
    """延迟导入 CompositeEvaluator。"""
    from src.observability.evaluation.composite_evaluator import CompositeEvaluator
    return CompositeEvaluator


class EvaluatorFactory:
    """用于创建评估器提供商实例的工厂。

    应用的设计原则：
    - 工厂模式：集中对象创建逻辑。
    - 配置驱动：基于 settings.yaml 选择提供商。
    - 回退：禁用的评估返回 NoneEvaluator。
    - 快速失败：对未知提供商抛出明确的错误。
    """

    _PROVIDERS: dict[str, type[BaseEvaluator]] = {
        "custom": CustomEvaluator,
    }

    # 延迟加载的提供商（按需导入以避免硬依赖）
    _LAZY_PROVIDERS: dict[str, Any] = {
        "ragas": _get_ragas_evaluator,
        "composite": _get_composite_evaluator,
    }

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseEvaluator]) -> None:
        """注册一个新的评估器提供商实现。

        参数：
            name: 提供商标识符（例如，'ragas'、'custom'）。
            provider_class: 实现该提供商的 BaseEvaluator 子类。

        异常：
            ValueError: 如果 provider_class 没有继承 BaseEvaluator。
        """
        if not issubclass(provider_class, BaseEvaluator):
            raise ValueError(
                f"提供商类 {provider_class.__name__} 必须继承自 BaseEvaluator"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseEvaluator:
        """基于配置创建评估器实例。

        参数：
            settings: 包含评估配置的应用设置。
            **override_kwargs: 可选参数以覆盖配置值。

        返回：
            已配置评估器提供商的实例。

        异常：
            ValueError: 如果配置的提供商不受支持或缺失。
            RuntimeError: 如果提供商初始化失败。
        """
        try:
            # 接受完整的 Settings（带 .evaluation 属性）或
            # 直接接受一个裸的 EvaluationSettings 对象。
            if hasattr(settings, "evaluation"):
                evaluation_settings = settings.evaluation
            elif hasattr(settings, "provider") and hasattr(settings, "enabled"):
                evaluation_settings = settings
            else:
                raise AttributeError("settings 没有 'evaluation' 属性")
            if evaluation_settings is None:
                raise AttributeError("settings.evaluation 为 None")
            provider_name = evaluation_settings.provider.lower()
            enabled = bool(evaluation_settings.enabled)
        except AttributeError as e:
            raise ValueError(
                "缺少必需的配置：settings.evaluation.provider。"
                "请确保在 settings.yaml 中指定了 'evaluation.provider'"
            ) from e

        if not enabled or provider_name in {"none", "disabled"}:
            return NoneEvaluator(settings=settings, **override_kwargs)

        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None and provider_name in cls._LAZY_PROVIDERS:
            try:
                provider_class = cls._LAZY_PROVIDERS[provider_name]()
                cls._PROVIDERS[provider_name] = provider_class  # 缓存以供下次调用
            except ImportError as e:
                raise ValueError(
                    f"提供商 '{provider_name}' 需要额外的依赖项：{e}"
                ) from e
        if provider_class is None:
            all_providers = sorted(set(cls._PROVIDERS.keys()) | set(cls._LAZY_PROVIDERS.keys()))
            available = ", ".join(all_providers) if all_providers else "无"
            raise ValueError(
                f"不支持的评估器提供商：'{provider_name}'。"
                f"可用提供商：{available}。"
            )

        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"无法实例化评估器提供商 '{provider_name}'：{e}"
            ) from e

    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的提供商名称。

        返回：
            可用提供商标识符的排序列表。
        """
        return sorted(cls._PROVIDERS.keys())
