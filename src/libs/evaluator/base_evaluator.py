"""评估器提供商的抽象基类。

本模块定义了评估提供商的可插拔接口，
支持通过配置驱动的实例化，在不同评估后端之间无缝切换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseEvaluator(ABC):
    """评估提供商的抽象基类。

    所有评估器实现必须继承自此类并实现 evaluate() 方法。
    这确保了不同评估后端之间的一致性接口。

    应用的设计原则：
    - 可插拔：子类可以更改而无需修改上游代码。
    - 可观测：接受可选的 TraceContext 以集成可观测性。
    - 配置驱动：通过工厂基于设置创建实例。
    """

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """评估检索和生成质量。

        参数：
            query: 用户查询字符串。
            retrieved_chunks: 要评估的检索到的块或记录。
            generated_answer: 可选的生成答案文本。
            ground_truth: 可选的基准真实数据（ID 或答案）。
            trace: 可选的 TraceContext 用于可观测性（预留用于阶段 F）。
            **kwargs: 提供商特定的参数。

        返回：
            指标名称到浮点数值的字典。

        异常：
            ValueError: 如果输入无效。
            RuntimeError: 如果评估意外失败。
        """
        pass

    def validate_query(self, query: str) -> None:
        """验证查询字符串。

        参数：
            query: 要验证的查询字符串。

        异常：
            ValueError: 如果查询无效。
        """
        if not isinstance(query, str):
            raise ValueError(f"查询必须是字符串，得到 {type(query).__name__}")
        if not query.strip():
            raise ValueError("查询不能为空或仅包含空白字符")

    def validate_retrieved_chunks(self, retrieved_chunks: List[Any]) -> None:
        """验证检索到的块结构。

        参数：
            retrieved_chunks: 要验证的检索到的块列表。

        异常：
            ValueError: 如果检索到的块无效。
        """
        if not isinstance(retrieved_chunks, list):
            raise ValueError("retrieved_chunks 必须是列表")
        if not retrieved_chunks:
            raise ValueError("retrieved_chunks 不能为空")


class NoneEvaluator(BaseEvaluator):
    """不执行操作的评估器，返回空指标。

    当评估禁用时使用此实现。
    """

    def __init__(self, settings: Any = None, **kwargs: Any) -> None:
        self.settings = settings
        self.kwargs = kwargs

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)
        return {}
