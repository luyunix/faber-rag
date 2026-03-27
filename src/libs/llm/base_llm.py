"""LLM 提供商的抽象基类。

本模块定义了语言模型提供商的可插拔接口，
支持通过配置驱动的实例化，在不同后端（OpenAI、Azure、Ollama 等）之间无缝切换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """表示聊天对话中的单条消息。

    属性：
        role: 消息发送者的角色（'system'、'user' 或 'assistant'）。
        content: 消息的文本内容。
    """
    role: str
    content: str


@dataclass
class ChatResponse:
    """来自 LLM 聊天的响应。

    属性：
        content: 生成的文本响应。
        model: 生成响应的模型标识符。
        usage: 可选的令牌使用统计（prompt_tokens、completion_tokens、total_tokens）。
        raw_response: 可选的提供商原始响应，用于调试。
    """
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


class BaseLLM(ABC):
    """LLM 提供商的抽象基类。

    所有 LLM 实现必须继承自此类并实现 chat() 方法。
    这确保了不同提供商（OpenAI、Azure、DeepSeek、Ollama 等）之间的一致性接口。

    应用的设计原则：
    - 可插拔：子类可以更改而无需修改上游代码。
    - 可观测：接受可选的 TraceContext 以集成可观测性。
    - 配置驱动：通过工厂基于设置创建实例。
    """

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """生成聊天补全响应。

        参数：
            messages: 对话消息列表（角色 + 内容）。
            trace: 可选的 TraceContext 用于可观测性（预留用于阶段 F）。
            **kwargs: 提供商特定的参数（temperature、max_tokens 等）。

        返回：
            包含生成文本和元数据的 ChatResponse。

        异常：
            ValueError: 如果消息列表为空或格式错误。
            RuntimeError: 如果 LLM 提供商调用失败。
        """
        pass

    def validate_messages(self, messages: List[Message]) -> None:
        """验证消息列表结构。

        参数：
            messages: 要验证的消息列表。

        异常：
            ValueError: 如果消息列表为空或包含无效角色。
        """
        if not messages:
            raise ValueError("消息列表不能为空")

        valid_roles = {"system", "user", "assistant"}
        for i, msg in enumerate(messages):
            if not isinstance(msg, Message):
                raise ValueError(f"索引 {i} 处的消息不是 Message 实例")
            if msg.role not in valid_roles:
                raise ValueError(
                    f"索引 {i} 处的消息具有无效的角色 '{msg.role}'。"
                    f"必须是以下之一：{valid_roles}"
                )
            if not msg.content or not msg.content.strip():
                raise ValueError(f"索引 {i} 处的消息内容为空")
