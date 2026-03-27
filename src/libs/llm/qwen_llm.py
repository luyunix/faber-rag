"""Qwen LLM implementation via DashScope OpenAI-compatible API.

This module provides integration with Alibaba Cloud's Qwen (通义千问) 
large language models through the DashScope OpenAI-compatible API endpoint.

Supported models:
- qwen-turbo: Fast and cost-effective
- qwen-plus: Balanced performance
- qwen-max: Most capable model
- qwen-coder-plus: Optimized for code generation

Usage:
    from src.libs.llm.qwen_llm import QwenLLM
    from src.core.settings import load_settings
    
    settings = load_settings()
    llm = QwenLLM(settings)
    response = llm.chat([Message(role="user", content="Hello")])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from src.libs.llm.openai_llm import OpenAILLM

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.libs.llm.base_llm import ChatResponse, Message


class QwenLLMError(Exception):
    """Exception raised by QwenLLM."""
    pass


class QwenLLM(OpenAILLM):
    """Qwen LLM provider using OpenAI-compatible API.
    
    This class inherits from OpenAILLM since Qwen's DashScope API
    is fully compatible with OpenAI's API format. The main differences
    are:
    - Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
    - API Key: DashScope API key (starts with 'sk-')
    - Models: qwen-turbo, qwen-plus, qwen-max, etc.
    
    All chat completion functionality is inherited from OpenAILLM.
    """
    
    def __init__(self, settings: Settings, **override_kwargs: Any) -> None:
        """初始化 Qwen LLM.
        
        参数：
            settings: 包含 LLM 配置的应用设置。
            **override_kwargs: 配置值的可选覆盖。
        """
        # Qwen 使用 OpenAI 兼容 API，因此我们可以复用 OpenAILLM
        # 需要确保 settings.llm.api_key、settings.llm.temperature、settings.llm.max_tokens 已设置
        super().__init__(settings, **override_kwargs)
    
    def __repr__(self) -> str:
        """Return string representation of the Qwen LLM."""
        return f"QwenLLM(model={self.model_name})"
