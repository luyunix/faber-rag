"""
LLM 模块。

该包包含 LLM 客户端的抽象和实现：
- 基础 LLM 类（纯文本）
- 基础视觉 LLM 类（多模态：文本 + 图像）
- LLM 工厂
- 提供商实现（OpenAI、DeepSeek、Qwen）
"""

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.openai_llm import OpenAILLM, OpenAILLMError
from src.libs.llm.openai_vision_llm import OpenAIVisionLLM, OpenAIVisionLLMError
from src.libs.llm.deepseek_llm import DeepSeekLLM, DeepSeekLLMError
from src.libs.llm.qwen_llm import QwenLLM, QwenLLMError
from src.libs.llm.qwen_vision_llm import QwenVisionLLM, QwenVisionLLMError

# 将纯文本 LLM 提供商注册到工厂
# 确保类属性已初始化
if not hasattr(LLMFactory, '_PROVIDERS'):
    LLMFactory._PROVIDERS = {}

LLMFactory.register_provider("openai", OpenAILLM)
LLMFactory.register_provider("deepseek", DeepSeekLLM)
LLMFactory.register_provider("qwen", QwenLLM)

# 注意：视觉 LLM 提供商在 llm_factory.py 中注册

__all__ = [
    # 基础类
    "BaseLLM",
    "BaseVisionLLM",
    # 数据类型
    "ChatResponse",
    "Message",
    "ImageInput",
    # 工厂
    "LLMFactory",
    # 纯文本 LLM 实现
    "OpenAILLM",
    "OpenAILLMError",
    "DeepSeekLLM",
    "DeepSeekLLMError",
    "QwenLLM",
    "QwenLLMError",
    # 视觉 LLM 实现
    "OpenAIVisionLLM",
    "OpenAIVisionLLMError",
    "QwenVisionLLM",
    "QwenVisionLLMError",
]
