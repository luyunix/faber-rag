"""Qwen Vision LLM implementation via DashScope OpenAI-compatible API.

This module provides integration with Alibaba Cloud's Qwen-VL (通义千问视觉)
multimodal large language models for image understanding tasks.

Supported models:
- qwen-vl-plus: Balanced performance and cost
- qwen-vl-max: Most capable vision model

Use cases:
- Image captioning (generating descriptions for images)
- Visual question answering (VQA)
- Document understanding with embedded images
- Chart and diagram analysis

Usage:
    from src.libs.llm.qwen_vision_llm import QwenVisionLLM
    from src.core.settings import load_settings
    from src.libs.llm.base_vision_llm import ImageInput
    
    settings = load_settings()
    vision_llm = QwenVisionLLM(settings)
    image = ImageInput(path="diagram.png")
    response = vision_llm.chat_with_image(
        text="Describe this image",
        image=image
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.llm.openai_vision_llm import OpenAIVisionLLM

if TYPE_CHECKING:
    from src.core.settings import Settings


class QwenVisionLLMError(Exception):
    """Exception raised by QwenVisionLLM."""
    pass


class QwenVisionLLM(OpenAIVisionLLM):
    """Qwen Vision LLM provider using OpenAI-compatible API.
    
    This class inherits from OpenAIVisionLLM since Qwen-VL's DashScope API
    is fully compatible with OpenAI's Vision API format. The main differences
    are:
    - Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
    - API Key: DashScope API key (starts with 'sk-')
    - Models: qwen-vl-plus, qwen-vl-max
    
    All vision capabilities (image captioning, chat with images) are
    inherited from OpenAIVisionLLM.
    """
    
    def __init__(self, settings: Settings, **override_kwargs: Any) -> None:
        """Initialize Qwen Vision LLM.
        
        Args:
            settings: Application settings with Vision LLM configuration.
            **override_kwargs: Optional overrides for config values.
        """
        # Qwen-VL uses OpenAI-compatible API, so we can reuse OpenAIVisionLLM        super().__init__(settings, **override_kwargs)
    
    def __repr__(self) -> str:
        """Return string representation of the Qwen Vision LLM."""
        return f"QwenVisionLLM(model={self.model_name})"
