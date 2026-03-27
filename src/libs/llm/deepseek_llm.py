"""DeepSeek LLM 实现。

本模块提供 DeepSeek LLM 实现，使用 DeepSeek 的 API。
DeepSeek 使用 OpenAI 兼容的 API 格式，但有自己的端点和身份验证。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class DeepSeekLLMError(RuntimeError):
    """当 DeepSeek API 调用失败时抛出。"""


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM 提供商实现。
    
    此类为 DeepSeek 的 chat API 实现 BaseLLM 接口。
    DeepSeek 提供 OpenAI 兼容的 API，有自己的端点。
    
    属性:
        api_key: 用于认证的 API 密钥。
        base_url: API 的基 URL。
        model: 要使用的模型标识符。
        default_temperature: 默认的生成温度。
        default_max_tokens: 默认的最大 token 数。
    
    示例:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> llm = DeepSeekLLM(settings)
        >>> response = llm.chat([Message(role='user', content='Hello')])
    """
    
    DEFAULT_BASE_URL = "https://api.deepseek.com"
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 DeepSeek LLM 提供商。
        
        参数:
            settings: 包含 LLM 配置的应用设置。
            api_key: 可选的 API 密钥覆盖（回退到环境变量 DEEPSEEK_API_KEY）。
            base_url: 可选的 base URL 覆盖。
            **kwargs: 额外的配置覆盖。
        
        抛出:
            ValueError: 如果未提供 API 密钥且在环境中未找到。
        """
        self.model = settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens
        
        # API 密钥：显式 > 环境变量
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not provided. Set DEEPSEEK_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # 基础 URL：显式 > 默认值
        self.base_url = base_url or self.DEFAULT_BASE_URL
        
        # 存储任何额外的 kwargs 以备将来使用
        self._extra_config = kwargs
    
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """使用 DeepSeek API 生成聊天完成。
        
        参数:
            messages: 对话消息列表
            trace: 可选的 TraceContext 用于可观测性 (保留给 Stage F)
            **kwargs: 覆盖参数 (temperature, max_tokens 等)
        
        返回:
            ChatResponse 包含生成的内容和元数据
        
        抛出:
            ValueError: 如果消息无效
            DeepSeekLLMError: 如果 API 调用失败
        """
        # 验证输入
        self.validate_messages(messages)
        
        # 准备请求参数
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        model = kwargs.get("model", self.model)
        
        # 将消息转换为 API 格式
        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        # 调用 API
        try:
            response_data = self._call_api(
                messages=api_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # 解析响应
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage")
            
            return ChatResponse(
                content=content,
                model=response_data.get("model", model),
                usage=usage,
                raw_response=response_data,
            )
        except KeyError as e:
            raise DeepSeekLLMError(
                f"[DeepSeek] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, DeepSeekLLMError):
                raise
            raise DeepSeekLLMError(
                f"[DeepSeek] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """向 DeepSeek 发送实际 API 调用。
        
        此方法分离出来以便在测试中轻松模拟。
        
        参数:
            messages: API 格式的消息
            model: 模型标识符
            temperature: 生成温度
            max_tokens: 要生成的最大令牌数
        
        返回:
            原始 API 响应 (字典)
        
        抛出:
            DeepSeekLLMError: 如果 API 调用失败
        """
        import httpx
        
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    error_detail = self._parse_error_response(response)
                    raise DeepSeekLLMError(
                        f"[DeepSeek] API error (HTTP {response.status_code}): {error_detail}"
                    )
                
                return response.json()
        except httpx.TimeoutException as e:
            raise DeepSeekLLMError(
                f"[DeepSeek] Request timed out after 60 seconds"
            ) from e
        except httpx.RequestError as e:
            raise DeepSeekLLMError(
                f"[DeepSeek] Connection failed: {type(e).__name__}: {e}"
            ) from e
    
    def _parse_error_response(self, response: Any) -> str:
        """从 API 响应解析错误详情。
        
        参数:
            response: HTTP 响应对象
        
        返回:
            人类可读的错误消息
        """
        try:
            error_data = response.json()
            if "error" in error_data:
                error = error_data["error"]
                if isinstance(error, dict):
                    return error.get("message", str(error))
                return str(error)
            return response.text
        except Exception:
            return response.text or "Unknown error"
