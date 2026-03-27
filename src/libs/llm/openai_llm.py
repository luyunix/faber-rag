"""OpenAI 兼容 LLM 实现。

本模块提供 OpenAI LLM 实现，使用标准 OpenAI API。
也可以通过配置 base_url 使用其他 OpenAI 兼容的端点。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class OpenAILLMError(RuntimeError):
    """当 OpenAI API 调用失败时抛出。"""


class OpenAILLM(BaseLLM):
    """OpenAI LLM 提供商实现。
    
    此类为 OpenAI 的 chat completion API 实现 BaseLLM 接口。
    它支持标准 OpenAI API 和任何 OpenAI 兼容的端点。
    
    属性:
        api_key: 用于认证的 API 密钥。
        base_url: API 的基 URL（默认：OpenAI 的端点）。
        model: 要使用的模型标识符。
        default_temperature: 默认的生成温度。
        default_max_tokens: 默认的最大 token 数。
    
    示例:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> llm = OpenAILLM(settings)
        >>> response = llm.chat([Message(role='user', content='Hello')])
    """
    
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 OpenAI LLM 提供商。
        
        参数:
            settings: 包含 LLM 配置的应用设置。
            api_key: 可选的 API 密钥覆盖（回退到 settings.llm.api_key 或环境变量）。
            base_url: 可选的 base URL 覆盖。
            **kwargs: 额外的配置覆盖。
        
        抛出:
            ValueError: 如果未提供 API 密钥且在环境中未找到。
        
        注意:
            当 settings 中存在 azure_endpoint 时，提供商会自动构建
            Azure 兼容的 OpenAI URL 并使用 api-key auth header。
        """
        self.model = settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens
        
        # API 密钥：显式 > settings > 环境变量
        self.api_key = (
            api_key
            or getattr(settings.llm, 'api_key', None)
            or os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set in settings.yaml (llm.api_key), "
                "OPENAI_API_KEY environment variable, or pass api_key parameter."
            )
        
        # Azure 兼容模式检测
        azure_endpoint = getattr(settings.llm, 'azure_endpoint', None)
        self.api_version = getattr(settings.llm, 'api_version', None)

        # base_url: explicit param > settings.llm.base_url > azure_endpoint > default
        settings_base_url = getattr(settings.llm, 'base_url', None)

        if base_url:
            self.base_url = base_url
            self._use_azure_auth = False
        elif settings_base_url:
            self.base_url = settings_base_url
            self._use_azure_auth = False
        elif azure_endpoint:
            # Azure 兼容模式：构建基于部署的 URL
            deployment = getattr(settings.llm, 'deployment_name', None) or self.model
            self.base_url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{deployment}"
            self._use_azure_auth = True
            if not self.api_version:
                self.api_version = "2024-02-15-preview"
        else:
            self.base_url = self.DEFAULT_BASE_URL
            self._use_azure_auth = False
        
        # 存储任何额外的 kwargs 以备将来使用
        self._extra_config = kwargs
    
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """使用 OpenAI API 生成聊天完成。
        
        参数:
            messages: 对话消息列表
            trace: 可选的 TraceContext 用于可观测性 (保留给 Stage F)
            **kwargs: 覆盖参数 (temperature, max_tokens 等)
        
        返回:
            ChatResponse 包含生成的内容和元数据
        
        抛出:
            ValueError: 如果消息无效
            OpenAILLMError: 如果 API 调用失败
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
            raise OpenAILLMError(
                f"[OpenAI] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, OpenAILLMError):
                raise
            raise OpenAILLMError(
                f"[OpenAI] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """向 OpenAI 发送实际 API 调用。
        
        此方法分离出来以便在测试中轻松模拟。
        
        参数:
            messages: API 格式的消息
            model: 模型标识符
            temperature: 生成温度
            max_tokens: 要生成的最大令牌数
        
        返回:
            原始 API 响应 (字典)
        
        抛出:
            OpenAILLMError: 如果 API 调用失败
        """
        import httpx
        
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        if self.api_version:
            url += f"?api-version={self.api_version}"
        
        if self._use_azure_auth:
            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json",
            }
        else:
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
                    raise OpenAILLMError(
                        f"[OpenAI] API error (HTTP {response.status_code}): {error_detail}"
                    )
                
                return response.json()
        except httpx.TimeoutException as e:
            raise OpenAILLMError(
                f"[OpenAI] Request timed out after 60 seconds"
            ) from e
        except httpx.RequestError as e:
            raise OpenAILLMError(
                f"[OpenAI] Connection failed: {type(e).__name__}: {e}"
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
