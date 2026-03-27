"""OpenAI 嵌入实现。

本模块提供与标准 OpenAI 嵌入 API 配合使用的 OpenAI 嵌入实现。
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding


class OpenAIEmbeddingError(RuntimeError):
    """当 OpenAI 嵌入 API 调用失败时抛出。"""


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI 嵌入提供商实现。

    此类为 OpenAI 的嵌入 API 实现 BaseEmbedding 接口。
    它支持 text-embedding-3-small、text-embedding-3-large 和旧模型
    如 text-embedding-ada-002。

    属性：
        api_key: 用于认证的 API 密钥。
        model: 要使用的模型标识符。
        dimensions: 可选的维度缩减（仅适用于 text-embedding-3-*）。
        base_url: API 的基本 URL（默认：OpenAI 端点）。

    示例：
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> embedding = OpenAIEmbedding(settings)
        >>> vectors = embedding.embed(["hello world", "test"])
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 OpenAI 嵌入提供商。

        参数：
            settings: 包含嵌入配置的应用设置。
            api_key: 可选的 API 密钥覆盖（优先于 settings.embedding.api_key 或环境变量）。
            base_url: 可选的基本 URL 覆盖。
            **kwargs: 额外的配置覆盖。

        异常：
            ValueError: 如果未提供 API 密钥且未在环境中找到。

        注意：
            当设置中存在 azure_endpoint 时，提供商会自动
            构造 Azure 兼容的 OpenAI URL 并使用 api-key 认证。
        """
        self.model = settings.embedding.model

        # 提取可选的维度设置
        self.dimensions = getattr(settings.embedding, 'dimensions', None)

        # API 密钥：显式 > 设置 > 环境变量
        self.api_key = (
            api_key
            or getattr(settings.embedding, 'api_key', None)
            or os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "未提供 OpenAI API 密钥。请在 settings.yaml（embedding.api_key）中设置，"
                "设置 OPENAI_API_KEY 环境变量，或传递 api_key 参数。"
            )

        # Azure 兼容模式检测
        azure_endpoint = getattr(settings.embedding, 'azure_endpoint', None)
        self.api_version = getattr(settings.embedding, 'api_version', None)
        self._use_azure_auth = False

        if base_url:
            self.base_url = base_url
        else:
            settings_base_url = getattr(settings.embedding, 'base_url', None)
            self.base_url = settings_base_url if settings_base_url else self.DEFAULT_BASE_URL

        # 存储任何额外的 kwargs 以备将来使用
        self._extra_config = kwargs

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """使用 OpenAI API 为一批文本生成嵌入。

        参数：
            texts: 要嵌入的文本字符串列表。不能为空。
            trace: 可选的 TraceContext 用于可观测性（预留用于阶段 F）。
            **kwargs: 覆盖参数（dimensions 等）。

        返回：
            嵌入向量列表，每个向量是一个浮点数列表。
            外层列表的长度与 len(texts) 匹配。

        异常：
            ValueError: 如果文本列表为空或包含无效条目。
            OpenAIEmbeddingError: 如果 API 调用失败。
        """
        # 验证输入
        self.validate_texts(texts)

        # 延迟导入 OpenAI 客户端（避免在模块级别产生依赖）
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "未安装 OpenAI Python 包。"
                "使用以下命令安装：pip install openai"
            ) from e

        # 初始化 OpenAI 客户端
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
        # Azure 兼容模式：添加 api-version 查询参数和 api-key 请求头
        if self._use_azure_auth and self.api_version:
            client_kwargs["default_query"] = {"api-version": self.api_version}
            client_kwargs["default_headers"] = {"api-key": self.api_key}

        client = OpenAI(**client_kwargs)

        # 准备 API 调用参数
        api_params = {
            "input": texts,
            "model": self.model,
        }

        # 如果指定了维度，则添加（仅适用于 text-embedding-3-* 模型）
        # text-embedding-ada-002 不支持 dimensions 参数
        dimensions = kwargs.get("dimensions", self.dimensions)
        if dimensions is not None and (
            self.model.startswith("text-embedding-3") or
            self.model.startswith("text-embedding-v")
        ):
            api_params["dimensions"] = dimensions

        # 调用 OpenAI API
        try:
            response = client.embeddings.create(**api_params)
        except Exception as e:
            raise OpenAIEmbeddingError(
                f"OpenAI 嵌入 API 调用失败：{e}"
            ) from e

        # 从响应中提取嵌入
        # 响应格式：response.data 是具有 .embedding 属性的对象列表
        try:
            embeddings = [item.embedding for item in response.data]
        except (AttributeError, KeyError) as e:
            raise OpenAIEmbeddingError(
                f"无法解析 OpenAI 嵌入 API 响应：{e}"
            ) from e

        # 验证输出长度与输入长度匹配
        if len(embeddings) != len(texts):
            raise OpenAIEmbeddingError(
                f"输出长度不匹配：期望 {len(texts)}，实际得到 {len(embeddings)}"
            )

        return embeddings

    def get_dimension(self) -> Optional[int]:
        """获取已配置模型的嵌入维度。

        返回：
            嵌入维度，或如果不确定则返回 None。

        注意：
            对于具有自定义维度的 text-embedding-3-* 模型，返回
            配置的维度。对于其他模型，返回其默认值。
        """
        # 如果显式配置了维度，则返回它
        if self.dimensions is not None:
            return self.dimensions

        # 模型特定的默认值
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            # Qwen 嵌入模型（DashScope）
            "text-embedding-v1": 1536,
            "text-embedding-v2": 1536,
            "text-embedding-v3": 1024,
        }

        return model_dimensions.get(self.model)
