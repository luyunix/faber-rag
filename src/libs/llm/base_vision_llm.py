"""视觉 LLM 提供商的抽象基类。

本模块定义了视觉语言模型提供商的可插拔接口，
通过配置实现文本 + 图像的多模态交互，
支持在不同后端（Azure Vision、Ollama Vision 等）之间无缝切换。

视觉 LLM 扩展了标准 LLM，除了支持文本提示外还支持图像输入，
支持图像描述、视觉问答和
含嵌入式图像的文档理解等任务。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from src.libs.llm.base_llm import ChatResponse, Message


@dataclass
class ImageInput:
    """表示视觉 LLM 的图像输入。

    支持多种输入格式：
    - 文件路径：本地图像文件，将读取并编码
    - 字节：原始图像字节（已加载）
    - Base64：已编码的图像字符串

    属性：
        path: 图像文件的磁盘路径（如果从磁盘加载）。
        data: 原始图像字节（如果已加载）。
        base64: Base64 编码的图像字符串（如果已编码）。
        mime_type: 图像的 MIME 类型（例如，'image/png'、'image/jpeg'）。
    """
    path: Optional[Union[str, Path]] = None
    data: Optional[bytes] = None
    base64: Optional[str] = None
    mime_type: str = "image/png"

    def __post_init__(self) -> None:
        """验证恰好提供了一种输入格式。"""
        provided_inputs = sum([
            self.path is not None,
            self.data is not None,
            self.base64 is not None,
        ])
        if provided_inputs == 0:
            raise ValueError("必须提供以下之一：path、data 或 base64")
        if provided_inputs > 1:
            raise ValueError("必须恰好提供以下之一：path、data 或 base64")


class BaseVisionLLM(ABC):
    """视觉 LLM 提供商的抽象基类。

    视觉 LLM 接受文本和图像输入，支持多模态
    理解任务，如图像描述、视觉问答和
    含嵌入式图像的文档分析。

    所有视觉 LLM 实现必须继承自此类并实现 chat_with_image() 方法。
    这确保了不同提供商（Azure Vision、Ollama Vision 等）之间的一致性接口。

    应用的设计原则：
    - 可插拔：子类可以更改而无需修改上游代码。
    - 可观测：接受可选的 TraceContext 以集成可观测性。
    - 配置驱动：通过工厂基于设置创建实例。
    - 接口隔离：最小化接口，专注于多模态输入。
    - 扩展点：图像预处理（压缩、格式转换）可以在
      子类中添加，而无需更改基础接口。
    """

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: Optional[list[Message]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """基于文本提示和图像输入生成响应。

        此方法支持多模态交互，模型可以"看到"
        图像并回答关于它的问题或生成描述。

        参数：
            text: 关于图像的文本提示或问题。
            image: 图像输入（路径、字节或 base64）。
            messages: 可选的对话历史上下文。如果提供，
                文本 + 图像将作为最新的用户消息追加。
            trace: 可选的 TraceContext 用于可观测性（预留用于阶段 F）。
            **kwargs: 提供商特定的参数（temperature、max_tokens 等）。

        返回：
            包含生成文本和元数据的 ChatResponse。

        异常：
            ValueError: 如果文本为空或图像输入无效。
            RuntimeError: 如果视觉 LLM 提供商调用失败。

        示例：
            >>> image = ImageInput(path="diagram.png")
            >>> response = vision_llm.chat_with_image(
            ...     text="描述此图表",
            ...     image=image
            ... )
            >>> print(response.content)
            "此图表展示了系统架构，包含..."
        """
        pass

    def validate_text(self, text: str) -> None:
        """验证文本提示。

        参数：
            text: 要验证的文本提示。

        异常：
            ValueError: 如果文本为空或不是字符串。
        """
        if not isinstance(text, str):
            raise ValueError(f"文本必须是字符串，得到 {type(text).__name__}")
        if not text or not text.strip():
            raise ValueError("文本提示不能为空")

    def validate_image(self, image: ImageInput) -> None:
        """验证图像输入。

        参数：
            image: 要验证的图像输入。

        异常：
            ValueError: 如果图像不是 ImageInput 实例。
        """
        if not isinstance(image, ImageInput):
            raise ValueError(
                f"图像必须是 ImageInput 实例，得到 {type(image).__name__}"
            )

    def preprocess_image(
        self,
        image: ImageInput,
        max_size: Optional[tuple[int, int]] = None,
    ) -> ImageInput:
        """在发送到视觉 LLM 之前预处理图像。

        此方法为图像预处理提供扩展点，例如：
        - 调整大小以满足提供商的大小限制
        - 格式转换（例如，PNG 转 JPEG）
        - 压缩以减少有效载荷大小

        默认实现返回未更改的图像。子类可以
        覆盖以添加提供商特定的预处理。

        参数：
            image: 要预处理的输入图像。
            max_size: 可选的最大尺寸（宽度、高度），单位为像素。

        返回：
            预处理的 ImageInput（如果无需更改，可能与原实例相同）。

        注意：
            预处理应该是幂等的 - 使用相同的输入多次调用
            应该产生相同的输出。
        """
        # 默认：不预处理
        return image
