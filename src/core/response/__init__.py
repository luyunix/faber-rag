"""
响应模块。

此包包含响应构建组件：
- 响应构建器
- 引用生成器
- 多模态组装器
"""

from src.core.response.citation_generator import Citation, CitationGenerator
from src.core.response.multimodal_assembler import (
    ImageContent,
    ImageReference,
    MultimodalAssembler,
)
from src.core.response.response_builder import MCPToolResponse, ResponseBuilder

__all__ = [
    "Citation",
    "CitationGenerator",
    "ImageContent",
    "ImageReference",
    "MCPToolResponse",
    "MultimodalAssembler",
    "ResponseBuilder",
]
