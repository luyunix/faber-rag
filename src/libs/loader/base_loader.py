"""文档加载器的抽象基类。

本模块定义了文档加载器的可插拔接口，
支持无缝加载不同格式的文档（PDF、Markdown 等），
并输出统一的结构。

设计原则：
- 单一职责：加载器仅处理格式统一 + 结构提取
- 类型安全：返回 core.types 中定义的标准 Document 类型
- 不分块：加载器不对文档分块，仅解析和规范化
- 优雅降级：可选功能（如图像提取）的失败不应阻止文本解析
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.core.types import Document


class BaseLoader(ABC):
    """文档加载器的抽象基类。
    
    所有加载器必须实现 load() 方法以解析文件
    并返回标准化的 Document 对象，包含：
    - text: 规范化的内容（优先使用 Markdown 格式）
    - metadata: 至少必须包含 'source_path'
    
    加载器应处理：
    - 特定格式的解析逻辑
    - 元数据提取（标题、页数等）
    - 结构规范化（尽可能转为 Markdown）
    - 可选：图像提取和占位符插入
    """
    
    @abstractmethod
    def load(self, file_path: str | Path) -> Document:
        """加载并解析文档文件。
        
        参数：
            file_path: 要加载的文档文件路径。
            
        返回：
            包含解析内容和元数据的 Document 对象。
            metadata 必须至少包含 'source_path'。
            
        异常：
            FileNotFoundError: 如果文件不存在。
            ValueError: 如果文件格式无效或不支持。
            RuntimeError: 如果解析严重失败。
            
        示例：
            >>> loader = PdfLoader()
            >>> doc = loader.load("data/documents/report.pdf")
            >>> assert "source_path" in doc.metadata
            >>> assert doc.text  # 非空文本
        """
        pass
    
    @staticmethod
    def _validate_file(file_path: str | Path) -> Path:
        """验证文件存在且可读。
        
        参数：
            file_path: 要验证的路径。
            
        返回：
            解析后的 Path 对象。
            
        异常：
            FileNotFoundError: 如果文件不存在。
            PermissionError: 如果文件不可读。
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return path
