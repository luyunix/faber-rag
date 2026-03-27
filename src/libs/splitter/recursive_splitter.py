"""使用 LangChain 的递归文本分割器实现。

本模块提供基于递归字符的文本分割策略，
尊重文档结构（标题、代码块），并分层分割文本以维护语义连贯性。
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None  # type: ignore[misc, assignment]

from src.libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """递归字符文本分割器。
    
    此分割器使用 LangChain 的 RecursiveCharacterTextSplitter，
    通过按顺序尝试不同的分隔符（段落、句子、单词）来分割文本，
    同时尊重 Markdown 结构元素，如标题和代码块。
    
    应用的设计原则：
    - 可插拔：实现 BaseSplitter 接口以便工厂实例化。
    - 配置驱动：从 settings 读取 chunk_size 和 chunk_overlap。
    - 快速失败：如果未安装 langchain-text-splitters 则抛出 ImportError。
    - 优雅降级：验证输入并提供清晰的错误消息。
    
    属性:
        chunk_size: 每个分块的最大字符数。
        chunk_overlap: 分块之间的重叠字符数。
        separators: 要尝试的分隔符列表（默认为 Markdown 感知分隔符）。
        
    抛出异常:
        ImportError: 如果未安装 langchain-text-splitters 包。
    """
    
    DEFAULT_SEPARATORS = [
        "\n\n",  # Double newline (paragraphs)
        "\n",    # Single newline
        ". ",    # Sentence endings
        "! ",
        "? ",
        "; ",
        ", ",
        " ",     # Spaces
        "",      # Characters
    ]
    
    def __init__(
        self,
        settings: Any,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 RecursiveSplitter。
        
        参数:
            settings: 包含摄取配置的应用设置。
            chunk_size: 可选的 chunk size 覆盖（默认为 settings.ingestion.chunk_size）。
            chunk_overlap: 可选的重叠覆盖（默认为 settings.ingestion.chunk_overlap）。
            separators: 可选的分隔符字符串列表（默认为 Markdown 感知分隔符）。
            **kwargs: 传递给 LangChain 分割器的额外参数。
        
        抛出异常:
            ImportError: 如果未安装 langchain-text-splitters。
            ValueError: 如果 chunk_size 或 chunk_overlap 无效。
        """
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain-text-splitters is not installed. "
                "Install it with: pip install langchain-text-splitters"
            )
        
        self.settings = settings
        
        # 从 settings 中提取配置，带覆盖
        try:
            ingestion_config = settings.ingestion
            self.chunk_size = chunk_size if chunk_size is not None else ingestion_config.chunk_size
            self.chunk_overlap = chunk_overlap if chunk_overlap is not None else ingestion_config.chunk_overlap
        except AttributeError as e:
            raise ValueError(
                "Missing ingestion configuration in settings. "
                "Expected settings.ingestion.chunk_size and settings.ingestion.chunk_overlap"
            ) from e
        
        # 验证配置
        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got: {self.chunk_size}")
        
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be a non-negative integer, got: {self.chunk_overlap}")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS
        
        # 初始化 LangChain 分割器
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
            **kwargs,
        )
    
    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """递归分割文本为分块。
        
        此方法通过分层尝试不同的分隔符来分割文本，
        保持文档结构，如 Markdown 标题和代码块。
        
        参数:
            text: 要分割的输入文本。必须是非空字符串。
            trace: 可选的 TraceContext 用于可观测性（保留给 Stage F）。
            **kwargs: 额外参数（当前未使用，保留供未来扩展）。
        
        返回:
            文本分块列表。每个分块尊重配置的 chunk_size 和 chunk_overlap。
            顺序保持原始文本序列。
        
        抛出异常:
            ValueError: 如果输入文本无效（空、错误类型）。
            RuntimeError: 如果分割意外失败。
        
        示例:
            >>> splitter = RecursiveSplitter(settings)
            >>> chunks = splitter.split_text("# Header\\n\\nParagraph 1.\\n\\nParagraph 2.")
            >>> len(chunks)
            1  # 如果文本适合 chunk_size
        """
        # 验证输入
        self.validate_text(text)
        
        try:
            # 执行分割
            chunks = self._splitter.split_text(text)
            
            # 处理边界情况：LangChain 可能为非常短的文本返回空列表
            if not chunks:
                chunks = [text]
            
            # 验证输出
            self.validate_chunks(chunks)
            
            return chunks
            
        except Exception as e:
            # 捕获任何 LangChain 错误并提供上下文
            raise RuntimeError(
                f"RecursiveSplitter failed to split text: {e}. "
                f"Text length: {len(text)}, chunk_size: {self.chunk_size}, "
                f"chunk_overlap: {self.chunk_overlap}"
            ) from e
