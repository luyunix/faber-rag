"""引用生成器，用于生成结构化的引用信息。

本模块从检索结果生成引用信息，
使 MCP 工具能够返回正确格式的引用，
供 AI 助手用于来源归属。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.core.types import RetrievalResult


@dataclass
class Citation:
    """表示单个引用/参考。

    属性：
        index: 引用索引号（从 1 开始，显示为 [1]、[2] 等）
        chunk_id: 源块（chunk）的唯一标识符
        source: 源文件路径或文档名称
        page: 源文档中的页码（如果适用）
        score: 检索的相关性分数
        text_snippet: 所引用内容的简短摘录
        metadata: 其他元数据（标题、章节等）
    """
    index: int
    chunk_id: str
    source: str
    score: float
    text_snippet: str
    page: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于 JSON 序列化。"""
        result = {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "score": round(self.score, 4),
            "text_snippet": self.text_snippet,
        }
        if self.page is not None:
            result["page"] = self.page
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class CitationGenerator:
    """从检索结果生成引用信息。

    本类将 RetrievalResult 对象转换为带有
    正确索引和元数据提取的 Citation 对象。

    示例：
        >>> generator = CitationGenerator()
        >>> results = [RetrievalResult(chunk_id="doc1_001", score=0.95, ...)]
        >>> citations = generator.generate(results)
        >>> print(citations[0].index)  # 1
        >>> print(citations[0].source)  # "docs/guide.pdf"
    """

    def __init__(
        self,
        snippet_max_length: int = 200,
        include_metadata_fields: Optional[List[str]] = None,
    ) -> None:
        """初始化 CitationGenerator。

        参数：
            snippet_max_length: text_snippet 的最大字符数（默认：200）
            include_metadata_fields: 要包含的元数据字段的可选列表
                为 None 时，包含 'title'、'section'、'chunk_index'
        """
        self.snippet_max_length = snippet_max_length
        self.include_metadata_fields = include_metadata_fields or [
            "title", "section", "chunk_index", "doc_type"
        ]

    def generate(self, results: List[RetrievalResult]) -> List[Citation]:
        """从检索结果生成引用。

        参数：
            results: 来自搜索的 RetrievalResult 对象列表

        返回：
            带有从 1 开始索引的 Citation 对象列表
        """
        citations = []

        for idx, result in enumerate(results, start=1):
            citation = self._create_citation(idx, result)
            citations.append(citation)

        return citations

    def _create_citation(self, index: int, result: RetrievalResult) -> Citation:
        """从单个 RetrievalResult 创建 Citation。

        参数：
            index: 从 1 开始的引用索引
            result: 要转换的 RetrievalResult

        返回：
            带有提取信息的 Citation 对象
        """
        metadata = result.metadata or {}

        # 提取源路径
        source = metadata.get("source_path", "unknown")

        # 提取页码（可能是整数或字符串）
        page = metadata.get("page") or metadata.get("page_num")
        if page is not None:
            try:
                page = int(page)
            except (ValueError, TypeError):
                page = None

        # 生成文本片段
        text_snippet = self._generate_snippet(result.text)

        # 提取选定的元数据字段
        extra_metadata = {}
        for field_name in self.include_metadata_fields:
            if field_name in metadata and field_name not in ("source_path", "page", "page_num"):
                extra_metadata[field_name] = metadata[field_name]

        return Citation(
            index=index,
            chunk_id=result.chunk_id,
            source=source,
            score=result.score,
            text_snippet=text_snippet,
            page=page,
            metadata=extra_metadata,
        )

    def _generate_snippet(self, text: str) -> str:
        """从文本生成截断的片段。

        参数：
            text: 完整文本内容

        返回：
            如有需要则截断并添加省略号的文本
        """
        if not text:
            return ""

        # 清理空白字符
        cleaned = " ".join(text.split())

        if len(cleaned) <= self.snippet_max_length:
            return cleaned

        # 截断并添加省略号
        truncated = cleaned[:self.snippet_max_length].rsplit(" ", 1)[0]
        return truncated + "..."

    def format_citation_marker(self, index: int) -> str:
        """格式化用于内联的引用标记。

        参数：
            index: 从 1 开始的引用索引

        返回：
            格式化的标记如 "[1]"
        """
        return f"[{index}]"
