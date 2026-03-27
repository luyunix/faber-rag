"""整个管道的核心数据类型和契约。

此模块定义了在所有管道阶段中使用的基本数据结构：
- 摄取（加载器、转换、嵌入、存储）
- 检索（查询引擎、搜索、重排序）
- mcp_server（工具、响应格式化）

设计原则：
- 集中式契约：所有阶段使用这些类型以避免耦合
- 可序列化：所有类型支持dict/JSON转换
- 可扩展元数据：最少必填字段，灵活扩展
- 类型安全：完整的类型提示用于静态分析
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class Document:
    """表示从源加载的原始文档。

    这是分割前的加载器（例如 PDF 加载器）的输出。

    属性：
        id: 文档的唯一标识符（例如基于文件哈希或路径的 ID）
        text: 标准化的 Markdown 格式文档内容。
              图片表示为占位符：[IMAGE: {image_id}]
        metadata: 文档级别的元数据，包括：
            - source_path（必填）: 原始文件路径
            - doc_type: 文档类型（例如 'pdf'、'markdown'）
            - title: 提取或推断的文档标题
            - page_count: 总页数（如适用）
            - images: 图片引用列表（见下方的图片字段规范）
            - 其他任何自定义元数据

    图片字段规范（metadata.images）：
        结构：List[{"id": str, "path": str, "page": int, "text_offset": int,
                        "text_length": int, "position": dict}]
        字段：
            - id: 唯一的图片标识符（格式：{doc_hash}_{page}_{seq}）
            - path: 图片文件存储路径（约定：data/images/{collection}/{image_id}.png）
            - page: 原始文档中的页码（可选，适用于 PDF 等分页文档）
            - text_offset: 占位符在 Document.text 中的起始字符位置（从 0 开始）
            - text_length: 占位符字符串的长度（通常为 len("[IMAGE: {image_id}]")）
            - position: 原始文档中的物理位置信息（可选，例如 PDF 坐标、像素位置）
        注意：text_offset 和 text_length 支持精确定位占位符，
              支持同一张图片多次出现的场景

    示例：
        >>> doc = Document(
        ...     id="doc_abc123",
        ...     text="# Title\\n\\nContent...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "doc_type": "pdf",
        ...         "title": "Annual Report 2025"
        ...     }
        ... )
    """
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证必需的元数据字段。"""
        if "source_path" not in self.metadata:
            raise ValueError("Document metadata must contain 'source_path'")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以进行序列化。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """从字典创建 Document。"""
        return cls(**data)


@dataclass
class Chunk:
    """表示文档分割后的文本块。

    这是分割器的输出，也是 Transform 管道的输入。
    每个块保持对其源文档的可追溯性。

    属性：
        id: 唯一的块标识符（例如基于哈希或顺序）
        text: 块内容（原始文档文本的子集）。
              图片表示为占位符：[IMAGE: {image_id}]
        metadata: 从 Document 继承并扩展的块级别元数据：
            - source_path（必填）: 原始文件路径
            - chunk_index: 文档中的顺序位置（从 0 开始）
            - start_offset: 原始文档中的字符偏移量（可选）
            - end_offset: 原始文档中的字符偏移量（可选）
            - source_ref: 父文档 ID 的引用（可选）
            - images: 落在此块范围内的 Document.images 子集（可选）
            - 从 Document 传播的任何文档级别元数据
        start_offset: 原始文档中的起始字符位置（可选）
        end_offset: 原始文档中的结束字符位置（可选）
        source_ref: 父 Document.id 的引用（可选）

    注意：如果块包含图片占位符，metadata.images 应只包含
          与此块文本范围相关的图片引用。

    示例：
        >>> chunk = Chunk(
        ...     id="chunk_abc123_001",
        ...     text="## Section 1\\n\\nFirst paragraph...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "chunk_index": 0,
        ...         "page": 1
        ...     },
        ...     start_offset=0,
        ...     end_offset=150
        ... )
    """
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    source_ref: Optional[str] = None
    
    def __post_init__(self):
        """验证必需的元数据字段。"""
        if "source_path" not in self.metadata:
            raise ValueError("Chunk metadata must contain 'source_path'")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以进行序列化。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """从字典创建 Chunk。"""
        return cls(**data)


@dataclass
class ChunkRecord:
    """表示完全处理完毕、可存储和检索的块。

    这是 embedding 管道的输出，也是存储在向量数据库中的数据结构。
    它通过向量表示扩展了 Chunk。

    属性：
        id: 唯一的块标识符（幂等性 upsert 需要保持稳定）
        text: 块内容（与 Chunk.text 相同）。
              图片表示为占位符：[IMAGE: {image_id}]
        metadata: 扩展的元数据，包括：
            - source_path（必填）: 原始文件路径
            - chunk_index: 顺序位置
            - 来自 Chunk 的所有元数据
            - images: 来自 Chunk 的图片引用（见 Document.images 规范）
            - Transform 管道的任何增强（标题、摘要、标签）
            - image_captions: 如果应用了多模态增强，则为 Dict[image_id, caption_text]
        dense_vector: 稠密 embedding 向量（例如来自 OpenAI、BGE）
        sparse_vector: 用于 BM25/关键词匹配的稀疏向量（可选）

    注意：ImageCaptioner 生成的图片说明存储在 metadata.image_captions 中，
          为将 image_id 映射到生成的说明文字的字典。

    示例：
        >>> record = ChunkRecord(
        ...     id="chunk_abc123_001",
        ...     text="## Section 1\\n\\nFirst paragraph...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "chunk_index": 0,
        ...         "title": "Introduction",
        ...         "summary": "Overview of project goals"
        ...     },
        ...     dense_vector=[0.1, 0.2, ..., 0.3],
        ...     sparse_vector={"word1": 0.5, "word2": 0.3}
        ... )
    """
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """验证必需的元数据字段。"""
        if "source_path" not in self.metadata:
            raise ValueError("ChunkRecord metadata must contain 'source_path'")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以进行序列化。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkRecord":
        """从字典创建 ChunkRecord。"""
        return cls(**data)

    @classmethod
    def from_chunk(cls, chunk: Chunk, dense_vector: Optional[List[float]] = None,
                   sparse_vector: Optional[Dict[str, float]] = None) -> "ChunkRecord":
        """使用向量从 Chunk 创建 ChunkRecord。

        参数：
            chunk: 源 Chunk 对象
            dense_vector: 稠密 embedding 向量
            sparse_vector: 稀疏向量表示

        返回：
            所有字段都从 chunk 填充的 ChunkRecord
        """
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
            dense_vector=dense_vector,
            sparse_vector=sparse_vector
        )


# 方便使用的类型别名
Metadata = Dict[str, Any]
Vector = List[float]
SparseVector = Dict[str, float]


@dataclass
class ProcessedQuery:
    """表示已处理完毕、可供检索的查询。

    这是 QueryProcessor 的输出，包含下游稠密/稀疏检索器所需的提取关键词和解析后的过滤器。

    属性：
        original_query: 原始用户查询字符串
        keywords: 停用词移除后提取的关键词列表
        filters: 过滤条件字典（例如 {"collection": "api-docs"}）
        expanded_terms: 可选的同义词/扩展词列表（供将来使用）

    示例：
        >>> pq = ProcessedQuery(
        ...     original_query="如何配置 Azure OpenAI？",
        ...     keywords=["配置", "Azure", "OpenAI"],
        ...     filters={"collection": "docs"}
        ... )
    """
    
    original_query: str
    keywords: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    expanded_terms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以进行序列化。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedQuery":
        """从字典创建 ProcessedQuery。"""
        return cls(**data)


@dataclass
class RetrievalResult:
    """表示来自稠密/稀疏检索器的单个检索结果。

    这是 DenseRetriever、SparseRetriever 和 HybridSearch 的输出，
    为所有搜索方法的检索结果提供统一的契约。

    属性：
        chunk_id: 检索到的块的唯一标识符
        score: 相关性分数（越高 = 越相关，归一化到 [0, 1]）
        text: 检索到的块的实际文本内容
        metadata: 关联的元数据（source_path、chunk_index、title 等）

    示例：
        >>> result = RetrievalResult(
        ...     chunk_id="doc1_chunk_003",
        ...     score=0.85,
        ...     text="Azure OpenAI 配置步骤如下...",
        ...     metadata={
        ...         "source_path": "docs/azure-guide.pdf",
        ...         "chunk_index": 3,
        ...         "title": "Azure Configuration"
        ...     }
        ... )
    """
    
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后验证字段。"""
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if not isinstance(self.score, (int, float)):
            raise ValueError(f"score must be numeric, got {type(self.score).__name__}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以进行序列化。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        """从字典创建 RetrievalResult。"""
        return cls(**data)
