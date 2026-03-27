"""MCP 工具: get_document_summary

该工具通过 MCP 协议提供文档摘要检索能力。
它根据 doc_id 返回特定文档的标题、摘要和标签。

使用方法:
    工具名: get_document_summary
    输入模式:
        - doc_id (string, required): 要检索摘要的文档 ID
        - collection (string, optional): 要搜索的集合名称
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp import types

if TYPE_CHECKING:
    from src.mcp_server.protocol_handler import ProtocolHandler
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


# 工具元数据
TOOL_NAME = "get_document_summary"
TOOL_DESCRIPTION = """获取特定文档的摘要和元数据。

返回有关文档的结构化信息，包括:
- 标题（从内容提取或推断）
- 摘要（第一块预览或元数据摘要）
- 标签（文档级别的标签/类别）
- 源路径
- 块计数

在 list_collections 之后使用此工具可获取有关特定文档的详细信息。
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "doc_id": {
            "type": "string",
            "description": "要检索摘要的文档 ID。可以是完整 doc_id（例如，'doc_abc123'）或哈希部分。",
        },
        "collection": {
            "type": "string",
            "description": "要搜索的集合名称。如果未指定，则搜索默认集合。",
        },
    },
    "required": ["doc_id"],
}


@dataclass
class DocumentSummary:
    """文档的摘要信息。

    属性:
        doc_id: 文档标识符
        title: 文档标题（来自元数据或推断）
        summary: 文档内容的简要摘要或预览
        tags: 与文档关联的标签/类别列表
        source_path: 原始文件路径
        chunk_count: 文档的块数量
        metadata: 附加文档元数据
    """
    doc_id: str
    title: str
    summary: str
    tags: List[str] = field(default_factory=list)
    source_path: Optional[str] = None
    chunk_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示。"""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "summary": self.summary,
            "tags": self.tags,
            "source_path": self.source_path,
            "chunk_count": self.chunk_count,
            "metadata": self.metadata,
        }


@dataclass
class GetDocumentSummaryConfig:
    """get_document_summary 工具的配置。

    属性:
        persist_directory: ChromaDB 存储目录的路径
        default_collection: 未指定时的默认集合名称
        summary_max_length: 摘要预览的最大字符数
    """
    persist_directory: str = "./data/db/chroma"
    default_collection: str = "knowledge_hub"
    summary_max_length: int = 500


class DocumentNotFoundError(Exception):
    """当未找到具有指定 ID 的文档时引发。"""
    
    def __init__(self, doc_id: str, collection: Optional[str] = None):
        self.doc_id = doc_id
        self.collection = collection
        message = f"未找到文档 '{doc_id}'"
        if collection:
            message += f" 在集合 '{collection}' 中"
        super().__init__(message)


class GetDocumentSummaryTool:
    """用于检索文档摘要的 MCP 工具。

    此类封装了 get_document_summary 工具的逻辑，
    查询向量存储以检索文档元数据和内容预览。

    设计原则:
    - 配置驱动: 路径来自 settings.yaml
    - 错误恢复能力: 对缺失的文档给出清晰的错误消息
    - 可观测性: 用于调试的日志记录
    - 延迟初始化: ChromaDB 客户端在首次使用时创建

    示例:
        >>> tool = GetDocumentSummaryTool(settings)
        >>> result = await tool.execute(doc_id="doc_abc123")
        >>> print(result)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[GetDocumentSummaryConfig] = None,
    ) -> None:
        """初始化 GetDocumentSummaryTool。

        参数:
            settings: 应用程序设置。如果为 None，从默认路径加载。
            config: 工具配置。如果为 None，从设置派生。
        """
        self._settings = settings
        self._config = config
        self._chroma_client = None
        
    @property
    def settings(self) -> Settings:
        """Get settings, loading if necessary."""
        if self._settings is None:
            from src.core.settings import load_settings
            self._settings = load_settings()
        return self._settings
    
    @property
    def config(self) -> GetDocumentSummaryConfig:
        """Get configuration, deriving from settings if necessary."""
        if self._config is None:
            try:
                persist_dir = getattr(
                    self.settings.vector_store,
                    'persist_directory',
                    './data/db/chroma'
                )
                default_collection = getattr(
                    self.settings.vector_store,
                    'collection_name',
                    'knowledge_hub'
                )
            except AttributeError:
                persist_dir = './data/db/chroma'
                default_collection = 'knowledge_hub'
            
            self._config = GetDocumentSummaryConfig(
                persist_directory=persist_dir,
                default_collection=default_collection,
            )
        return self._config
    
    def _get_chroma_client(self) -> Any:
        """获取或创建 ChromaDB 客户端。

        返回:
            ChromaDB PersistentClient 实例。

        抛出异常:
            ImportError: 如果未安装 chromadb。
            RuntimeError: 如果客户端创建失败。
        """
        if self._chroma_client is not None:
            return self._chroma_client
        
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "chromadb package is required for get_document_summary. "
                "Install it with: pip install chromadb"
            )
        
        persist_path = Path(self.config.persist_directory).resolve()
        
        if not persist_path.exists():
            logger.warning(f"ChromaDB directory does not exist: {persist_path}")
            persist_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self._chroma_client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            return self._chroma_client
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB client at '{persist_path}': {e}"
            ) from e
    
    def _get_collection(self, collection_name: Optional[str] = None) -> Any:
        """获取 ChromaDB 集合。

        参数:
            collection_name: 集合名称。如果未指定，使用默认值。

        返回:
            ChromaDB 集合实例。

        抛出异常:
            ValueError: 如果集合不存在。
        """
        client = self._get_chroma_client()
        name = collection_name or self.config.default_collection
        
        try:
            # 尝试 to get existing collection            collection = client.get_collection(name=name)
            return collection
        except Exception as e:
            raise ValueError(
                f"Collection '{name}' does not exist: {e}"
            ) from e
    
    def _find_document_chunks(
        self,
        doc_id: str,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """查找属于文档的所有块。

        搜索 source_ref 匹配 doc_id 的块。
        如果 source_ref 不可用，则回退到块 ID 的部分匹配。

        参数:
            doc_id: 要搜索的文档 ID。
            collection_name: 要搜索的集合。

        返回:
            带有元数据的块数据列表。
        """
        collection = self._get_collection(collection_name)
        
        # 策略 1: 搜索 by source_ref metadata
        # Chunks should have source_ref pointing to parent document
        try:
            results = collection.get(
                where={"source_ref": doc_id},
                include=["metadatas", "documents"]
            )
            
            if results and results.get('ids'):
                chunks = []
                for i, chunk_id in enumerate(results['ids']):
                    chunks.append({
                        'id': chunk_id,
                        'text': results['documents'][i] if results.get('documents') else '',
                        'metadata': results['metadatas'][i] if results.get('metadatas') else {}
                    })
                if chunks:
                    return chunks
        except Exception as e:
            logger.debug(f"source_ref search failed: {e}")
        
        # 策略 2: 搜索 by doc_id in chunk ID prefix
        # 分块 ID follow format: {doc_id}_{index:04d}_{hash}
        try:
            # Get all chunks and filter by ID prefix
            all_results = collection.get(include=["metadatas", "documents"])
            
            if all_results and all_results.get('ids'):
                chunks = []
                for i, chunk_id in enumerate(all_results['ids']):
                    # 检查 if chunk_id starts with doc_id
                    if chunk_id.startswith(doc_id) or doc_id in chunk_id:
                        chunks.append({
                            'id': chunk_id,
                            'text': all_results['documents'][i] if all_results.get('documents') else '',
                            'metadata': all_results['metadatas'][i] if all_results.get('metadatas') else {}
                        })
                if chunks:
                    return chunks
        except Exception as e:
            logger.debug(f"ID prefix search failed: {e}")
        
        # No chunks found        return []
    
    def get_document_summary(
        self,
        doc_id: str,
        collection: Optional[str] = None,
    ) -> DocumentSummary:
        """获取特定文档的摘要。

        参数:
            doc_id: 要检索的文档 ID。
            collection: 要搜索的集合名称。

        返回:
            包含标题、摘要、标签等的 DocumentSummary。

        抛出异常:
            DocumentNotFoundError: 如果未找到文档。
        """
        chunks = self._find_document_chunks(doc_id, collection)
        
        if not chunks:
            raise DocumentNotFoundError(doc_id, collection)
        
        # Sort chunks by chunk_index if available        chunks.sort(key=lambda c: c.get('metadata', {}).get('chunk_index', 0))
        
        # Extract document-level info from first chunk's metadata        first_chunk = chunks[0]
        metadata = first_chunk.get('metadata', {})
        
        # Extract title        title = self._extract_title(metadata, first_chunk.get('text', ''))
        
        # Extract or generate summary        summary = self._extract_summary(chunks)
        
        # Extract tags        tags = self._extract_tags(metadata)
        
        # Extract source path        source_path = metadata.get('source_path', metadata.get('source', None))
        
        # Collect additional metadata (excluding internal fields)        additional_metadata = self._filter_metadata(metadata)
        
        return DocumentSummary(
            doc_id=doc_id,
            title=title,
            summary=summary,
            tags=tags,
            source_path=source_path,
            chunk_count=len(chunks),
            metadata=additional_metadata,
        )
    
    def _extract_title(self, metadata: Dict[str, Any], first_text: str) -> str:
        """从元数据或内容提取文档标题。

        优先级:
        1. metadata['title']
        2. 内容中的第一个标题
        3. source_path 中的文件名
        4. "无标题文档"

        参数:
            metadata: 块的元数据。
            first_text: 第一块的文本内容。

        返回:
            提取或推断的标题。
        """
        # Priority 1: Explicit title in metadata
        if metadata.get('title'):
            return str(metadata['title'])
        
        # Priority 2: First markdown heading
        if first_text:
            lines = first_text.split('\n')
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if line.startswith('# '):
                    return line[2:].strip()
        
        # Priority 3: Filename from source_path
        source_path = metadata.get('source_path', metadata.get('source'))
        if source_path:
            filename = Path(source_path).stem
            # Convert snake_case/kebab-case to Title 情况
            title = filename.replace('_', ' ').replace('-', ' ').title()
            return title
        
        # Priority 4: 默认
        return "Untitled Document"
    
    def _extract_summary(self, chunks: List[Dict[str, Any]]) -> str:
        """提取或生成文档摘要。

        优先级:
        1. 任何块中的 metadata['summary']
        2. 第一个块文本的前 N 个字符

        参数:
            chunks: 文档块列表。

        返回:
            摘要文本。
        """
        # Priority 1: Explicit summary in metadata
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            if metadata.get('summary'):
                return str(metadata['summary'])
        
        # Priority 2: Preview from first chunk
        first_text = chunks[0].get('text', '') if chunks else ''
        if first_text:
            # Clean up and truncate
            summary = first_text.strip()
            
            # Skip markdown headers for preview
            lines = summary.split('\n')
            content_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    content_lines.append(line)
                    if len(' '.join(content_lines)) > self.config.summary_max_length:
                        break
            
            summary = ' '.join(content_lines)
            
            if len(summary) > self.config.summary_max_length:
                summary = summary[:self.config.summary_max_length - 3] + "..."
            
            return summary if summary else "No content preview available."
        
        return "No summary available."
    
    def _extract_tags(self, metadata: Dict[str, Any]) -> List[str]:
        """从元数据提取标签。

        参数:
            metadata: 块的元数据。

        返回:
            标签列表。
        """
        tags = []
        
        # 检查 for explicit tags field
        if 'tags' in metadata:
            tag_value = metadata['tags']
            if isinstance(tag_value, list):
                tags.extend(str(t) for t in tag_value)
            elif isinstance(tag_value, str):
                # Could be comma-separated
                tags.extend(t.strip() for t in tag_value.split(',') if t.strip())
        
        # Add doc_type as a tag if available
        if metadata.get('doc_type'):
            doc_type = str(metadata['doc_type']).upper()
            if doc_type not in tags:
                tags.append(doc_type)
        
        return tags
    
    def _filter_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """过滤元数据以排除内部字段。

        参数:
            metadata: 原始元数据字典。

        返回:
            仅包含用户相关字段的过滤后的元数据。
        """
        # 从附加元数据中排除的字段
        exclude_fields = {
            'source_ref', 'chunk_index', 'start_offset', 'end_offset',
            '_placeholder', 'text', 'title', 'summary', 'tags',
            'source_path', 'source'
        }
        
        return {
            k: v for k, v in metadata.items()
            if k not in exclude_fields and not k.startswith('_')
        }
    
    def format_response(self, summary: DocumentSummary) -> str:
        """将文档摘要格式化为可读的字符串。

        参数:
            summary: DocumentSummary 对象。

        返回:
            适合 MCP 响应的格式化字符串。
        """
        lines = [
            f"## Document: {summary.title}",
            "",
            f"**Document ID:** `{summary.doc_id}`",
        ]
        
        if summary.source_path:
            lines.append(f"**Source:** {summary.source_path}")
        
        lines.append(f"**Chunks:** {summary.chunk_count}")
        
        if summary.tags:
            tags_str = ", ".join(f"`{tag}`" for tag in summary.tags)
            lines.append(f"**Tags:** {tags_str}")
        
        lines.extend([
            "",
            "### Summary",
            "",
            summary.summary,
        ])
        
        if summary.metadata:
            lines.extend([
                "",
                "### Additional Metadata",
                "",
            ])
            for key, value in summary.metadata.items():
                lines.append(f"- **{key}:** {value}")
        
        return "\n".join(lines)
    
    def format_error(self, error: Exception) -> str:
        """将错误格式化为可读的字符串。

        参数:
            error: 发生的异常。

        返回:
            格式化的错误消息。
        """
        if isinstance(error, DocumentNotFoundError):
            return f"## 文档未找到\n\n{str(error)}\n\n请验证文档 ID 和集合名称。"
        elif isinstance(error, ValueError):
            return f"## 无效请求\n\n{str(error)}"
        else:
            return f"## 错误\n\n发生错误: {str(error)}"
    
    async def execute(
        self,
        doc_id: str,
        collection: Optional[str] = None,
    ) -> types.CallToolResult:
        """执行 get_document_summary 工具。

        参数:
            doc_id: 要检索摘要的文档 ID。
            collection: 可选的集合名称。

        返回:
            包含格式化文档摘要或错误的 CallToolResult。
        """
        logger.info(f"Executing get_document_summary (doc_id={doc_id}, collection={collection})")
        
        try:
            # 在线程中运行阻塞式 ChromaDB I/O 以避免阻塞
            # 异步事件循环 / MCP stdio 传输
            summary = await asyncio.to_thread(
                self.get_document_summary, doc_id, collection,
            )
            response_text = self.format_response(summary)
            
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=response_text,
                    )
                ],
                isError=False,
            )
            
        except DocumentNotFoundError as e:
            logger.warning(f"Document not found: {e}")
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=self.format_error(e),
                    )
                ],
                isError=True,
            )
            
        except Exception as e:
            logger.exception("Error executing get_document_summary")
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=self.format_error(e),
                    )
                ],
                isError=True,
            )


def register_tool(protocol_handler: ProtocolHandler) -> None:
    """Register the get_document_summary tool with the protocol handler.
    
    This function is called by _register_default_tools() in protocol_handler.py
    to register this tool when the MCP server starts.
    
    Args:
        protocol_handler: ProtocolHandler instance to register with.
    """
    tool = GetDocumentSummaryTool()
    
    async def handler(
        doc_id: str,
        collection: Optional[str] = None,
    ) -> types.CallToolResult:
        """Handler function for MCP tool calls."""
        return await tool.execute(doc_id=doc_id, collection=collection)
    
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=handler,
    )
    
    logger.info(f"Registered MCP tool: {TOOL_NAME}")
