"""MCP 工具: list_collections

该工具通过 MCP 协议提供集合列表能力。
它列出向量存储中所有可用的集合及其统计信息。

使用方法:
    工具名: list_collections
    输入模式:
        - include_stats (boolean, optional): 包含每个集合的统计信息
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp import types

if TYPE_CHECKING:
    from src.mcp_server.protocol_handler import ProtocolHandler
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


# 工具元数据
TOOL_NAME = "list_collections"
TOOL_DESCRIPTION = """列出知识库中所有可用的文档集合。

返回有关每个集合的信息，包括:
- 集合名称
- 文档数量（如果 include_stats=true）
- 集合元数据

在查询之前使用此工具发现有用的集合。
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "include_stats": {
            "type": "boolean",
            "description": "是否包含每个集合的统计信息（文档计数）。",
            "default": True,
        },
    },
    "required": [],
}


@dataclass
class CollectionInfo:
    """单个集合的信息。

    属性:
        name: 集合名称
        count: 集合中的文档/块数量（可选）
        metadata: 集合元数据字典
    """
    name: str
    count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示。"""
        result: Dict[str, Any] = {"name": self.name}
        if self.count is not None:
            result["count"] = self.count
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class ListCollectionsConfig:
    """list_collections 工具的配置。

    属性:
        persist_directory: ChromaDB 存储目录的路径
        include_stats_default: include_stats 参数的默认值
    """
    persist_directory: str = "./data/db/chroma"
    include_stats_default: bool = True


class ListCollectionsTool:
    """用于列出知识库集合的 MCP 工具。

    此类封装了 list_collections 工具的逻辑，
    查询向量存储以枚举可用的集合。

    设计原则:
    - 配置驱动: 路径来自 settings.yaml
    - 错误恢复能力: 优雅处理缺失的目录
    - 可观测性: 用于调试的日志记录

    示例:
        >>> tool = ListCollectionsTool(settings)
        >>> result = await tool.execute(include_stats=True)
        >>> print(result)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[ListCollectionsConfig] = None,
    ) -> None:
        """初始化 ListCollectionsTool。

        参数:
            settings: 应用程序设置。如果为 None，从默认路径加载。
            config: 工具配置。如果为 None，从设置派生。
        """
        self._settings = settings
        self._config = config
        
    @property
    def settings(self) -> Settings:
        """Get settings, loading if necessary."""
        if self._settings is None:
            from src.core.settings import load_settings
            self._settings = load_settings()
        return self._settings
    
    @property
    def config(self) -> ListCollectionsConfig:
        """Get configuration, deriving from settings if necessary."""
        if self._config is None:
            try:
                persist_dir = getattr(
                    self.settings.vector_store,
                    'persist_directory',
                    './data/db/chroma'
                )
            except AttributeError:
                persist_dir = './data/db/chroma'
            
            self._config = ListCollectionsConfig(
                persist_directory=persist_dir
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
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "chromadb package is required for list_collections. "
                "Install it with: pip install chromadb"
            )
        
        persist_path = Path(self.config.persist_directory).resolve()
        
        if not persist_path.exists():
            logger.warning(f"ChromaDB 目录不存在: {persist_path}")
            # 仍然返回客户端 - 它只会没有集合
            persist_path.mkdir(parents=True, exist_ok=True)
        
        try:
            client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            return client
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB client at '{persist_path}': {e}"
            ) from e
    
    def list_collections(
        self,
        include_stats: bool = True
    ) -> List[CollectionInfo]:
        """列出所有可用的集合。

        参数:
            include_stats: 是否包含文档计数。

        返回:
            CollectionInfo 对象列表。
        """
        try:
            client = self._get_chroma_client()
        except (ImportError, RuntimeError) as e:
            logger.error(f"Failed to get ChromaDB client: {e}")
            return []
        
        collections_info: List[CollectionInfo] = []
        
        try:
            # Get all collections from ChromaDB
            collections = client.list_collections()
            
            for collection in collections:
                if isinstance(collection, str):
                    collection_name = collection
                    collection_obj = client.get_collection(collection_name)
                else:
                    collection_name = collection.name
                    collection_obj = collection

                info = CollectionInfo(
                    name=collection_name,
                    metadata=collection_obj.metadata
                )
                
                if include_stats:
                    try:
                        info.count = collection_obj.count()
                    except Exception as e:
                        logger.warning(
                            f"Failed to get count for collection '{collection_name}': {e}"
                        )
                        info.count = None
                
                collections_info.append(info)
                
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
        
        logger.info(f"Found {len(collections_info)} collections")
        return collections_info
    
    def format_response(
        self,
        collections: List[CollectionInfo]
    ) -> str:
        """将集合列表格式化为可读的字符串。

        参数:
            collections: CollectionInfo 对象列表。

        返回:
            适合 MCP 响应的格式化字符串。
        """
        if not collections:
            return "No collections found in the knowledge base."
        
        lines = [
            f"## Available Collections ({len(collections)} total)\n"
        ]
        
        for i, coll in enumerate(collections, 1):
            line = f"{i}. **{coll.name}**"
            
            if coll.count is not None:
                line += f" - {coll.count} documents"
            
            if coll.metadata:
                # 过滤器 out internal metadata
                user_metadata = {
                    k: v for k, v in coll.metadata.items()
                    if not k.startswith('_') and not k.startswith('hnsw:')
                }
                if user_metadata:
                    meta_str = ", ".join(f"{k}={v}" for k, v in user_metadata.items())
                    line += f" ({meta_str})"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    async def execute(
        self,
        include_stats: bool = True,
    ) -> types.CallToolResult:
        """执行 list_collections 工具。

        参数:
            include_stats: 是否为每个集合包含统计信息。

        返回:
            包含格式化集合列表的 CallToolResult。
        """
        logger.info(f"Executing list_collections (include_stats={include_stats})")
        
        try:
            # 在线程中运行阻塞式 ChromaDB I/O 以避免阻塞
            # 异步事件循环 / MCP stdio 传输
            collections = await asyncio.to_thread(
                self.list_collections, include_stats,
            )
            response_text = self.format_response(collections)
            
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=response_text,
                    )
                ],
                isError=False,
            )
            
        except Exception as e:
            logger.exception("Error executing list_collections")
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Error listing collections: {str(e)}",
                    )
                ],
                isError=True,
            )


def register_tool(protocol_handler: ProtocolHandler) -> None:
    """Register the list_collections tool with the protocol handler.
    
    This function is called by _register_default_tools() in protocol_handler.py
    to register this tool when the MCP server starts.
    
    Args:
        protocol_handler: ProtocolHandler instance to register with.
    """
    tool = ListCollectionsTool()
    
    async def handler(
        include_stats: bool = True,
    ) -> types.CallToolResult:
        """Handler function for MCP tool calls."""
        return await tool.execute(include_stats=include_stats)
    
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=handler,
    )
    
    logger.info(f"Registered MCP tool: {TOOL_NAME}")
