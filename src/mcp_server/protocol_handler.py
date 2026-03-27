"""用于处理 JSON-RPC 2.0 消息的 MCP 协议处理器。

本模块提供 ProtocolHandler 类，封装以下内容：
- 工具注册和模式管理
- JSON-RPC 错误代码处理
- 初始化时的能力协商
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from mcp import types
from mcp.server.lowlevel import Server

from src.observability.logger import get_logger


# JSON-RPC 2.0 错误代码
class JSONRPCErrorCodes:
    """标准 JSON-RPC 2.0 错误代码。"""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


@dataclass
class ToolDefinition:
    """MCP 工具的定义。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any]


@dataclass
class ProtocolHandler:
    """处理 MCP 协议操作，包括工具注册和执行。

    该类封装以下内容：
    - 工具注册和模式验证
    - 工具执行和错误处理
    - 初始化响应的能力声明

    属性:
        server_name: MCP 服务器的名称。
        server_version: 服务器的版本字符串。
        tools: 可用工具的注册表。
    """

    server_name: str
    server_version: str
    tools: Dict[str, ToolDefinition] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """数据类初始化后初始化日志记录器。"""
        self._logger = get_logger(log_level="INFO")

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """向协议处理器注册工具。

        参数:
            name: 工具的唯一名称。
            description: 描述工具功能的人类可读说明。
            input_schema: 工具输入参数的 JSON 模式。
            handler: 执行工具逻辑的异步函数。

        抛出异常:
            ValueError: 如果已存在同名工具。
        """
        if name in self.tools:
            raise ValueError(f"Tool '{name}' is already registered")

        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
        )
        self._logger.info("Registered tool: %s", name)

    def get_tool_schemas(self) -> List[types.Tool]:
        """获取用于 tools/list 响应的工具模式列表。

        返回:
            包含名称、描述和 inputSchema 的 Tool 对象列表。
        """
        return [
            types.Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.input_schema,
            )
            for tool in self.tools.values()
        ]

    async def execute_tool(
        self, name: str, arguments: Dict[str, Any]
    ) -> types.CallToolResult:
        """通过名称执行已注册的工具。

        参数:
            name: 要执行的工具名称。
            arguments: 传递给工具处理器的参数。

        返回:
            包含内容块或错误指示的 CallToolResult。

        抛出异常:
            ValueError: 如果未找到工具。
        """
        if name not in self.tools:
            self._logger.warning("Tool not found: %s", name)
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Error: Tool '{name}' not found",
                    )
                ],
                isError=True,
            )

        tool = self.tools[name]
        try:
            self._logger.info("Executing tool: %s", name)
            result = await tool.handler(**arguments)

            # 处理不同的返回类型
            if isinstance(result, types.CallToolResult):
                return result
            if isinstance(result, str):
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=result)],
                    isError=False,
                )
            if isinstance(result, list):
                return types.CallToolResult(content=result, isError=False)
            # 默认：转换为字符串
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(result))],
                isError=False,
            )

        except TypeError as e:
            # 无效的参数
            self._logger.error("工具 %s 的参数无效: %s", name, e)
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"错误: 无效的参数 - {e}",
                    )
                ],
                isError=True,
            )
        except Exception as e:
            # 内部错误 - 不泄露堆栈跟踪
            self._logger.exception("执行工具 %s 时发生内部错误", name)
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Error: Internal server error while executing '{name}'",
                    )
                ],
                isError=True,
            )

    def get_capabilities(self) -> Dict[str, Any]:
        """获取初始化响应的服务器能力。

        返回:
            服务器能力字典。
        """
        return {
            "tools": {} if self.tools else {},
        }


def _register_default_tools(protocol_handler: ProtocolHandler) -> None:
    """向协议处理器注册所有默认 MCP 工具。

    参数:
        protocol_handler: 要注册工具的 ProtocolHandler 实例。
    """
    # 导入并注册 query_knowledge_hub 工具
    from src.mcp_server.tools.query_knowledge_hub import register_tool as register_query_tool
    register_query_tool(protocol_handler)
    
    # 导入并注册 list_collections 工具
    from src.mcp_server.tools.list_collections import register_tool as register_list_tool
    register_list_tool(protocol_handler)
    
    # 导入并注册 get_document_summary 工具
    from src.mcp_server.tools.get_document_summary import register_tool as register_summary_tool
    register_summary_tool(protocol_handler)


def create_mcp_server(
    server_name: str,
    server_version: str,
    protocol_handler: Optional[ProtocolHandler] = None,
    register_tools: bool = True,
) -> Server:
    """创建并配置带有协议处理器的 MCP 服务器。

    此工厂函数创建一个底层的 MCP Server 实例，并
    注册 tools/list 和 tools/call 所需的处理器。

    参数:
        server_name: 服务器名称。
        server_version: 版本字符串。
        protocol_handler: 可选的预配置协议处理器。
            如果为 None，将创建一个新的。
        register_tools: 是否注册默认工具（默认：True）。

    返回:
        配置好的 Server 实例，准备就绪。
    """
    if protocol_handler is None:
        protocol_handler = ProtocolHandler(
            server_name=server_name,
            server_version=server_version,
        )

    # 注册默认工具（如果请求）
    if register_tools:
        _register_default_tools(protocol_handler)

    # 创建底层 server
    server = Server(server_name)

    # 注册 tools/list 处理器
    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        """处理 tools/list 请求。"""
        return protocol_handler.get_tool_schemas()

    # 注册 tools/call 处理器
    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: Dict[str, Any]
    ) -> types.CallToolResult:
        """处理 tools/call 请求。"""
        return await protocol_handler.execute_tool(name, arguments)

    # 存储 protocol handler 到 server 以便访问
    server._protocol_handler = protocol_handler  # type: ignore[attr-defined]

    return server


def get_protocol_handler(server: Server) -> ProtocolHandler:
    """从服务器实例获取协议处理器。

    参数:
        server: 由 create_mcp_server 创建的 Server 实例。

    返回:
        与服务器关联的 ProtocolHandler。

    抛出异常:
        AttributeError: 如果服务器不是通过 create_mcp_server 创建的。
    """
    return server._protocol_handler  # type: ignore[attr-defined]
