"""通过 HTTP SSE 传输连接到 MCP 服务器的 MCP 客户端。

本模块提供一个简单的客户端，用于通过 HTTP SSE（服务器发送事件）
传输协议连接到 MCP 服务器。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class MCPToolResult:
    """调用 MCP 工具的结果。

    属性:
        content: 工具响应中的内容块列表。
        is_error: 工具调用是否导致错误。
        raw_response: 原始 JSON-RPC 响应字典。
    """
    content: List[Dict[str, Any]]
    is_error: bool
    raw_response: Dict[str, Any]
    
    def get_text(self) -> str:
        """从结果中提取文本内容。

        返回:
            所有文本内容块的拼接文本，
            如果没有文本内容则返回空字符串。
        """
        texts = []
        for block in self.content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif hasattr(block, "type") and block.type == "text":
                texts.append(block.text)
        return "\n".join(texts)


class MCPClient:
    """使用 HTTP SSE 传输的 MCP 客户端。

    此客户端通过 HTTP 服务器发送事件连接到 MCP 服务器，
    提供调用远程工具的简单接口。

    示例:
        >>> client = MCPClient("http://localhost:8080")
        >>> result = await client.call_tool(
        ...     tool_name="query_knowledge_hub",
        ...     arguments={"query": "Azure 配置", "top_k": 5}
        ... )
        >>> print(result.get_text())
    """
    
    def __init__(
        self,
        server_url: str,
        timeout: float = 30.0,
    ) -> None:
        """初始化 MCP 客户端。

        参数:
            server_url: MCP 服务器端点的 URL。
            timeout: 请求超时时间（秒）。
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._request_id = 0
    
    async def __aenter__(self) -> MCPClient:
        """异步上下文管理器入口。"""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器退出。"""
        if self._client:
            await self._client.aclose()
    
    def _next_request_id(self) -> int:
        """生成下一个请求 ID。"""
        self._request_id += 1
        return self._request_id
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> MCPToolResult:
        """调用 MCP 工具。

        参数:
            tool_name: 要调用的工具名称。
            arguments: 工具参数字典。

        返回:
            MCPToolResult，包含工具的响应。

        抛出异常:
            MCPClientError: 如果工具调用失败。
        """
        if not self._client:
            raise MCPClientError("Client not initialized. Use async context manager.")
        
        request_id = self._next_request_id()
        
        # 构建 JSON-RPC request
        request_body = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            }
        }
        
        logger.debug(f"Calling tool {tool_name} with args: {arguments}")
        
        try:
            # Send POST request to server
            logger.debug(f"Sending request to {self.server_url}/call")
            response = await self._client.post(
                f"{self.server_url}/call",
                json=request_body,
            )
            logger.debug(f"Received response: status={response.status_code}")
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            # 检查 for JSON-RPC error
            if "error" in response_data:
                error = response_data["error"]
                error_msg = error.get("message", "Unknown error")
                error_code = error.get("code", -1)
                logger.error(f"MCP error {error_code}: {error_msg}")
                return MCPToolResult(
                    content=[{"type": "text", "text": f"Error: {error_msg}"}],
                    is_error=True,
                    raw_response=response_data,
                )
            
            # Extract result
            result = response_data.get("result", {})
            content = result.get("content", [])
            is_error = result.get("isError", False)
            
            logger.debug(f"Tool {tool_name} returned {len(content)} content blocks")
            
            return MCPToolResult(
                content=content,
                is_error=is_error,
                raw_response=response_data,
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise MCPClientError(f"HTTP error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise MCPClientError(f"Request failed: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise MCPClientError(f"Invalid JSON response: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error calling tool {tool_name}")
            raise MCPClientError(f"Unexpected error: {e}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """从服务器获取可用工具列表。

        返回:
            工具定义列表。

        抛出异常:
            MCPClientError: 如果请求失败。
        """
        if not self._client:
            raise MCPClientError("Client not initialized.")
        
        request_id = self._next_request_id()
        
        request_body = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/list",
            "params": {}
        }
        
        try:
            response = await self._client.post(
                f"{self.server_url}/call",
                json=request_body,
            )
            response.raise_for_status()
            
            response_data = response.json()
            
            if "error" in response_data:
                error = response_data["error"]
                error_msg = error.get("message", "Unknown error")
                raise MCPClientError(f"Error listing tools: {error_msg}")
            
            result = response_data.get("result", {})
            tools = result.get("tools", [])
            
            logger.debug(f"Server returned {len(tools)} tools")
            
            return tools
            
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            raise MCPClientError(f"Failed to list tools: {e}")


class MCPClientError(Exception):
    """MCP 客户端引发的异常。"""
    pass
