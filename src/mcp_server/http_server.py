"""使用 HTTP SSE 传输的 MCP 服务器。

本模块为 MCP 服务器提供 HTTP SSE（服务器发送事件）传输层，
允许客户端通过 HTTP POST 请求连接。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List

from mcp.server.lowlevel import Server

from src.mcp_server.protocol_handler import create_mcp_server, get_protocol_handler
from src.observability.logger import get_logger

logger = get_logger(log_level="INFO")

SERVER_NAME = "faya-rag"
SERVER_VERSION = "0.1.0"


async def handle_request(
    server: Server,
    request_data: Dict[str, Any],
) -> Dict[str, Any]:
    """处理单个 JSON-RPC 请求。

    参数:
        server: MCP 服务器实例。
        request_data: 解析后的 JSON-RPC 请求。

    返回:
        JSON-RPC 响应字典。
    """
    method = request_data.get("method")
    params = request_data.get("params", {})
    request_id = request_data.get("id")
    
    logger.debug(f"Handling method: {method}")
    
    # 处理 tools/list
    if method == "tools/list":
        # 从协议处理器获取工具列表
        protocol_handler = get_protocol_handler(server)
        tools = protocol_handler.get_tool_schemas()
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [tool.dict() if hasattr(tool, "dict") else tool for tool in tools]
            }
        }

    # 处理 tools/call
    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        protocol_handler = get_protocol_handler(server)
        result = await protocol_handler.execute_tool(tool_name, arguments)

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result.model_dump() if hasattr(result, "model_dump") else (result.dict() if hasattr(result, "dict") else result)
        }

    # 处理 initialize
    elif method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2025-06-18",
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                },
                "capabilities": {
                    "tools": {}
                }
            }
        }

    # 未知方法
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}"
            }
        }


async def run_http_server_async(host: str = "0.0.0.0", port: int = 8080) -> None:
    """运行使用 HTTP 传输的 MCP 服务器。

    参数:
        host: 绑定的主机地址。
        port: 监听的端口。
    """
    try:
        from aiohttp import web
    except ImportError:
        logger.error("未安装 aiohttp。请运行: uv pip install aiohttp")
        sys.exit(1)

    # 创建 MCP 服务器
    mcp_server = create_mcp_server(SERVER_NAME, SERVER_VERSION)

    # 创建 HTTP 服务器
    app = web.Application()

    # 健康检查端点
    async def health_check(request):
        return web.json_response({"status": "ok"})

    # 用于 JSON-RPC 请求的主端点
    async def handle_jsonrpc(request):
        try:
            request_data = await request.json()
            logger.debug(f"收到请求: {request_data}")

            response_data = await handle_request(mcp_server, request_data)

            return web.json_response(response_data)

        except json.JSONDecodeError as e:
            logger.error(f"无效的 JSON: {e}")
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "解析错误: 无效的 JSON"
                    }
                },
                status=400
            )
        except Exception as e:
            logger.exception(f"处理请求时出错: {e}")
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": request_data.get("id") if 'request_data' in locals() else None,
                    "error": {
                        "code": -32603,
                        "message": f"内部错误: {str(e)}"
                    }
                },
                status=500
            )

    # 注册路由
    app.router.add_get("/health", health_check)
    app.router.add_post("/call", handle_jsonrpc)
    app.router.add_post("/", handle_jsonrpc)

    logger.info(f"在 http://{host}:{port} 启动 MCP HTTP 服务器")
    logger.info("端点:")
    logger.info("  POST /call - JSON-RPC 调用")
    logger.info("  GET  /health - 健康检查")

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host, port)
    await site.start()

    logger.info(f"服务器运行在 http://{host}:{port}")

    # 保持运行
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


def run_http_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """运行使用 HTTP 传输的 MCP 服务器（同步包装器）。

    参数:
        host: 绑定的主机地址。
        port: 监听的端口。
    """
    try:
        asyncio.run(run_http_server_async(host, port))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)


def main() -> None:
    """HTTP MCP 服务器的入口点。"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Server with HTTP transport")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    
    args = parser.parse_args()
    
    run_http_server(args.host, args.port)


if __name__ == "__main__":
    main()
