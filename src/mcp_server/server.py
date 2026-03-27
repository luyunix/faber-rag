"""使用官方 MCP SDK 的 MCP 服务器入口点。

本模块使用官方 Python MCP SDK 和 stdio 传输方式实现 MCP 服务器。
确保标准输出仅包含协议消息，所有日志都写入标准错误。
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from src.mcp_server.protocol_handler import create_mcp_server
from src.observability.logger import get_logger

if TYPE_CHECKING:
    pass


SERVER_NAME = "faber-rag"
SERVER_VERSION = "0.1.0"


def _redirect_all_loggers_to_stderr() -> None:
    """将所有根日志处理器重定向到标准错误。

    MCP stdio 传输层保留标准输出用于 JSON-RPC 消息。
    任何写入标准输出的日志都会破坏协议流。
    """
    import logging as _logging

    root = _logging.getLogger()
    stderr_handler = _logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        _logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    # Replace any existing stream handlers that might point to stdout
    for handler in root.handlers[:]:
        if isinstance(handler, _logging.StreamHandler) and not isinstance(
            handler, _logging.FileHandler
        ):
            root.removeHandler(handler)
    root.addHandler(stderr_handler)


def _preload_heavy_imports() -> None:
    """在**主线程**中预先导入重量级的第三方模块。

    MCP SDK 使用 anyio + 后台线程处理 stdin/stdout I/O。
    当工具处理器运行 ``asyncio.to_thread(fn)`` 时，*fn* 在新工作线程中执行。
    如果线程尝试 ``import chromadb``（它会间接引入 onnxruntime, numpy, sqlite3 C 扩展等），
    该导入可能与 stdin 读取线程发生死锁，因为两者会竞争 Python 的全局*导入锁*。

    在这里预导入——在 anyio 启动其 I/O 线程之前——可以完全避免死锁：
    后续在工作线程中的 ``import`` 语句只需访问 ``sys.modules`` 即可立即返回。
    """
    # chromadb 是最耗资源的模块（onnxruntime, numpy 等）
    try:
        import chromadb  # noqa: F401
        import chromadb.config  # noqa: F401
    except ImportError:
        pass  # 安装时可选

    # 工具在 asyncio.to_thread 中延迟导入的内部模块
    try:
        import src.core.query_engine.query_processor  # noqa: F401
        import src.core.query_engine.hybrid_search  # noqa: F401
        import src.core.query_engine.dense_retriever  # noqa: F401
        import src.core.query_engine.sparse_retriever  # noqa: F401
        import src.core.query_engine.reranker  # noqa: F401
        import src.ingestion.storage.bm25_indexer  # noqa: F401
        import src.libs.embedding.embedding_factory  # noqa: F401
        import src.libs.vector_store.vector_store_factory  # noqa: F401
    except ImportError:
        pass


async def run_stdio_server_async() -> int:
    """通过 stdio 异步运行 MCP 服务器。

    返回:
        退出码。
    """
    # 在这里导入以避免 mcp 未安装时的导入错误    import mcp.server.stdio

    # 确保所有日志都写入 stderr（stdout 保留给 JSON-RPC）    _redirect_all_loggers_to_stderr()

    # 在主线程中预加载重量级依赖，以在工具处理器稍后调用 asyncio.to_thread() 时防止导入锁死锁
    _preload_heavy_imports()

    logger = get_logger(log_level="INFO")
    logger.info("Starting MCP server (stdio transport) with official SDK.")

    # 创建带有协议处理器的服务器
    server = create_mcp_server(SERVER_NAME, SERVER_VERSION)

    # 使用 stdio 传输运行
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

    logger.info("MCP server shutting down.")
    return 0


def run_stdio_server() -> int:
    """通过 stdio 运行 MCP 服务器（同步包装器）。

    返回:
        退出码。
    """
    return asyncio.run(run_stdio_server_async())


def main() -> int:
    """stdio MCP 服务器的入口点。"""
    return run_stdio_server()


if __name__ == "__main__":
    sys.exit(main())