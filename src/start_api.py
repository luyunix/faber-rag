"""启动 Faber RAG API 服务器。

为前端 UI 提供 REST API 接口。
"""

import logging
import sys

from src.api.server import main as start_api_server
from src.observability.logger import get_logger

logger = get_logger(log_level="INFO")


def main():
    """启动 API 服务器。"""
    logger.info("Starting Faber RAG API Server...")
    try:
        start_api_server()
    except KeyboardInterrupt:
        logger.info("API server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"API server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
