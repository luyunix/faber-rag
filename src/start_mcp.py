#!/usr/bin/env python3
"""Faber RAG 项目启动脚本

此脚本用于启动 MCP HTTP Server (默认端口 8080)。

"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ProcessManager:
    """管理多个子进程的生命周期"""

    def __init__(self):
        self.processes: list[subprocess.Popen] = []
        self.running = True

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理停止信号"""
        logger.info(f"收到停止信号 {signum}，正在关闭服务...")
        self.running = False
        self.stop_all()
        sys.exit(0)

    def start_process(
        self,
        cmd: list[str],
        name: str,
        env: Optional[dict] = None
    ) -> subprocess.Popen:
        """启动一个子进程"""
        logger.info(f"启动 {name}...")

        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        process = subprocess.Popen(
            cmd,
            env=process_env,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        self.processes.append(process)
        logger.info(f"✅ {name} 已启动 (PID: {process.pid})")

        return process

    def stop_all(self):
        """停止所有进程"""
        logger.info("正在停止所有服务...")

        for process in reversed(self.processes):
            try:
                if process.poll() is None:  # 进程还在运行
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    logger.info(f"✅ 进程 {process.pid} 已停止")
            except Exception as e:
                logger.error(f"停止进程 {process.pid} 时出错：{e}")

        self.processes.clear()
        logger.info("所有服务已停止")


def check_port_available(port: int) -> bool:
    """检查端口是否可用"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False


def kill_process_on_port(port: int) -> None:
    """杀死占用端口的进程"""
    try:
        result = subprocess.run(
            ['lsof', '-ti', str(port)],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    logger.info(f"已终止占用端口 {port} 的进程 (PID: {pid})")
                except ProcessLookupError:
                    pass
    except Exception as e:
        logger.warning(f"清理端口 {port} 时出错：{e}")


def wait_for_service(url: str, timeout: int = 30) -> bool:
    """等待服务可用"""
    import httpx

    logger.info(f"等待服务启动：{url}")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(url, timeout=2)
            if response.status_code == 200:
                logger.info(f"✅ 服务已就绪：{url}")
                return True
        except Exception:
            pass
        time.sleep(1)

    logger.warning(f"⚠️  服务 {url} 未在 {timeout} 秒内就绪")
    return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Faber RAG 项目启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/start_mcp.py                          # 使用默认端口启动
  python scripts/start_mcp.py --mcp-port 9000         # 指定 MCP 端口
        """
    )

    parser.add_argument(
        '--mcp-port',
        type=int,
        default=8080,
        help='MCP Server 端口 (默认：8080)'
    )
    parser.add_argument(
        '--no-mcp',
        action='store_true',
        help='不启动 MCP Server'
    )
    parser.add_argument(
        '--host',
        default='localhost',
        help='绑定主机 (默认：localhost)'
    )
    parser.add_argument(
        '--clean-ports',
        action='store_true',
        help='启动前清理被占用的端口'
    )

    args = parser.parse_args()

    # 项目根目录
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    logger.info("=" * 60)
    logger.info("🚀 Faber RAG 启动脚本")
    logger.info("=" * 60)
    logger.info(f"📁 项目根目录：{project_root}")
    logger.info(f"🌐 MCP Server: http://{args.host}:{args.mcp_port}")
    logger.info("=" * 60)

    # 清理端口
    if args.clean_ports:
        if not args.no_mcp:
            kill_process_on_port(args.mcp_port)
        time.sleep(1)

    # 检查端口
    if not args.no_mcp and not check_port_available(args.mcp_port):
        logger.error(f"❌ MCP 端口 {args.mcp_port} 已被占用，请使用 --clean-ports 或指定其他端口")
        sys.exit(1)

    # 创建进程管理器
    manager = ProcessManager()

    try:
        # 启动 MCP 服务器
        if not args.no_mcp:
            mcp_cmd = [
                sys.executable,
                "-m",
                "src.mcp_server.http_server",
                "--host", args.host,
                "--port", str(args.mcp_port)
            ]
            manager.start_process(mcp_cmd, "MCP HTTP Server")

            # 等待 MCP 服务器启动
            time.sleep(2)
            wait_for_service(f"http://{args.host}:{args.mcp_port}/health")

        # 显示访问信息
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ 所有服务已启动成功！")
        logger.info("=" * 60)

        if not args.no_mcp:
            logger.info(f"🔗 MCP Server:  http://{args.host}:{args.mcp_port}")
            logger.info(f"   - 健康检查：http://{args.host}:{args.mcp_port}/health")
            logger.info(f"   - JSON-RPC:  http://{args.host}:{args.mcp_port}/call")

        logger.info("=" * 60)
        logger.info("按 Ctrl+C 停止所有服务")
        logger.info("=" * 60)

        # 保持运行
        while manager.running:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("\n收到中断信号，正在退出...")
    except Exception as e:
        logger.exception(f"启动过程中出错：{e}")
        manager.stop_all()
        sys.exit(1)
    finally:
        manager.stop_all()


if __name__ == "__main__":
    main()
