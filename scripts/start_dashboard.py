"""用于启动模块化 RAG 仪表盘的便捷脚本。

用法::

    python scripts/start_dashboard.py
    python scripts/start_dashboard.py --port 8502
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="启动模块化 RAG 仪表盘")
    parser.add_argument("--port", type=int, default=8501, help="仪表盘服务的端口号")
    parser.add_argument("--host", type=str, default="localhost", help="绑定的主机地址")
    args = parser.parse_args()

    app_path = Path(__file__).resolve().parent.parent / "src" / "observability" / "dashboard" / "app.py"
    if not app_path.exists():
        print(f"错误：未找到仪表盘应用，路径：{app_path}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
    ]
    print(f"正在启动仪表盘：{' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
