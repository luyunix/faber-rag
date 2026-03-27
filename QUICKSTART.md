# Faber RAG 快速启动指南

## 🚀 一键启动

使用统一的启动脚本：

```bash
# 使用默认端口启动所有服务
python scripts/start.py

# 或使用快捷命令（需要先安装）
uv pip install -e .
faber-start
```

## 📋 启动选项

```bash
# 查看帮助
python scripts/start.py --help

# 使用默认端口启动
python scripts/start.py

# 指定端口
python scripts/start.py --mcp-port 9000 --dashboard-port 8502

# 只启动 MCP Server
python scripts/start.py --no-dashboard

# 只启动 Dashboard
python scripts/start.py --no-mcp

# 启动前清理被占用的端口
python scripts/start.py --clean-ports
```

## 🌐 访问服务

启动成功后，可以访问：

- **MCP Server**: http://localhost:8080
  - 健康检查：http://localhost:8080/health
  - JSON-RPC: http://localhost:8080/call

- **Dashboard**: http://localhost:8501
  - MCP 查询页面：点击左侧菜单 "🌐 MCP 查询"

## 🛑 停止服务

按 `Ctrl+C` 停止所有服务

## 🔧 故障排除

### 端口被占用

```bash
# 方法 1: 使用 --clean-ports 自动清理
python scripts/start.py --clean-ports

# 方法 2: 手动清理
lsof -ti:8080 | xargs kill -9
lsof -ti:8501 | xargs kill -9
```

### 服务启动失败

检查日志输出，常见问题：
- 端口被占用
- 依赖未安装：`uv pip install -e .`
- API Key 未配置：检查 `.env` 文件

## 📝 单独启动

如果需要单独启动某个服务：

```bash
# 只启动 MCP HTTP Server
python -m src.mcp_server.http_server --port 8080

# 只启动 Dashboard
streamlit run src/observability/dashboard/app.py --server.port 8501
```

## 🎯 测试 MCP 服务

```bash
# 健康检查
curl http://localhost:8080/health

# 列出可用工具
curl -X POST http://localhost:8080/call \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

# 执行查询
curl -X POST http://localhost:8080/call \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"query_knowledge_hub","arguments":{"query":"test"}}}'
```
