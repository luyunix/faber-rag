# API 服务器说明

## 概述

`src/api/server.py` 是一个 FastAPI 服务器，为 Faber RAG UI 前端提供 REST API 接口。

## 安装依赖

```bash
# 使用 uv 安装
uv pip install fastapi uvicorn pydantic aiohttp

# 或使用 pip
pip install fastapi uvicorn pydantic aiohttp
```

## 启动服务器

### 方法 1：使用脚本

```bash
python scripts/start_api.py
```

### 方法 2：直接运行

```bash
python -m src.api.server
```

### 方法 3：使用 uvicorn

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

## API 端点

### 健康检查
- `GET /health` - 检查服务器状态

### 配置服务
- `GET /api/config/components` - 获取组件配置列表
- `GET /api/config/collections/stats` - 获取集合统计信息

### 数据服务
- `GET /api/data/collections` - 获取所有集合名称
- `GET /api/data/documents` - 获取文档列表
- `GET /api/data/documents/{source_hash}/chunks` - 获取文档片段
- `GET /api/data/documents/{source_hash}/images` - 获取文档图片
- `DELETE /api/data/documents` - 删除文档
- `DELETE /api/data/reset` - 清空所有数据
- `POST /api/ingestion/upload` - 上传并处理文档

### 追踪服务
- `GET /api/traces` - 获取追踪列表
- `GET /api/traces/{trace_id}` - 获取单个追踪详情

### 评估服务
- `POST /api/evaluation/run` - 运行评估
- `POST /api/evaluation/evaluate-trace` - 评估单个追踪
- `GET /api/evaluation/history` - 获取评估历史

### MCP 服务
- `POST /api/mcp/query` - 通过 MCP 协议执行查询
- `GET /api/mcp/health` - 检查 MCP 服务器健康状态

## API 文档

启动服务器后，访问以下地址查看自动生成的 API 文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 同时启动多个服务

开发时，你可能需要同时运行以下服务：

1. **API 服务器** (端口 8000)
   ```bash
   python scripts/start_api.py
   ```

2. **MCP 服务器** (端口 8080)
   ```bash
   python scripts/start_mcp.py
   ```

3. **前端开发服务器** (端口 3000)
   ```bash
   cd ../faber-rag-ui
   npm run dev
   ```

## 环境变量

创建 `.env` 文件配置以下环境变量：

```env
# API 服务器配置
API_HOST=0.0.0.0
API_PORT=8000

# 数据库和存储配置
VECTOR_STORE_PERSIST_DIR=data/db/chroma
BM25_INDEX_DIR=data/db/bm25
IMAGE_DB_PATH=data/db/image_index.db
IMAGE_ROOT=data/images
```
