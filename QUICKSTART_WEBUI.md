# Faber RAG Web UI 快速启动指南

## 项目结构

```
faber-rag/                  # Python 后端项目
├── src/
│   ├── api/               # FastAPI 服务器
│   │   └── server.py     # REST API 实现
│   ├── observability/
│   │   └── dashboard/    # Streamlit 仪表盘 (旧)
│   └── ...
├── scripts/
│   ├── start_api.py      # 启动 API 服务器
│   ├── start_all.sh      # 启动所有服务
│   └── ...
└── pyproject.toml

faber-rag-ui/              # TypeScript 前端项目
├── src/
│   ├── pages/            # 页面组件
│   ├── services/         # API 调用
│   └── ...
└── package.json
```

## 快速启动

### 方法 1：使用启动脚本（推荐）

```bash
cd /Users/lyn/Desktop/faber-rag
./scripts/start_all.sh
```

选择选项 3 启动全部服务。

### 方法 2：手动启动

**终端 1 - API 服务器**
```bash
cd /Users/lyn/Desktop/faber-rag
python scripts/start_api.py
```

**终端 2 - MCP 服务器**
```bash
cd /Users/lyn/Desktop/faber-rag
python scripts/start_mcp.py
```

**终端 3 - 前端**
```bash
cd /Users/lyn/WebstormProjects/faber-rag-ui
npm run dev
```

## 访问地址

- **前端界面**: http://localhost:3000
- **API 文档**: http://localhost:8000/docs
- **API 服务**: http://localhost:8000
- **MCP 服务**: http://localhost:8080

## 首次使用

1. 确保已安装所有依赖：
   ```bash
   # 后端依赖
   cd /Users/lyn/Desktop/faber-rag
   uv pip install fastapi uvicorn pydantic aiohttp

   # 前端依赖
   cd /Users/lyn/WebstormProjects/faber-rag-ui
   npm install
   ```

2. 启动服务后，访问 http://localhost:3000

3. 使用"文档处理"页面上传 PDF、TXT、MD 或 DOCX 文件

4. 使用"MCP 查询"页面进行知识库查询

## 配置说明

### 后端配置

编辑 `.env` 文件（在 faber-rag 目录）：
```env
# API 服务器
API_HOST=0.0.0.0
API_PORT=8000

# 存储
VECTOR_STORE_PERSIST_DIR=data/db/chroma
BM25_INDEX_DIR=data/db/bm25
IMAGE_ROOT=data/images

# LLM 配置（根据你的提供商配置）
OPENAI_API_KEY=your_api_key
```

### 前端配置

编辑 `.env` 文件（在 faber-rag-ui 目录）：
```env
VITE_API_BASE_URL=http://localhost:8000/api
VITE_MCP_SERVER_URL=http://localhost:8080
```

## 功能说明

### 1. 系统总览
- 查看组件配置（LLM、Embedding、Vector Store 等）
- 查看数据统计（文档数、Chunk 数、查询数）

### 2. 数据浏览
- 浏览已摄取的文档
- 查看文档片段和图片
- 删除文档
- 清空所有数据

### 3. 文档处理
- 上传文件（PDF、TXT、MD、DOCX）
- 自动分块、向量化、存储
- 查看处理历史和阶段耗时

### 4. 查询历史
- 查看所有查询追踪
- 查看每个查询的阶段详情
- 使用 Ragas 进行单查询评估

### 5. 评估面板
- 运行评估测试
- 查看评估指标
- 查看评估历史

### 6. MCP 查询
- 通过 MCP 协议查询知识库
- 混合检索 + 重排序
- 查看查询历史

## 故障排除

### 端口被占用

如果 8000、8080 或 3000 端口被占用，修改对应的配置：

- API 端口：修改 `src/api/server.py` 中的 `port=8000`
- MCP 端口：修改 MCP 服务器启动参数
- 前端端口：修改前端 `vite.config.ts` 中的 `port: 3000`

### 连接失败

1. 检查后端服务是否正常运行
2. 检查前端 `.env` 中的 `VITE_API_BASE_URL` 是否正确
3. 查看浏览器控制台的错误信息

### 依赖安装失败

```bash
# 使用镜像源
pip install -i https://mirrors.aliyun.com/pypi/simple/ fastapi uvicorn
npm install --registry=https://registry.npmmirror.com
```

## 开发说明

### 添加新的 API 端点

1. 在 `src/api/server.py` 中添加新的路由函数
2. 在 `src/services/api.ts` 中添加对应的 API 调用函数
3. 在 `src/types/index.ts` 中添加类型定义

### 添加新的页面

1. 在 `src/pages/` 创建新的页面组件
2. 在 `src/App.tsx` 中添加路由
3. 在 `src/components/Layout.tsx` 的导航配置中添加链接
