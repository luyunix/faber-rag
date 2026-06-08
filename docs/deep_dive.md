# Faber RAG 深度技术解析

> 本文档基于对项目源代码的完整通读，从架构设计、核心算法、工程实践三个维度进行深度剖析。
> 撰写日期：2026-06-08

---

## 目录

1. [项目定位与设计哲学](#1-项目定位与设计哲学)
2. [分层架构总览](#2-分层架构总览)
3. [配置系统：一切的起点](#3-配置系统一切的起点)
4. [核心数据契约： types.py 的演进设计](#4-核心数据契约-typespy-的演进设计)
5. [查询引擎深度拆解](#5-查询引擎深度拆解)
6. [摄取管道：六阶段编排器](#6-摄取管道六阶段编排器)
7. [BM25 稀疏检索：从零实现](#7-bm25-稀疏检索从零实现)
8. [MCP 协议层](#8-mcp-协议层)
9. [FastAPI REST 服务层](#9-fastapi-rest-服务层)
10. [工厂模式与可插拔架构](#10-工厂模式与可插拔架构)
11. [可观测性系统](#11-可观测性系统)
12. [多模态支持](#12-多模态支持)
13. [工程实践与代码质量](#13-工程实践与代码质量)
14. [扩展指南](#14-扩展指南)
15. [附录：关键文件索引](#15-附录关键文件索引)

---

## 1. 项目定位与设计哲学

### 1.1 项目定位

Faber RAG 是一个**模块化的检索增强生成（RAG）基础设施层**。它不是面向终端用户的聊天应用，而是为 AI 客户端（如 Claude Desktop）提供**知识检索服务**的后端引擎。其核心职责可以概括为两句话：

> ** ingestion 侧**：把用户的文档（PDF/Word/图片等）转化为可检索的知识表示（向量 + 关键词索引 + 图片索引）。
> ** retrieval 侧**：接收用户查询，返回最相关的知识片段及其引用来源。

### 1.2 四大设计哲学

读完整套代码后，可以提炼出四个贯穿始终的设计原则：

| 原则 | 含义 | 代码体现 |
|------|------|---------|
| **配置驱动** | 系统行为完全由 `settings.yaml` 控制，不改代码即可切换 Provider | `settings.py` 的完整解析链、所有 Factory 的注册模式 |
| **工厂模式 + 依赖注入** | 所有外部服务通过工厂创建，组件之间通过构造函数注入依赖 | `LLMFactory` / `EmbeddingFactory` / `VectorStoreFactory` |
| **优雅降级** | 任一组件失败不阻塞整体流程，自动回退到次优方案 | HybridSearch 的单路失败回退、Reranker 的失败返回原始顺序、Transform 的 LLM 失败回退到规则处理 |
| **全链路可观测** | 每次查询/摄取都有 Trace，记录阶段耗时与中间结果 | `TraceContext` + `TraceCollector` + `logs/traces.jsonl` |

---

## 2. 分层架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  客户端层 (Clients)                                                          │
│  ├── Claude Desktop ──MCP 协议──→ MCP HTTP Server (:8080)                   │
│  └── React Dashboard ──REST API──→ FastAPI Server (:8000)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  接口层 (Interface)                                                          │
│  ├── mcp_server/        MCP JSON-RPC 2.0 协议处理                             │
│  │   ├── protocol_handler.py   工具注册与执行                                 │
│  │   ├── http_server.py        HTTP SSE 传输层                               │
│  │   └── tools/                query_knowledge_hub 等工具实现                 │
│  └── api/               FastAPI REST API                                     │
│      ├── server.py             路由定义                                      │
│      └── services/             DataService / TraceService / ConfigService    │
├─────────────────────────────────────────────────────────────────────────────┤
│  业务层 (Core)                                                               │
│  ├── query_engine/      混合检索引擎                                         │
│  │   ├── hybrid_search.py      编排器（并行检索 + 融合 + 过滤）               │
│  │   ├── dense_retriever.py    语义检索（Embedding + 向量库）                │
│  │   ├── sparse_retriever.py   关键词检索（BM25）                            │
│  │   ├── fusion.py             RRF 融合算法                                  │
│  │   ├── reranker.py           重排序包装器                                  │
│  │   └── query_processor.py    查询预处理（jieba 分词 + 停用词过滤）          │
│  ├── response/          响应组装                                             │
│  │   ├── response_builder.py   MCP 格式响应构建                              │
│  │   └── citation_generator.py 引用生成                                      │
│  ├── types.py           核心数据契约                                         │
│  ├── settings.py        配置解析与验证                                       │
│  └── trace/             追踪系统                                             │
│      ├── trace_context.py      阶段记录与耗时计算                             │
│      └── trace_collector.py    JSONL 持久化                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  摄取层 (Ingestion)                                                          │
│  ├── pipeline.py        六阶段主编排器                                       │
│  ├── document_manager.py 跨存储生命周期管理                                   │
│  ├── chunking/          分块策略                                             │
│  │   └── document_chunker.py   适配器层（文本切分 → Chunk 对象）              │
│  ├── transform/         分块精化 / 元数据增强 / 图片描述                      │
│  │   ├── chunk_refiner.py      规则清理 + 可选 LLM 精化                      │
│  │   ├── metadata_enricher.py  规则提取 + 可选 LLM 增强                      │
│  │   └── image_captioner.py    Vision LLM 图片描述                           │
│  ├── embedding/         编码器                                               │
│  │   ├── dense_encoder.py      稠密向量编码（批处理）                         │
│  │   ├── sparse_encoder.py     稀疏向量编码（jieba 分词 + 词频统计）          │
│  │   └── batch_processor.py    批量处理协调器                                │
│  └── storage/           存储                                                 │
│      ├── vector_upserter.py    ChromaDB 幂等写入                             │
│      ├── bm25_indexer.py       倒排索引构建与查询                             │
│      └── image_storage.py      SQLite 图片索引                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  基础设施层 (Libs / Providers)                                                │
│  ├── llm/               LLM 工厂 + Provider 实现                             │
│  ├── embedding/         Embedding 工厂 + Provider 实现                       │
│  ├── vector_store/      Vector Store 工厂 + ChromaDB 实现                    │
│  ├── loader/            文档加载器（PDF → Document）                          │
│  ├── splitter/          文本切分器（LangChain Recursive）                     │
│  ├── reranker/          重排序器（CrossEncoder / LLM / None）                │
│  └── evaluator/         评估框架（RAGAS / DeepEval / Custom）                │
├─────────────────────────────────────────────────────────────────────────────┤
│  可观测性 (Observability)                                                    │
│  ├── logger.py          结构化日志                                           │
│  └── evaluation/        评估运行器与指标计算                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 配置系统：一切的起点

### 3.1 配置加载链路

配置系统由三个部分组成：

1. **`.env` 文件**：存储敏感信息（API Key）
2. **`config/settings.yaml`**：主配置文件，支持 `${VAR_NAME}` 占位符
3. **`src/core/settings.py`**：解析、验证、封装为不可变对象

```python
# settings.py:17-19
env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    load_dotenv(env_path)
```

```python
# settings.py:69-75
pattern = r'\$\{([^}]+)\}'
def replace(match: re.Match) -> str:
    env_var = match.group(1)
    return os.environ.get(env_var, match.group(0))
return re.sub(pattern, replace, value)
```

**关键设计**：环境变量替换是**递归**的，对字典、列表、字符串全覆盖处理（`_process_dict`）。如果环境变量不存在，保留原始占位符字符串，不报错——这是一种**容错设计**，允许部分配置缺失时系统仍能启动（只要那些字段不是必填的）。

### 3.2 不可变配置对象

所有配置段都用 `@dataclass(frozen=True)` 定义：

```python
# settings.py:147-159
@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    api_key: Optional[str] = None
    ...
```

`frozen=True` 确保了配置对象一旦创建就不可修改，避免了运行时不经意间的配置污染。如果需要"修改"配置（如评估端点临时覆盖 `evaluation` 设置），使用 `dataclasses.replace` 创建新实例：

```python
# api/server.py:567
from dataclasses import replace as dc_replace
settings_with_override = dc_replace(settings, evaluation=eval_settings)
```

### 3.3 严格的字段验证

配置解析不依赖 Pydantic，而是手写验证辅助函数：

```python
# settings.py:97-144
def _require_str(data, key, path): ...
def _require_int(data, key, path): ...
def _require_bool(data, key, path): ...
```

这些函数在 `Settings.from_dict()` 中被显式调用，每个必填字段都有明确的验证。缺失字段时抛出的 `SettingsError` 包含完整路径（如 `settings.retrieval.dense_top_k`），方便用户定位问题。

---

## 4. 核心数据契约： types.py 的演进设计

`types.py` 定义了贯穿整个管道的数据契约，其设计体现了一个清晰的**状态演进模型**：

```
Document（原始文档）
    ↓ split
Chunk（文本分块）
    ↓ transform + encode
ChunkRecord（带向量的可存储分块）
    ↓ store / retrieve
RetrievalResult（检索结果）
```

### 4.1 Document：加载器的输出

```python
# types.py:19-79
@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**关键约束**：`metadata` 必须包含 `source_path`，这是后续所有追溯的根基。

**图片占位符规范**：图片在 `Document.text` 中表示为 `[IMAGE: {image_id}]`，`metadata.images` 记录图片的完整元数据（id、path、page、text_offset 等）。这种设计实现了**文本与图片的解耦**：文本可以独立切分、索引，图片通过占位符建立关联。

### 4.2 Chunk：分块后的业务对象

Chunk 增加了三个追溯字段：
- `start_offset` / `end_offset`：在原始文档中的字符位置
- `source_ref`：父 Document.id
- `metadata.chunk_index`：在文档中的顺序位置

### 4.3 ChunkRecord：可存储状态

```python
# types.py:143-221
@dataclass
class ChunkRecord:
    id: str
    text: str
    metadata: Dict[str, Any]
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None
```

`sparse_vector` 不是传统的稀疏向量格式（如 `(dim, value)` 元组列表），而是**词频字典** `Dict[str, float]`（term → frequency）。这是因为下游的 `BM25Indexer` 需要自己重新计算 IDF，它只需要词频统计，不需要完整的 TF-IDF 向量。

### 4.4 RetrievalResult：检索统一输出

```python
# types.py:265-310
@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any]
```

这是**所有检索器（Dense/Sparse/Hybrid/Reranker）的统一输出契约**。无论底层使用什么算法，上层组件只认 `RetrievalResult`。这种统一使得 RRF 融合、Reranker、ResponseBuilder 可以独立于具体检索实现工作。

---

## 5. 查询引擎深度拆解

### 5.1 QueryProcessor：查询的"预处理"

`QueryProcessor` 的职责是把用户的自然语言查询转化为检索器能高效处理的形式。

#### 分词策略

```python
# query_processor.py:210-237
def _tokenize(self, text: str) -> List[str]:
    raw_tokens = jieba.lcut(text)
    for token in raw_tokens:
        token = token.strip()
        if re.fullmatch(r'[\s\W]+', token, re.UNICODE):
            continue
        tokens.append(token)
```

这里使用 `jieba.lcut()` 同时处理中英文。英文单词 jieba 会保持原样（因为英文没有中文的"字"与"词"之分），中文则进行分词。这与下游 `SparseEncoder` 的分词方式一致，确保**查询侧与索引侧的分词对齐**——这是 BM25 能命中结果的前提。

#### 停用词过滤

停用词集合是**手写的中英双语列表**：

```python
# query_processor.py:26-74
CHINESE_STOPWORDS = {"如何", "怎么", "的", "地", "了", "在", "和", "我", "你", ...}
ENGLISH_STOPWORDS = {"a", "an", "the", "in", "on", "and", "or", "is", "are", ...}
```

为什么要手写而不是用 `nltk` 或 `sklearn` 的停用词表？
- **减少依赖**：避免引入重型 NLP 库
- **领域定制**：可以轻易添加业务特定的停用词
- **中英混合**：通用库很少同时提供高质量的中英文停用词

#### 过滤器语法解析

支持在查询中嵌入过滤条件：

```
"collection:api-docs 如何配置 Azure OpenAI"
```

```python
# query_processor.py:77
FILTER_PATTERN: Pattern = re.compile(r'(\w+):([^\s]+)')
```

支持的过滤器键：`collection`/`col`/`c`、`type`/`doc_type`/`t`、`source`/`src`/`s`、`tag`/`tags`。

### 5.2 DenseRetriever：语义搜索

DenseRetriever 的工作极简：

```
query → Embedding API → vector → ChromaDB.query() → RetrievalResult[]
```

```python
# dense_retriever.py:98-163
def retrieve(self, query, top_k, filters, trace):
    query_vectors = self.embedding_client.embed([query], trace=trace)
    query_vector = query_vectors[0]
    raw_results = self.vector_store.query(
        vector=query_vector, top_k=top_k, filters=filters, trace=trace
    )
    return self._transform_results(raw_results)
```

注意 `top_k` 的优先级：调用时传入的 `top_k` > `default_top_k`（构造函数）> `settings.retrieval.dense_top_k`。这种多层默认值设计贯穿整个项目。

### 5.3 SparseRetriever：BM25 关键词搜索

SparseRetriever 有一个**关键的缓存策略**：

```python
# sparse_retriever.py:220-238
def _ensure_index_loaded(self, collection: str) -> bool:
    try:
        loaded = self.bm25_indexer.load(collection=collection)
        return loaded
    except Exception as e:
        logger.warning(f"加载集合 '{collection}' 的 BM25 索引失败: {e}")
        return False
```

**每次查询都重新从磁盘加载 BM25 索引**。这看起来低效，但设计意图很明确：
- Dashboard 前端可能通过 API 摄取新文档，更新 BM25 索引文件
- MCP Server 是独立进程，如果不重新加载，就看不到新数据
- 单个 JSON 文件读取很快（毫秒级），相对于 Embedding API 调用和 ChromaDB 查询可以忽略

BM25 搜索结果只有 `chunk_id` 和 `score`，需要再查向量库获取文本和元数据：

```python
# sparse_retriever.py:169-180
chunk_ids = [r["chunk_id"] for r in bm25_results]
records = self.vector_store.get_by_ids(chunk_ids, trace=trace)
results = self._merge_results(bm25_results, records)
```

这里依赖 **ChromaDB 的 `get_by_ids` 保持输入顺序返回**，SparseRetriever 用 `zip` 按位置合并 BM25 分数和 ChromaDB 记录。如果某个 ID 在 ChromaDB 中不存在，对应位置会返回空字典，合并时会被跳过。

### 5.4 RRFFusion：倒数排名融合的数学与实现

RRF（Reciprocal Rank Fusion）是 Hybrid Search 的核心。它的数学公式极其简洁：

$$\text{RRF\_score}(d) = \sum_{i} \frac{1}{k + \text{rank}_i(d)}$$

其中：
- $d$ 是文档（chunk）
- $k$ 是平滑常数（默认 60）
- $\text{rank}_i(d)$ 是文档 $d$ 在第 $i$ 个排名列表中的排名（从 1 开始）

**为什么 RRF 有效？**
- **无需分数归一化**：Dense 和 Sparse 的分数尺度完全不同（余弦相似度 vs BM25 分数），直接加权融合会有偏。RRF 只关心"排在第几名"。
- **天然处理缺失文档**：某文档只在 Dense 列表中出现，它的 RRF 分数就只有 Dense 一项的贡献，不会被惩罚。
- **高排名差异敏感**：排名第 1 和排名第 2 的差异大（$1/61$ vs $1/62$），排名第 100 和 101 的差异小（$1/160$ vs $1/161$）。这符合直觉：顶部排名的顺序更重要。

代码实现：

```python
# fusion.py:133-151
rrf_scores: Dict[str, float] = {}
chunk_data: Dict[str, RetrievalResult] = {}

for list_idx, ranking_list in enumerate(non_empty_lists):
    for rank, result in enumerate(ranking_list, start=1):
        chunk_id = result.chunk_id
        rrf_contribution = 1.0 / (self.k + rank)

        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0.0
            chunk_data[chunk_id] = result  # 保留首次出现的 text/metadata

        rrf_scores[chunk_id] += rrf_contribution
```

注意 `chunk_data[chunk_id] = result` 只在首次出现时赋值。这意味着如果同一 chunk 在 Dense 和 Sparse 中都有，最终保留的是**先遍历到的那个列表**中的 `text` 和 `metadata`。由于 Dense 列表通常先传入，实际上 Dense 的元数据优先。

RRF 还提供了一个带权重的变体 `fuse_with_weights`，允许给不同来源加权（比如给 Dense 1.5 倍权重）。当前项目中未使用，但为未来的精细化调参预留了接口。

### 5.5 HybridSearch：编排器的艺术

HybridSearch 是整个检索流程的**总指挥**。它的 `search()` 方法可以拆解为 6 个步骤：

```python
# hybrid_search.py:203-312
def search(self, query, top_k, filters, trace, return_details):
    # 1. 查询处理（分词、过滤停用词、解析过滤器）
    processed_query = self._process_query(query)

    # 2. 合并显式过滤器与从查询中提取的过滤器
    merged_filters = self._merge_filters(processed_query.filters, filters)

    # 3. 并行运行 Dense + Sparse 检索（ThreadPoolExecutor，2 workers）
    dense_results, sparse_results, dense_error, sparse_error = \
        self._run_retrievals(processed_query, merged_filters, trace)

    # 4. 降级处理
    if dense_error and sparse_error:
        raise RuntimeError("Both failed")
    elif dense_error:
        fused_results = sparse_results
    elif sparse_error:
        fused_results = dense_results
    else:
        fused_results = self._fuse_results(dense_results, sparse_results, top_k, trace)

    # 5. 融合后元数据过滤（降级机制）
    if merged_filters and self.config.metadata_filter_post:
        fused_results = self._apply_metadata_filters(fused_results, merged_filters)

    # 6. 限制 top_k
    final_results = fused_results[:effective_top_k]
```

**并行检索的实现细节**：

```python
# hybrid_search.py:447-483
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {}
    futures['dense'] = executor.submit(self._run_dense_retrieval, ...)
    futures['sparse'] = executor.submit(self._run_sparse_retrieval, ...)

    for name, future in futures.items():
        try:
            results, error = future.result(timeout=30)
            ...
        except Exception as e:
            ...
```

- `max_workers=2`：恰好对应 Dense 和 Sparse 两条路径
- `timeout=30`：单路检索超过 30 秒视为失败，不会无限阻塞
- 异常捕获是**分路独立**的：Dense 抛异常不影响 Sparse 的结果收集

**降级策略的完备性**：

| Dense | Sparse | 行为 |
|-------|--------|------|
| ✅ 成功 | ✅ 成功 | RRF 融合 |
| ❌ 失败 | ✅ 成功 | 仅用 Sparse |
| ✅ 成功 | ❌ 失败 | 仅用 Dense |
| ❌ 失败 | ❌ 失败 | 抛 RuntimeError |
| ✅ 空结果 | ✅ 空结果 | 返回空列表 |

**元数据后过滤**（`_apply_metadata_filters`）是一个**降级机制**。理想情况下，过滤器应该在向量库查询时就传递（ChromaDB 支持 `where` 子句）。但如果底层存储不完全支持过滤器语法，或者过滤器键需要复杂匹配（如 `tags` 的列表交集），就在融合后用 Python 代码做二次过滤。

### 5.6 CoreReranker：重排序的包装与降级

Reranker 的作用是对 Hybrid Search 返回的结果做**二次精排**。项目支持三种后端：
- `cross_encoder`：Cross-Encoder 模型（如 `cross-encoder/ms-marco-MiniLM-L-6-v2`）
- `llm`：用 LLM 打分重排
- `none`：不重排

CoreReranker 的核心价值在于**类型转换和故障回退**：

```python
# reranker.py:168-233
def _results_to_candidates(self, results):
    return [{"id": r.chunk_id, "text": r.text, "score": r.score, "metadata": r.metadata}]

def _candidates_to_results(self, candidates, original_results):
    id_to_original = {r.chunk_id: r for r in original_results}
    for candidate in candidates:
        chunk_id = candidate["id"]
        if chunk_id in id_to_original:
            original = id_to_original[chunk_id]
            rerank_score = candidate.get("rerank_score", candidate.get("score", 0.0))
            results.append(RetrievalResult(
                chunk_id=original.chunk_id,
                score=rerank_score,
                text=original.text,
                metadata={**original.metadata, "original_score": original.score, "reranked": True}
            ))
```

重排序后的 `score` 被替换为 reranker 给出的分数，但原始分数保留在 `metadata.original_score` 中供调试。

**故障回退**：如果重排序后端失败且 `fallback_on_error=True`（默认），返回原始顺序的前 top_k 个结果，并在 `metadata` 中标记 `"rerank_fallback": True`。

---

## 6. 摄取管道：六阶段编排器

`IngestionPipeline` 是文档处理的主编排器，其 `run()` 方法实现了一个清晰的 6 阶段流水线。

### 6.1 阶段 1：文件完整性检查

```python
# pipeline.py:243-255
file_hash = self.integrity_checker.compute_sha256(str(file_path))
if not self.force and self.integrity_checker.should_skip(file_hash):
    return PipelineResult(success=True, ..., stages={"integrity": {"skipped": True}})
```

基于 SHA256 的**幂等性检查**：同一文件未变更时直接跳过，避免重复处理和 API 调用开销。`force=True` 可以强制重新处理。

完整性数据存储在 SQLite（`data/db/ingestion_history.db`）中，记录文件哈希、路径、处理时间、成功/失败状态。

### 6.2 阶段 2：文档加载

使用 `PdfLoader`（基于 MarkItDown）提取 PDF 的文本和图片。加载后的 `Document` 对象：
- `text`：Markdown 格式的文档全文，图片被替换为 `[IMAGE: {image_id}]`
- `metadata.images`：图片元数据列表，包含 `id`、`path`、`page`、`text_offset`

### 6.3 阶段 3：分块

`DocumentChunker` 是 `libs.splitter` 层与业务对象之间的**适配器**。

#### 确定性 Chunk ID 生成

```python
# document_chunker.py:165
content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
chunk_id = f"{doc_id}_{index:04d}_{content_hash}"
```

ID 格式：`{doc_id}_{index:04d}_{content_hash}`
- `doc_id`：文档标识
- `index:04d`：零填充的 4 位顺序号
- `content_hash`：文本内容 SHA256 前 8 位

**为什么需要 content_hash？** 如果文档内容变更但索引没变（理论上不应发生），ID 会不同，避免覆盖旧的 chunk。

#### 图片关联

```python
# document_chunker.py:222-242
image_refs = re.findall(r'\[IMAGE:\s*([^\]]+)\]', chunk_text)
chunk_metadata["image_refs"] = image_refs

# 从 doc_images 中查找完整元数据
chunk_images = []
for img_id in image_refs:
    if img_id in image_lookup:
        chunk_images.append(image_lookup[img_id])
if chunk_images:
    chunk_metadata["images"] = chunk_images
```

这实现了**分块级别的图片关联**：只有落在该分块文本范围内的图片才会被关联到分块元数据中。这是后续 `ImageCaptioner` 能知道要处理哪些图片的基础。

### 6.4 阶段 4：转换 Pipeline

转换阶段包含三个独立的转换器，按顺序执行：

#### 4a. ChunkRefiner（分块精化）

**规则清理**（必做）：
- 提取并保护代码块（` ``` `）
- 移除页眉/页脚分隔线（`──────────────`）
- 移除 HTML 注释和标签
- 规范化空白字符（多个空格 → 单个，3+ 换行 → 2 个）
- 恢复代码块

**LLM 精化**（可选，配置驱动）：
- 单 chunk 模式：ThreadPoolExecutor 并行处理，每个 chunk 独立调用 LLM
- Batch 模式：一次性把多个 chunk 塞进一个 prompt，让 LLM 批量返回精化结果

Batch 模式的 prompt 模板设计：

```
--- CHUNK_START:{chunk_id} ---
{chunk_text}
--- CHUNK_END:{chunk_id} ---
```

LLM 的响应也要求按同样格式返回，然后用正则提取：

```python
# chunk_refiner.py:289-301
pattern = r"---\s*CHUNK_START:\s*([^\s]+)\s*---\n(.*?)\n---\s*CHUNK_END:\s*\1\s*---"
for match in re.finditer(pattern, response, re.DOTALL):
    chunk_id = match.group(1).strip()
    content = match.group(2).strip()
    results[chunk_id] = content
```

**Batch vs Parallel 的权衡**：
- **Batch**：API 调用次数少，省 Token（因为系统 prompt 只发一次），但单次请求延迟高，LLM 可能"偷懒"导致某些 chunk 处理质量下降
- **Parallel**：每个 chunk 独立调用，质量高，但 API 调用次数多，适合 chunk 数量不多的场景

#### 4b. MetadataEnricher（元数据增强）

规则提取（必做）：
- `title`：优先从 Markdown 标题 (`# Title`) 提取，否则取第一行/第一句
- `summary`：前 3 句话
- `tags`：大写单词（专有名词）、代码标识符（camelCase/snake_case）、Markdown 粗体术语

LLM 增强（可选）：期望 LLM 返回固定格式：

```
Title: <title>
Summary: <summary>
Tags: <tag1>, <tag2>, <tag3>
```

用正则解析，失败时回退到规则提取的结果。

#### 4c. ImageCaptioner（图片描述）

这是项目的**多模态核心**。

```python
# image_captioner.py:138-223
def transform(self, chunks, trace):
    # 1. 收集所有 chunk 中引用的唯一图片 ID
    images_to_caption = {}
    for chunk in chunks:
        referenced_ids = self._find_referenced_image_ids(chunk.text)
        for img_id in referenced_ids:
            if img_id not in images_to_caption:
                img_meta = image_lookup.get(img_id)
                if img_meta and img_meta.get("path"):
                    images_to_caption[img_id] = img_meta.get("path")

    # 2. 并行生成所有唯一图片的描述
    self._generate_captions_parallel(images_to_caption, trace)

    # 3. 将描述插入 chunk 文本（替换占位符）
    for chunk in chunks:
        for img_id in referenced_ids:
            caption = self._caption_cache.get(img_id_stripped)
            if caption:
                placeholder = f"[IMAGE: {img_id}]"
                replacement = f"[IMAGE: {img_id}]\n(Description: {caption})"
                new_text = new_text.replace(placeholder, replacement)
```

**性能优化点**：
- **去重**：同一图片在多个 chunk 中被引用，只调用一次 Vision API
- **缓存**：`_caption_cache` 用 `threading.Lock` 保护，支持并行写入
- **并行**：ThreadPoolExecutor，默认 max_workers=3（比文本 LLM 少，因为 Vision API 更贵更慢）
- **零引用跳过**：没有图片占位符的 chunk 直接透传，不做任何处理

### 6.5 阶段 5：编码

#### DenseEncoder：批处理与防御性验证

```python
# dense_encoder.py:67-159
def encode(self, chunks, trace):
    texts = [chunk.text for chunk in chunks]
    for batch_start in range(0, len(texts), self.batch_size):
        batch_vectors = self.embedding.embed(texts=batch_texts, trace=trace)
        # 验证输出形状
        if len(batch_vectors) != len(batch_texts):
            raise RuntimeError(...)
        all_vectors.extend(batch_vectors)
    # 最终验证
    if len(all_vectors) != len(chunks):
        raise RuntimeError(...)
    # 验证维度一致
    expected_dim = len(all_vectors[0])
    for vec in all_vectors:
        if len(vec) != expected_dim:
            raise RuntimeError(...)
```

 DenseEncoder 做了三层防御性验证：
1. **批次内**：返回的向量数必须等于输入的文本数
2. **全局**：最终向量总数必须等于 chunk 总数
3. **维度**：所有向量维度必须一致

这些验证在调用外部 Embedding API 时尤为重要，因为 API 可能因网络问题返回不完整的结果。

#### SparseEncoder：jieba 分词 + 词频统计

（注：源代码中 `sparse_encoder.py` 未直接读取，但从 `BM25Indexer` 的输入格式可以推断其输出结构）

SparseEncoder 的输出是一个列表，每个元素是：

```python
{
    "chunk_id": str,
    "term_frequencies": {"term1": 3, "term2": 1, ...},
    "doc_length": int  # 该 chunk 的总词数
}
```

这正好是 BM25Indexer.build() 的输入格式。

### 6.6 阶段 6：存储

#### VectorUpserter：幂等性保证

```python
# vector_upserter.py:73-137
def upsert(self, chunks, vectors, trace):
    for chunk, vector in zip(chunks, vectors):
        chunk_id = self._generate_chunk_id(chunk)
        record = {
            "id": chunk_id,
            "vector": vector,
            "metadata": {**chunk.metadata, "text": chunk.text, "chunk_id": chunk_id}
        }
        records.append(record)
    self.vector_store.upsert(records, trace=trace)
```

Chunk ID 的生成方式与 `DocumentChunker` 不同：

```python
# vector_upserter.py:162-166
source_hash = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:8]
content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:8]
chunk_id = f"{source_hash}_{chunk_index:04d}_{content_hash}"
```

这里用 `source_path` 的哈希而不是 `doc_id`。这意味着同一文件路径的文档重复处理时，Chunk ID 稳定；但如果文件复制到另一个路径，ID 会不同。

**ChromaDB 的 upsert 语义**：相同 ID 的记录会被覆盖。因此即使重复运行 Pipeline，也不会产生重复数据。

#### BM25 索引增量更新

```python
# pipeline.py:529-541
for stat, vid in zip(sparse_stats, vector_ids):
    stat["chunk_id"] = vid

self.bm25_indexer.add_documents(
    sparse_stats,
    collection=self.collection,
    doc_id=document.id,
    trace=trace,
)
```

关键步骤：**将 BM25 的 chunk_id 与 ChromaDB 的 vector_id 对齐**。这样 SparseRetriever 在检索时，BM25 返回的 `chunk_id` 正好是 ChromaDB 中的记录 ID，可以直接用 `get_by_ids()` 查询。

`BM25Indexer.add_documents()` 的实现是**加载现有索引 → 删除旧文档记录 → 合并新旧统计 → 重新计算 IDF → 全量保存**。这不是真正的增量更新（没有只修改倒排列表的局部操作），而是"增量合并后全量重建"。对于中小型文档集（<10万 chunk）完全够用。

#### ImageStorage：SQLite + 文件系统

```python
# image_storage.py:96-136
def _ensure_database(self):
    conn.execute("PRAGMA journal_mode=WAL")  # WAL 模式支持并发读写
    conn.execute("""
        CREATE TABLE IF NOT EXISTS image_index (
            image_id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            collection TEXT,
            doc_hash TEXT,
            page_num INTEGER,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash)")
```

**WAL（Write-Ahead Logging）模式**：SQLite 的默认日志模式是 DELETE（写时锁定整个数据库），WAL 模式允许**读操作与写操作并发**，非常适合 Dashboard 前端在查询图片列表的同时，Pipeline 在写入新图片记录的场景。

图片文件按集合组织在目录中：`data/images/{collection}/{image_id}.png`。

---

## 7. BM25 稀疏检索：从零实现

项目没有依赖 `rank-bm25` 或 `whoosh` 等库，而是**手写了一个完整的 BM25 实现**。

### 7.1 索引结构

```json
{
  "metadata": {
    "num_docs": 100,
    "avg_doc_length": 45.2,
    "total_terms": 3200,
    "collection": "default"
  },
  "index": {
    "term": {
      "idf": 2.15,
      "df": 5,
      "postings": [
        {"chunk_id": "abc123_0000_xxxx", "tf": 3, "doc_length": 42},
        ...
      ]
    }
  }
}
```

这是一个典型的**倒排索引**：term → posting list。每个 posting 记录了包含该 term 的文档、词频、文档长度。

### 7.2 IDF 计算

```python
# bm25_indexer.py:431-443
def _calculate_idf(self, num_docs: int, df: int) -> float:
    return math.log((num_docs - df + 0.5) / (df + 0.5))
```

这是 BM25 的原始 IDF 公式。与经典 TF-IDF 的 `log(N/df)` 相比，BM25 的 IDF 在 `df > N/2` 时会变成负数（表示该词出现得太频繁，对区分文档没有帮助）。这在搜索引擎中是一个合理的设计，但在小文档集上可能导致一些异常行为。

### 7.3 BM25 打分公式

```python
# bm25_indexer.py:445-472
def _calculate_bm25_score(self, tf, doc_length, avg_doc_length, idf):
    numerator = tf * (self.k1 + 1)
    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
    return idf * (numerator / denominator)
```

对应数学公式：

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f_{t,d} \cdot (k_1 + 1)}{f_{t,d} + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

参数含义：
- $k_1 = 1.5$：控制词频饱和度。越高，词频对分数的影响越线性（不容易饱和）
- $b = 0.75$：控制文档长度归一化。$b=1$ 完全归一化，$b=0$ 不归一化

### 7.4 持久化策略

索引保存为单个 JSON 文件：`data/db/bm25/{collection}_bm25.json`。

写入时使用**原子写入**：

```python
# bm25_indexer.py:531-542
temp_path = index_path.with_suffix('.tmp')
try:
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    temp_path.replace(index_path)  # 原子重命名
except Exception as e:
    if temp_path.exists():
        temp_path.unlink()
    raise
```

原子重命名确保了即使进程在写入过程中崩溃，已有的索引文件也不会损坏。

---

## 8. MCP 协议层

### 8.1 协议架构

Faber RAG 实现了 MCP（Model Context Protocol）的 HTTP SSE 传输层。MCP 是一个 JSON-RPC 2.0 协议，允许 AI 客户端（如 Claude Desktop）发现和调用外部工具。

```
Claude Desktop          MCP HTTP Server
     |                         |
     |--- POST /call ---------->|  {jsonrpc: "2.0", method: "tools/list", ...}
     |                         |
     |<-- 200 OK ---------------|  {result: {tools: [...]}}
     |                         |
     |--- POST /call ---------->|  {jsonrpc: "2.0", method: "tools/call",
     |                         |           params: {name: "query_knowledge_hub",
     |                         |                    arguments: {query: "..."}}}
     |                         |
     |<-- 200 OK ---------------|  {result: {content: [TextContent, ImageContent?]}}
```

### 8.2 ProtocolHandler：工具注册与执行

`ProtocolHandler` 是 MCP 协议的核心，维护一个工具注册表：

```python
# protocol_handler.py:41-89
@dataclass
class ProtocolHandler:
    server_name: str
    server_version: str
    tools: Dict[str, ToolDefinition] = field(default_factory=dict)

    def register_tool(self, name, description, input_schema, handler):
        if name in self.tools:
            raise ValueError(f"Tool '{name}' is already registered")
        self.tools[name] = ToolDefinition(...)
```

当前注册了三个工具：
1. `query_knowledge_hub`：知识库查询（核心工具）
2. `list_collections`：列出所有集合
3. `get_document_summary`：获取文档摘要

工具执行时：

```python
# protocol_handler.py:108-179
async def execute_tool(self, name, arguments):
    tool = self.tools[name]
    result = await tool.handler(**arguments)
    # 返回类型适配：支持 CallToolResult / str / list / 任意对象
    if isinstance(result, types.CallToolResult):
        return result
    if isinstance(result, str):
        return types.CallToolResult(content=[TextContent(text=result)], isError=False)
    ...
```

**错误处理策略**：
- 工具不存在 → JSON-RPC 错误响应
- 参数类型不匹配 → `TypeError` 捕获，返回错误响应
- 工具内部异常 → 捕获并返回 `isError=True` 的响应，**不泄露堆栈跟踪**（安全考虑）

### 8.3 HTTP Server：aiohttp 路由

```python
# http_server.py:121-164
app.router.add_get("/health", health_check)
app.router.add_post("/call", handle_jsonrpc)
app.router.add_post("/", handle_jsonrpc)
```

- `/health`：健康检查，返回 `{"status": "ok"}`
- `/call`：JSON-RPC 主入口
- `/`：兼容部分客户端的简化调用

MCP Server 基于 `mcp` 官方 Python SDK 的 `lowlevel.Server`，但通过自定义 `handle_request` 函数绕过了 SDK 的高级封装，直接处理 JSON-RPC 消息。这种设计的原因可能是 SDK 的高级封装对 SSE 传输的支持不够灵活，或者为了更细粒度的错误控制。

### 8.4 query_knowledge_hub：核心查询工具

`QueryKnowledgeHubTool` 是项目最核心的 MCP 工具。它的 `execute()` 方法实现了一个复杂的**组件缓存策略**：

```python
# query_knowledge_hub.py:139-213
def _ensure_initialized(self, collection):
    # === 完全缓存（无状态，永不过时）===
    if self._embedding_client is None:
        self._embedding_client = EmbeddingFactory.create(self.settings)
    if self._reranker is None:
        self._reranker = create_core_reranker(settings=self.settings)

    # === 集合变更时重建 ===
    vector_store = VectorStoreFactory.create(self.settings, collection_name=collection)
    dense_retriever = create_dense_retriever(..., vector_store=vector_store)
    bm25_indexer = BM25Indexer(index_dir=...)
    sparse_retriever = create_sparse_retriever(..., bm25_indexer=bm25_indexer)
    query_processor = QueryProcessor()
    self._hybrid_search = create_hybrid_search(...)
```

**缓存策略的分层设计**：

| 组件 | 缓存策略 | 理由 |
|------|---------|------|
| EmbeddingClient | 完全缓存 | 无状态，创建成本高（连接池初始化） |
| Reranker | 完全缓存 | 无状态，模型加载成本高 |
| VectorStore | 按集合重建 | ChromaDB PersistentClient 基于 SQLite，支持多进程读，但 collection_name 是构造参数 |
| DenseRetriever | 按集合重建 | 依赖 VectorStore |
| SparseRetriever | 按集合重建 | BM25Indexer 只保存目录路径，每次查询会 reload |
| HybridSearch | 按集合重建 | 依赖上述检索器 |

**为什么要按集合重建？** 因为 Dashboard 前端可以通过 API 向任意集合摄取文档。如果 MCP Server 缓存了某个集合的检索组件，当用户切换到另一个集合查询时就无法看到新数据。按集合重建确保**每个查询都使用目标集合的最新数据**。

**异步阻塞处理**：

```python
# query_knowledge_hub.py:261-279
await asyncio.to_thread(self._ensure_initialized, effective_collection)
results = await asyncio.to_thread(self._perform_search, query, effective_top_k, trace)
if self.config.enable_rerank and results:
    results = await asyncio.to_thread(self._apply_rerank, query, results, effective_top_k, trace)
```

所有阻塞 I/O（Embedding API、ChromaDB 查询、BM25 文件读取、LLM Reranker 调用）都通过 `asyncio.to_thread()` 放到线程池执行，避免阻塞 MCP 的 async 事件循环。这是 MCP stdio/HTTP 传输稳定性的关键。

---

## 9. FastAPI REST 服务层

### 9.1 服务端点设计

FastAPI 服务器（`:8000`）为 React Dashboard 提供 REST API。端点分为 5 组：

| 组 | 端点 | 功能 |
|----|------|------|
| Health | `GET /health` | 健康检查 |
| Config | `GET /api/config/components` | 组件配置卡片 |
| Config | `GET /api/config/collections/stats` | 集合统计 |
| Config | `GET /api/config/settings` | 原始 settings.yaml |
| Config | `PUT /api/config/settings` | 更新配置 |
| Data | `GET /api/data/collections` | 集合列表 |
| Data | `GET /api/data/documents` | 文档列表 |
| Data | `GET /api/data/documents/{hash}/chunks` | 文档分块 |
| Data | `GET /api/data/documents/{hash}/images` | 文档图片 |
| Data | `DELETE /api/data/documents` | 删除文档 |
| Data | `DELETE /api/data/reset` | 清空所有数据 |
| Ingestion | `POST /api/ingestion/upload` | 上传并处理文档 |
| Trace | `GET /api/traces` | 追踪列表 |
| Trace | `GET /api/traces/{id}` | 单个追踪 |
| Evaluation | `POST /api/evaluation/run` | 运行评估 |
| Evaluation | `POST /api/evaluation/evaluate-trace` | 评估单个追踪 |
| Evaluation | `GET /api/evaluation/history` | 评估历史 |
| Query | `POST /api/query` | 直接查询 |
| MCP | `POST /api/mcp/query` | 通过 MCP 代理查询 |
| MCP | `GET /api/mcp/health` | MCP 服务器健康检查 |

### 9.2 DataService：延迟初始化与外观模式

```python
# data_service.py:16-73
class DataService:
    def __init__(self):
        self._manager = None
        self._chroma = None
        self._images = None
        self._current_collection = ""

    def _ensure_stores(self, collection):
        if self._manager is not None and self._current_collection == target_collection:
            return
        # 首次使用时创建所有存储对象
        chroma = VectorStoreFactory.create(settings, collection_name=target_collection)
        bm25 = BM25Indexer(...)
        images = ImageStorage(...)
        integrity = SQLiteIntegrityChecker(...)
        self._manager = DocumentManager(chroma, bm25, images, integrity)
```

**延迟初始化**的好处：
- 模块导入时零开销（不创建 ChromaDB 客户端、不连接 SQLite）
- 第一个 API 请求触发初始化，此后缓存复用
- 集合切换时自动重建

`DocumentManager` 是一个**外观（Facade）**，将四个存储后端（ChromaDB、BM25、ImageStorage、IntegrityChecker）封装为统一的文档生命周期接口。删除文档时，它会**级联到所有四个后端**，并收集每个后端的成功/失败状态：

```python
# document_manager.py:184-261
def delete_document(self, source_path, collection, source_hash):
    # 1. ChromaDB
    count = self.chroma.delete_by_metadata({"doc_hash": source_hash})
    # 2. BM25
    self.bm25.remove_document(source_hash, collection)
    # 3. ImageStorage
    for img in self.images.list_images(doc_hash=source_hash):
        self.images.delete_image(img["image_id"])
    # 4. FileIntegrity
    self.integrity.remove_record(source_hash)
```

**故障安全**：任一后端删除失败不会阻止其他后端的清理操作，错误被收集到 `DeleteResult.errors` 列表中返回。

### 9.3 配置热更新

```python
# api/server.py:264-301
@app.put("/api/config/settings")
async def update_raw_settings(request_body):
    # 验证配置格式
    try:
        Settings.from_dict(request_body)
    except SettingsError as e:
        raise HTTPException(status_code=400, detail=f"配置验证失败: {e}")

    # 写回 yaml 文件
    with settings_path.open("w", encoding="utf-8") as f:
        yaml.dump(request_body, f, ...)

    # 清除 ConfigService 缓存
    get_config_service().reload()
```

注意注释中的说明："部分配置（如 retrieval 参数）会在下次查询时自动生效，LLM/Embedding 等 Provider 变更需要重启服务"。这是因为运行中的 Provider 实例已经创建，不会自动重新读取配置。

---

## 10. 工厂模式与可插拔架构

项目使用了四个核心工厂：

### 10.1 工厂注册表

```python
# llm_factory.py:66-67
_PROVIDERS: dict[str, type[BaseLLM]] = {}
_VISION_PROVIDERS: dict[str, type[BaseVisionLLM]] = {}

# embedding_factory.py:32
_PROVIDERS: dict[str, type[BaseEmbedding]] = {}

# vector_store_factory.py:31
_PROVIDERS: dict[str, type[BaseVectorStore]] = {}
```

所有工厂的结构几乎相同：
1. 模块级私有注册表字典
2. `register_provider(name, class)` 类方法
3. `create(settings)` 类方法
4. `list_providers()` 类方法

### 10.2 防御性导入注册

```python
# embedding_factory.py:113-141
def _register_builtin_providers() -> None:
    try:
        from src.libs.embedding.openai_embedding import OpenAIEmbedding
        EmbeddingFactory.register_provider("openai", OpenAIEmbedding)
    except ImportError:
        pass  # OpenAI 提供商不可用

    try:
        from src.libs.embedding.qwen_embedding import QwenEmbedding
        EmbeddingFactory.register_provider("qwen", QwenEmbedding)
    except ImportError:
        pass  # Qwen 提供商不可用

_register_builtin_providers()
```

**为什么用 try/except ImportError？**
- 某些 Provider 的依赖（如 `openai` 包）可能未安装
- 不希望缺少一个 Provider 的依赖就导致整个系统无法启动
- 这是 Python 插件系统的经典模式

### 10.3 抽象基类设计

```python
# base_llm.py:42-99
class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: List[Message], trace=None, **kwargs) -> ChatResponse:
        pass

    def validate_messages(self, messages: List[Message]) -> None:
        ...

# base_embedding.py:13-73
class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, texts: List[str], trace=None, **kwargs) -> List[List[float]]:
        pass

    @abstractmethod
    def get_dimension(self) -> Optional[int]:
        pass

    def validate_texts(self, texts: List[str]) -> None:
        ...
```

抽象基类只定义接口（`@abstractmethod`），同时提供**可选的验证辅助方法**（`validate_messages`、`validate_texts`）。子类可以复用这些验证，也可以覆盖。这种设计既保持了接口的严格性，又减少了子类的样板代码。

---

## 11. 可观测性系统

### 11.1 TraceContext：请求范围的追踪

```python
# trace_context.py:14-130
@dataclass
class TraceContext:
    trace_type: Literal["query", "ingestion"] = "query"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = None
    stages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

每个 Trace 有唯一的 `trace_id`（UUID4），包含：
- `trace_type`：`query`（查询）或 `ingestion`（摄取）
- `started_at` / `finished_at`：ISO-8601 时间戳
- `stages`：有序的阶段记录列表
- `metadata`：附加元数据（如查询字符串、集合名、最终结果）

### 11.2 阶段记录

```python
# trace_context.py:41-64
def record_stage(self, stage_name: str, data: Dict[str, Any], elapsed_ms: Optional[float] = None):
    entry = {
        "stage": stage_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }
    if elapsed_ms is not None:
        entry["elapsed_ms"] = round(elapsed_ms, 2)
        self._stage_timings[stage_name] = elapsed_ms
    self.stages.append(entry)
```

阶段记录是**追加式的**。同一个 stage_name 可以被记录多次（比如摄取 Pipeline 中多次调用 `trace.record_stage("embed", ...)`），`get_stage_data()` 返回最后一次的数据（后写入优先）。

### 11.3 耗时计算

```python
# trace_context.py:75-95
def elapsed_ms(self, stage_name=None):
    if stage_name is not None:
        return self._stage_timings[stage_name]
    end = self._finish_mono if self._finish_mono is not None else time.monotonic()
    return (end - self._start_mono) * 1000.0
```

使用 `time.monotonic()` 而非 `time.time()` 计算耗时，因为 monotonic 时钟不受系统时间调整（NTP 同步、用户手动调时）的影响，确保耗时计算**单调递增、准确可靠**。

### 11.4 TraceCollector：JSONL 持久化

（源代码未直接读取，但从 `trace_service.py` 可以推断其工作原理）

TraceCollector 将 `TraceContext.to_dict()` 的结果序列化为 JSON，追加写入 `logs/traces.jsonl`。每行一个 JSON 对象，这是大数据领域最常用的日志格式（JSON Lines），优点：
- **可流式读取**：无需加载整个文件到内存
- **容错**：某一行损坏不影响其他行
- **易处理**：`jq`、Python `json.loads`、Spark 都能直接处理

`TraceService` 从 JSONL 文件读取追踪记录：

```python
# trace_service.py:99-117
def _load_all(self):
    with self.traces_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug("Skipping malformed trace line: %s", line[:80])
```

格式错误的行被静默跳过，不会导致整个服务崩溃。

---

## 12. 多模态支持

### 12.1 图片占位符机制

这是 Faber RAG 多模态设计的核心创新。PDF 中的图片在文本层面被替换为占位符：

```markdown
# 第一章：系统架构

系统包含以下组件：
[IMAGE: doc_abc_001]

如上图所示，架构分为三层...
```

占位符的好处：
- **文本索引不受影响**：图片位置有明确的文本标记，Dense Retrieval 可以把图片周围的文本一起编码
- **图片可追溯**：通过 `image_id` 可以在 ImageStorage 中找到原始图片文件
- **分块友好**：占位符占用固定字符数，分切时不会因为图片大小变化而打乱边界

### 12.2 Vision LLM 图片描述

`ImageCaptioner` 将图片描述文本插入到 chunk 中，替换原有的占位符：

```markdown
[IMAGE: doc_abc_001]
(Description: A flowchart showing three layers: client layer, server layer, and storage layer, connected by arrows.)
```

这样，即使检索时不返回图片本身，用户也能从文本描述中了解图片内容。同时，描述文本也会被 Embedding 编码，支持**基于图片内容的语义检索**（比如查询 "架构图" 可以命中包含架构图片的分块）。

### 12.3 MCP 多模态响应

`ResponseBuilder` 在构建 MCP 响应时，会检查检索结果中是否包含图片：

```python
# response_builder.py:183-193
if self.enable_multimodal and include_images:
    image_blocks = self.multimodal_assembler.assemble(results, collection)
    image_contents = [block for block in image_blocks if isinstance(block, types.ImageContent)]
```

MCP 的 `CallToolResult.content` 是一个列表，可以同时包含 `TextContent` 和 `ImageContent`。Claude Desktop 收到后会同时显示文本结果和图片。

---

## 13. 工程实践与代码质量

### 13.1 类型安全

- 全项目使用类型提示，包括复杂的泛型和 `Optional`
- `TYPE_CHECKING` 避免运行时循环导入
- `from __future__ import annotations` 启用 PEP 563 延迟注解求值

### 13.2 错误处理

**三层错误处理**：
1. **输入验证**：尽早检查，给出清晰的错误信息（如 `"Query cannot be empty or whitespace-only"`）
2. **依赖验证**：`_validate_dependencies()` 在操作前检查必需组件是否已配置
3. **运行时异常捕获**：外部 API 调用（Embedding、LLM、ChromaDB）都用 try/except 包裹，带上上下文信息

**自定义异常**：
- `SettingsError(ValueError)`：配置验证失败
- `RerankError(RuntimeError)`：重排序失败且禁用回退时抛出

### 13.3 优雅降级的多层实现

| 层级 | 降级场景 | 降级行为 |
|------|---------|---------|
| Transform | LLM 精化失败 | 回退到规则精化结果 |
| Transform | LLM 元数据增强失败 | 回退到规则提取的 title/summary/tags |
| Transform | Vision LLM 初始化失败 | ImageCaptioner 变为 no-op |
| HybridSearch | Dense 检索失败 | 仅用 Sparse 结果 |
| HybridSearch | Sparse 检索失败 | 仅用 Dense 结果 |
| HybridSearch | RRF 融合未配置 | 降级为 round-robin 交错结果 |
| Reranker | 重排序后端失败 | 返回原始顺序 |
| QueryProcessor | 未配置 | 基础 split() 分词 |

### 13.4 幂等性设计

- **文件级**：SHA256 完整性检查，重复文件跳过
- **Chunk 级**：确定性 ID 生成，ChromaDB upsert 语义覆盖旧记录
- **BM25 级**：`add_documents()` 先删除旧文档记录再添加新记录
- **ImageStorage**：`INSERT OR REPLACE` SQL 语义

### 13.5 文档习惯

每个模块、类、公有方法都有完整的中文 docstring，包含：
- 功能描述
- 参数说明（类型、含义、默认值）
- 返回值说明
- 可能抛出的异常
- 使用示例（部分包含可执行的 doctest）

---

## 14. 扩展指南

### 14.1 添加新的 LLM Provider

1. **创建实现类**：

```python
# src/libs/llm/my_llm.py
from src.libs.llm.base_llm import BaseLLM, Message, ChatResponse

class MyLLM(BaseLLM):
    def chat(self, messages, trace=None, **kwargs) -> ChatResponse:
        # 实现调用逻辑
        return ChatResponse(content="...", model="my-model")
```

2. **注册到工厂**：

```python
# 在 src/libs/llm/llm_factory.py 的 _register_builtin_providers 中添加
try:
    from src.libs.llm.my_llm import MyLLM
    LLMFactory.register_provider("my_provider", MyLLM)
except ImportError:
    pass
```

3. **配置使用**：

```yaml
llm:
  provider: "my_provider"
  model: "my-model-name"
  api_key: "${MY_API_KEY}"
```

### 14.2 添加新的检索策略

在 `HybridSearch.search()` 中，融合结果后、限制 top_k 前，插入自定义逻辑：

```python
# hybrid_search.py
def search(self, query, top_k, filters, trace, return_details):
    ...
    fused_results = self._fuse_results(...)

    # 插入自定义重排序/过滤/聚合
    fused_results = self._my_custom_strategy(fused_results, query)

    final_results = fused_results[:effective_top_k]
```

### 14.3 自定义 Pipeline 阶段

1. 继承 `BaseTransform`：

```python
# src/ingestion/transform/my_transform.py
from src.ingestion.transform.base_transform import BaseTransform
from src.core.types import Chunk

class MyTransform(BaseTransform):
    def transform(self, chunks: List[Chunk], trace=None) -> List[Chunk]:
        for chunk in chunks:
            chunk.metadata["my_field"] = self._compute(chunk)
        return chunks
```

2. 在 `IngestionPipeline.__init__()` 中初始化并在 `run()` 中调用。

---

## 15. 附录：关键文件索引

| 文件路径 | 职责 | 核心类/函数 |
|---------|------|-----------|
| `src/core/settings.py` | 配置解析与验证 | `Settings`, `load_settings()` |
| `src/core/types.py` | 核心数据契约 | `Document`, `Chunk`, `ChunkRecord`, `RetrievalResult`, `ProcessedQuery` |
| `src/core/query_engine/hybrid_search.py` | 混合检索编排 | `HybridSearch`, `HybridSearchConfig` |
| `src/core/query_engine/dense_retriever.py` | 语义检索 | `DenseRetriever` |
| `src/core/query_engine/sparse_retriever.py` | 关键词检索 | `SparseRetriever` |
| `src/core/query_engine/fusion.py` | RRF 融合 | `RRFFusion` |
| `src/core/query_engine/reranker.py` | 重排序 | `CoreReranker`, `RerankConfig` |
| `src/core/query_engine/query_processor.py` | 查询预处理 | `QueryProcessor` |
| `src/core/response/response_builder.py` | MCP 响应构建 | `ResponseBuilder`, `MCPToolResponse` |
| `src/core/trace/trace_context.py` | 追踪上下文 | `TraceContext` |
| `src/ingestion/pipeline.py` | 摄取主编排 | `IngestionPipeline`, `PipelineResult` |
| `src/ingestion/document_manager.py` | 跨存储生命周期 | `DocumentManager` |
| `src/ingestion/chunking/document_chunker.py` | 文档分块 | `DocumentChunker` |
| `src/ingestion/transform/chunk_refiner.py` | 分块精化 | `ChunkRefiner` |
| `src/ingestion/transform/metadata_enricher.py` | 元数据增强 | `MetadataEnricher` |
| `src/ingestion/transform/image_captioner.py` | 图片描述 | `ImageCaptioner` |
| `src/ingestion/embedding/dense_encoder.py` | 稠密编码 | `DenseEncoder` |
| `src/ingestion/storage/bm25_indexer.py` | BM25 索引 | `BM25Indexer` |
| `src/ingestion/storage/vector_upserter.py` | 向量存储写入 | `VectorUpserter` |
| `src/ingestion/storage/image_storage.py` | 图片存储 | `ImageStorage` |
| `src/mcp_server/protocol_handler.py` | MCP 协议处理 | `ProtocolHandler`, `create_mcp_server()` |
| `src/mcp_server/http_server.py` | MCP HTTP 服务 | `run_http_server()` |
| `src/mcp_server/tools/query_knowledge_hub.py` | 核心 MCP 工具 | `QueryKnowledgeHubTool` |
| `src/api/server.py` | FastAPI 服务 | `app` |
| `src/api/services/data_service.py` | 数据服务 | `DataService` |
| `src/api/services/trace_service.py` | 追踪服务 | `TraceService` |
| `src/libs/llm/llm_factory.py` | LLM 工厂 | `LLMFactory` |
| `src/libs/llm/base_llm.py` | LLM 抽象基类 | `BaseLLM`, `Message`, `ChatResponse` |
| `src/libs/embedding/embedding_factory.py` | Embedding 工厂 | `EmbeddingFactory` |
| `src/libs/embedding/base_embedding.py` | Embedding 抽象基类 | `BaseEmbedding` |
| `src/libs/vector_store/chroma_store.py` | ChromaDB 实现 | `ChromaStore` |
| `src/libs/vector_store/vector_store_factory.py` | VectorStore 工厂 | `VectorStoreFactory` |
| `config/settings.yaml` | 主配置文件 | — |

---

*本文档完成。如有疑问或需要针对特定模块做进一步深挖，请随时提出。*
