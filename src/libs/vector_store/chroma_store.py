"""ChromaDB VectorStore 实现。

本模块使用 ChromaDB 提供了 BaseVectorStore 的具体实现，
ChromaDB 是一个轻量级、开源的嵌入数据库，专为本地优先部署而设计。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from src.core.settings import resolve_path
from src.libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)

# 禁用 ChromaDB 遥测错误日志
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


class ChromaStore(BaseVectorStore):
    """VectorStore 的 ChromaDB 实现。
    
    此类使用 ChromaDB 提供本地优先、持久化的向量存储。
    它支持 upsert、query 和元数据过滤操作。
    
    应用的设计原则：
    - 可插拔：实现 BaseVectorStore 接口，可与其他提供者互换
    - 配置驱动：所有设置（persist_directory、collection_name）来自 settings.yaml
    - 幂等性：相同 ID 的 upsert 操作会覆盖现有记录
    - 可观测：接受可选的 TraceContext（预留给 Stage F）
    - 快速失败：初始化时验证依赖和配置
    
    属性:
        client: ChromaDB 客户端实例
        collection: 用于存储向量的 ChromaDB 集合
        collection_name: 集合名称
        persist_directory: 持久化存储的目录路径
    
    示例:
        >>> settings = Settings.load('config/settings.yaml')
        >>> store = ChromaStore(settings=settings)
        >>> records = [
        ...     {
        ...         'id': 'doc1_chunk0',
        ...         'vector': [0.1, 0.2, 0.3],
        ...         'metadata': {'source': 'doc1.pdf'}
        ...     }
        ... ]
        >>> store.upsert(records)
        >>> results = store.query([0.1, 0.2, 0.3], top_k=5)
    """
    
    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """使用配置初始化 ChromaStore。
        
        参数:
            settings: 包含 vector_store 配置的应用设置
            **kwargs: 可选覆盖参数，用于 collection_name 或 persist_directory
        
        异常:
            ImportError: 如果未安装 chromadb 包
            ValueError: 如果缺少必需配置
            RuntimeError: 如果 ChromaDB 客户端初始化失败
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb package is required for ChromaStore. "
                "Install it with: pip install chromadb"
            )
        
        # Extract configuration
        try:
            vector_store_config = settings.vector_store
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vector_store. "
                "Please ensure 'vector_store' section exists in settings.yaml"
            ) from e
        
        # Collection name (allow override)
        self.collection_name = kwargs.get(
            'collection_name',
            getattr(vector_store_config, 'collection_name', 'knowledge_hub')
        )
        
        # Persist directory (allow override)
        persist_dir_str = kwargs.get(
            'persist_directory',
            getattr(vector_store_config, 'persist_directory', './data/db/chroma')
        )
        self.persist_directory = resolve_path(persist_dir_str)
        
        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initializing ChromaStore: collection='{self.collection_name}', "
            f"persist_directory='{self.persist_directory}'"
        )
        
        # 初始化 ChromaDB 客户端，使用持久化存储
        try:
            # 在导入 chromadb 之前禁用遥测
            import os
            os.environ['ANONYMIZED_TELEMETRY'] = 'False'
            os.environ['CHROMADB_TELEMETRY'] = 'False'
            
            # ChromaDB uses different API
            try:
                from chromadb import PersistentClient as Client
            except ImportError:
                from chromadb import Client
            
            # 直接传递 path，不使用 settings 以避免版本兼容性问题
            self.client = Client(
                path=str(self.persist_directory),
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB client at '{self.persist_directory}': {e}"
            ) from e
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get or create collection '{self.collection_name}': {e}"
            ) from e
        
        logger.info(
            f"ChromaStore initialized successfully. "
            f"Collection count: {self.collection.count()}"
        )
    
    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """在 ChromaDB 中插入或更新记录。
        
        参数:
            records: 要 upsert 的记录列表。每条记录必须包含：
                - 'id': 唯一标识符（字符串）
                - 'vector': 嵌入向量（List[float]）
                - 'metadata': 可选的元数据字典
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F）
            **kwargs: 提供者特定的参数（Chroma 未使用）
        
        异常:
            ValueError: 如果记录列表为空或包含无效条目
            RuntimeError: 如果 upsert 操作失败
        """
        # 验证记录
        self.validate_records(records)
        
        # 为 ChromaDB 准备数据
        ids = []
        embeddings = []
        metadatas = []
        documents = []  # ChromaDB 需要 documents 字段
        
        for record in records:
            ids.append(str(record['id']))
            embeddings.append(record['vector'])
            
            # 提取元数据，如果不存在则默认为空字典
            metadata = record.get('metadata', {})
            # 确保所有元数据值都是 JSON 可序列化的
            # ChromaDB 要求字符串、整数、浮点数或布尔值
            sanitized_metadata = self._sanitize_metadata(metadata)
            
            # ChromaDB 要求非空元数据字典
            if not sanitized_metadata:
                sanitized_metadata = {'_placeholder': 'true'}
            
            metadatas.append(sanitized_metadata)
            
            # 文档：如果可用则使用 metadata.text，否则使用 id
            document = metadata.get('text', record['id'])
            documents.append(str(document))
        
        # 执行 upsert（ChromaDB 的 add() 对于相同 ID 是幂等的）
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            logger.debug(f"Successfully upserted {len(records)} records to ChromaDB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to upsert {len(records)} records to ChromaDB: {e}"
            ) from e
    
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """查询 ChromaDB 中相似的向量。
        
        参数:
            vector: 要搜索的查询向量（嵌入）
            top_k: 要返回的最大结果数
            filters: 可选的元数据过滤器（例如 {'source': 'doc1.pdf'}）
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F）
            **kwargs: 提供者特定的参数（Chroma 未使用）
        
        返回:
            匹配记录的列表，按相似度降序排列
            每条记录包含：
                - 'id': 记录标识符
                - 'score': 相似度分数（1.0 = 相同，0.0 = 正交）
                - 'metadata': 关联的元数据
        
        异常:
            ValueError: 如果向量为空或 top_k 无效
            RuntimeError: 如果查询操作失败
        """
        # 验证查询参数
        self.validate_query_vector(vector, top_k)
        
        # 从过滤器构建 ChromaDB where 子句
        where_clause = self._build_where_clause(filters) if filters else None
        
        # Perform query
        try:
            results = self.collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "distances", "documents"]
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query ChromaDB with top_k={top_k}: {e}"
            ) from e
        
        # 将结果转换为标准格式
        # ChromaDB 返回嵌套列表：[[id1, id2, ...]]
        output = []
        
        if results and results['ids'] and results['ids'][0]:
            ids = results['ids'][0]
            distances = results['distances'][0] if 'distances' in results else [0.0] * len(ids)
            metadatas = results['metadatas'][0] if 'metadatas' in results else [{}] * len(ids)
            documents = results['documents'][0] if 'documents' in results else [''] * len(ids)
            
            for i, record_id in enumerate(ids):
                # 将距离转换为相似度分数
                # ChromaDB 返回余弦距离（0=相同，2=相反）
                # 转换为相似度：score = 1 - (distance / 2)
                distance = distances[i]
                score = 1.0 - (distance / 2.0)
                
                output.append({
                    'id': record_id,
                    'score': max(0.0, score),  # 限制在 [0, 1] 范围内
                    'text': documents[i] if documents[i] else '',  # 包含文档中的文本
                    'metadata': metadatas[i] if metadatas[i] else {}
                })
        
        logger.debug(f"Query returned {len(output)} results")
        return output
    
    def delete(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """通过 ID 从 ChromaDB 删除记录。
        
        参数:
            ids: 要删除的记录 ID 列表
            trace: 可选的 TraceContext，用于可观测性
            **kwargs: 提供者特定的参数
        
        异常:
            ValueError: 如果 ids 列表为空
            RuntimeError: 如果删除操作失败
        """
        if not ids:
            raise ValueError("IDs list cannot be empty")
        
        try:
            self.collection.delete(ids=[str(id_) for id_ in ids])
            logger.debug(f"Successfully deleted {len(ids)} records from ChromaDB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete {len(ids)} records from ChromaDB: {e}"
            ) from e
    
    def clear(
        self,
        collection_name: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """清空 ChromaDB 集合中的所有记录。
        
        参数:
            collection_name: 可选的集合名称，用于清空。如果为 None，则清空当前集合
            trace: 可选的 TraceContext，用于可观测性
            **kwargs: 提供者特定的参数
        
        异常:
            RuntimeError: 如果清空操作失败
        """
        try:
            target_collection = collection_name or self.collection_name
            
            # 删除并重新创建集合（在 Chroma 中清空数据最高效的方式）
            self.client.delete_collection(name=target_collection)
            self.collection = self.client.get_or_create_collection(
                name=target_collection,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Successfully cleared collection '{target_collection}'")
        except Exception as e:
            raise RuntimeError(
                f"Failed to clear collection '{collection_name or self.collection_name}': {e}"
            ) from e

    def delete_by_metadata(
        self,
        filter_dict: Dict[str, Any],
        trace: Optional[Any] = None,
    ) -> int:
        """删除匹配元数据过滤器的记录。

        参数:
            filter_dict: 要匹配的元数据键/值对
                （例如 ``{"source_hash": "abc123"}``）
            trace: 可选的 TraceContext，用于可观测性

        返回:
            删除的记录数

        异常:
            ValueError: 如果 *filter_dict* 为空
            RuntimeError: 如果操作失败
        """
        if not filter_dict:
            raise ValueError("filter_dict cannot be empty")

        try:
            where = self._build_where_clause(filter_dict)
            # 先查询匹配的 ID
            results = self.collection.get(where=where, include=[])
            matching_ids = results.get("ids", [])

            if not matching_ids:
                logger.debug(f"delete_by_metadata: no records matched {filter_dict}")
                return 0

            self.collection.delete(ids=matching_ids)
            logger.info(
                f"delete_by_metadata: deleted {len(matching_ids)} records "
                f"matching {filter_dict}"
            )
            return len(matching_ids)
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete by metadata {filter_dict}: {e}"
            ) from e
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """清理元数据以确保 ChromaDB 兼容性。
        
        ChromaDB 要求元数据值为字符串、整数、浮点数或布尔值。
        此方法转换或过滤掉不兼容的类型。
        
        参数:
            metadata: 原始元数据字典
        
        返回:
            清理后的元数据字典
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                # 跳过空值
                continue
            elif isinstance(value, (list, tuple)):
                # 转换为逗号分隔的字符串
                sanitized[key] = ",".join(str(v) for v in value)
            else:
                # 转换为字符串作为后备
                sanitized[key] = str(value)
        
        return sanitized
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """从过滤器构建 ChromaDB where 子句。
        
        将标准过滤器字典转换为 ChromaDB 的查询格式。
        
        参数:
            filters: 标准过滤器字典（例如 {'source': 'doc1.pdf'}）
        
        返回:
            ChromaDB where 子句字典
        
        注意:
            ChromaDB 支持运算符如 $eq、$ne、$gt、$lt、$in 等。
            为简单起见，我们目前仅支持精确相等匹配。
            未来增强：支持复杂过滤器。
        """
        # 简单实现：仅支持精确相等匹配
        # 未来扩展：支持复杂过滤器（例如 {'score': {'$gt': 0.5}}）
        where = {}
        for key, value in filters.items():
            if isinstance(value, dict):
                # 已经是 ChromaDB 运算符格式（例如 {'$eq': 'value'}）
                where[key] = value
            else:
                # 简单相等
                where[key] = value
        
        return where
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取当前集合的统计信息。
        
        返回:
            包含集合统计信息的字典：
                - count: 集合中的记录数
                - name: 集合名称
                - metadata: 集合元数据
        """
        return {
            'count': self.collection.count(),
            'name': self.collection_name,
            'metadata': self.collection.metadata
        }
    
    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """通过 ID 从 ChromaDB 检索记录。
        
        此方法由 SparseRetriever 使用，用于获取
        BM25 搜索匹配的块的文本和元数据（仅返回 ID 和分数）。
        
        参数:
            ids: 要检索的记录 ID 列表
            trace: 可选的 TraceContext，用于可观测性
            **kwargs: 提供者特定的参数（Chroma 未使用）
        
        返回:
            与输入 ids 顺序相同的记录列表
            每条记录包含：
                - 'id': 记录标识符
                - 'text': 存储的文本内容
                - 'metadata': 关联的元数据
            如果 ID 未找到，则该位置返回空字典
        
        异常:
            ValueError: 如果 ids 列表为空
            RuntimeError: 如果检索操作失败
        """
        if not ids:
            raise ValueError("IDs list cannot be empty")
        
        # 确保所有 ID 都是字符串
        str_ids = [str(id_) for id_ in ids]
        
        try:
            # ChromaDB 的 get 方法通过 ID 检索记录
            results = self.collection.get(
                ids=str_ids,
                include=["metadatas", "documents"]
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get records by IDs from ChromaDB: {e}"
            ) from e
        
        # 构建从 ID 到结果的映射，实现 O(1) 查找
        id_to_result: Dict[str, Dict[str, Any]] = {}
        
        if results and results.get('ids'):
            result_ids = results['ids']
            documents = results.get('documents', [None] * len(result_ids))
            metadatas = results.get('metadatas', [{}] * len(result_ids))
            
            for i, record_id in enumerate(result_ids):
                id_to_result[record_id] = {
                    'id': record_id,
                    'text': documents[i] if documents and documents[i] else '',
                    'metadata': metadatas[i] if metadatas and metadatas[i] else {}
                }
        
        # 按照输入 ids 的顺序返回结果
        output = []
        for id_ in str_ids:
            if id_ in id_to_result:
                output.append(id_to_result[id_])
            else:
                # ID 未找到，返回空字典
                output.append({})
        
        logger.debug(f"Retrieved {len([r for r in output if r])} of {len(ids)} records by IDs")
        return output
