"""VectorStore 提供者的抽象基类。

本模块定义了 VectorStore 提供者的可插拔接口，
通过配置驱动的实例化，实现不同后端（Chroma、Qdrant、Milvus 等）
之间的无缝切换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseVectorStore(ABC):
    """VectorStore 提供者的抽象基类。
    
    所有 VectorStore 实现必须继承此类并实现
    upsert() 和 query() 方法。这确保了不同提供者
    （Chroma、Qdrant、Milvus 等）之间的一致性接口。
    
    应用的设计原则：
    - 可插拔：子类可以互换而无需更改上游代码
    - 可观测：接受可选的 TraceContext 用于可观测性集成
    - 配置驱动：实例通过工厂基于设置创建
    - 幂等性：upsert() 操作应该可以安全重复执行
    """
    
    @abstractmethod
    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """在向量存储中插入或更新记录。
        
        参数:
            records: 要 upsert 的记录列表。每条记录是包含以下字段的字典：
                - 'id': 唯一标识符（字符串）
                - 'vector': 嵌入向量（List[float]）
                - 'metadata': 可选的元数据字典（source、chunk_index 等）
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F）
            **kwargs: 提供者特定的参数
        
        异常:
            ValueError: 如果记录列表为空或包含无效条目
            RuntimeError: 如果向量存储操作失败
        
        示例:
            >>> records = [
            ...     {
            ...         'id': 'doc1_chunk0',
            ...         'vector': [0.1, 0.2, ..., 0.5],
            ...         'metadata': {'source': 'doc1.pdf', 'page': 1}
            ...     }
            ... ]
            >>> vector_store.upsert(records)
        
        注意:
            - 此操作应该是幂等的：多次 upsert 相同记录
              应该产生相同的最终状态
            - 实现应该高效地处理批量操作
        """
        pass
    
    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """查询向量存储中相似的向量。
        
        参数:
            vector: 要搜索的查询向量（嵌入）
            top_k: 要返回的最大结果数
            filters: 可选的元数据过滤器（例如 {'source': 'doc1.pdf'}）
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F）
            **kwargs: 提供者特定的参数
        
        返回:
            匹配记录的列表，按相似度降序排列
            每条记录是包含以下字段的字典：
                - 'id': 记录标识符
                - 'score': 相似度分数（越高 = 越相似）
                - 'metadata': 关联的元数据
                - 'vector': 可选，存储的向量（取决于提供者）
        
        异常:
            ValueError: 如果向量为空或 top_k 无效
            RuntimeError: 如果向量存储查询失败
        
        示例:
            >>> query_vector = [0.1, 0.2, ..., 0.5]
            >>> results = vector_store.query(query_vector, top_k=5)
            >>> for result in results:
            ...     print(f"ID: {result['id']}, Score: {result['score']}")
        """
        pass
    
    def validate_records(self, records: List[Dict[str, Any]]) -> None:
        """在 upsert 之前验证记录。
        
        参数:
            records: 要验证的记录列表
        
        异常:
            ValueError: 如果记录列表为空或包含无效条目
        """
        if not records:
            raise ValueError("Records list cannot be empty")
        
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Record at index {i} is not a dict (type: {type(record).__name__})"
                )
            
            # 验证必需字段
            if 'id' not in record:
                raise ValueError(f"Record at index {i} is missing required field: 'id'")
            if 'vector' not in record:
                raise ValueError(f"Record at index {i} is missing required field: 'vector'")
            
            # 验证向量格式
            vector = record['vector']
            if not isinstance(vector, (list, tuple)):
                raise ValueError(
                    f"Record at index {i} has invalid vector type: {type(vector).__name__}. "
                    "Expected list or tuple of floats."
                )
            
            if not vector:
                raise ValueError(f"Record at index {i} has empty vector")
    
    def validate_query_vector(self, vector: List[float], top_k: int) -> None:
        """验证查询参数。
        
        参数:
            vector: 要验证的查询向量
            top_k: 要验证的结果数
        
        异常:
            ValueError: 如果参数无效
        """
        if not isinstance(vector, (list, tuple)):
            raise ValueError(
                f"Query vector must be a list or tuple, got {type(vector).__name__}"
            )
        
        if not vector:
            raise ValueError("Query vector cannot be empty")
        
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
    
    def delete(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """通过 ID 从向量存储删除记录。
        
        参数:
            ids: 要删除的记录 ID 列表
            trace: 可选的 TraceContext，用于可观测性
            **kwargs: 提供者特定的参数
        
        异常:
            ValueError: 如果 ids 列表为空
            RuntimeError: 如果删除操作失败
            NotImplementedError: 如果提供者不支持删除
        
        注意:
            这是一个可选操作。不支持删除的提供者
            应该抛出 NotImplementedError 并附带清晰的消息
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement delete() method. "
            "This operation is optional and provider-dependent."
        )
    
    def clear(
        self,
        collection_name: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """清空向量存储或特定集合中的所有记录。
        
        参数:
            collection_name: 可选的集合名称，用于清空。如果为 None，则清空默认集合
            trace: 可选的 TraceContext，用于可观测性
            **kwargs: 提供者特定的参数
        
        异常:
            RuntimeError: 如果清空操作失败
            NotImplementedError: 如果提供者不支持清空
        
        注意:
            这主要用于测试和开发。生产环境中请谨慎使用
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement clear() method. "
            "This operation is optional and primarily for testing."
        )
    
    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """通过 ID 检索记录。
        
        此方法由 SparseRetriever 使用，用于获取
        BM25 搜索匹配的块的文本和元数据（仅返回 ID 和分数）。
        
        参数:
            ids: 要检索的记录 ID 列表
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F）
            **kwargs: 提供者特定的参数
        
        返回:
            与输入 ids 顺序相同的记录列表
            每条记录是包含以下字段的字典：
                - 'id': 记录标识符
                - 'text': 存储的文本内容
                - 'metadata': 关联的元数据
            如果 ID 未找到，则该位置返回空字典
        
        异常:
            ValueError: 如果 ids 列表为空
            RuntimeError: 如果检索操作失败
            NotImplementedError: 如果提供者不支持此操作
        
        示例:
            >>> ids = ["chunk_001", "chunk_002", "chunk_003"]
            >>> records = vector_store.get_by_ids(ids)
            >>> for record in records:
            ...     print(f"ID: {record['id']}, Text: {record['text'][:50]}...")
        
        注意:
            此操作对于混合搜索至关重要，其中 BM25 返回
            需要从向量存储中丰富文本和元数据的块 ID
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_by_ids() method. "
            "This operation is required for SparseRetriever support."
        )
