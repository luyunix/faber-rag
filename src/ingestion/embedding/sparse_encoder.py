"""用于从文本分块生成 BM25 词项统计的稀疏编码器。

本模块实现摄取 Pipeline 的稀疏编码器组件，
负责提取 BM25 索引所需的词项统计信息。

设计原则：
- 无状态处理：encode() 调用之间没有内部状态
- 可观测：接受 TraceContext 用于未来可观测性集成
- 确定性：相同输入产生相同词项统计
- 清晰契约：为下游 BM25Indexer 定义明确的输出结构
"""

from typing import List, Dict, Optional, Any
from collections import Counter
import re
import logging

import jieba

from src.core.types import Chunk

logger = logging.getLogger(__name__)


class SparseEncoder:
    """将文本分块编码为 BM25 词项统计。
    
    此编码器负责准备 BM25 索引所需的词级统计信息。
    实际的索引构建由 BM25Indexer (C12) 处理。
    
    输出结构：
        对于每个分块，生成：
        {
            "chunk_id": str,
            "term_frequencies": Dict[str, int],  # term -> 在此分块中的出现次数
            "doc_length": int,                    # 分块中的词项数
            "unique_terms": int                   # 分块中的词汇表大小
        }
    
    设计原则：
    - 分词：简单的空白字符 + 小写（可以在以后增强）
    - 停用词：默认无（可以在以后的版本中添加）
    - 确定性：相同的分块文本总是产生相同的统计信息
    
    示例：
        >>> from src.core.types import Chunk
        >>> encoder = SparseEncoder()
        >>> 
        >>> chunks = [Chunk(id="1", text="Hello world hello", metadata={})]
        >>> stats = encoder.encode(chunks)
        >>> stats[0]["term_frequencies"]["hello"]  # 2
        >>> stats[0]["doc_length"]  # 3
    """
    
    def __init__(
        self,
        min_term_length: int = 2,
        lowercase: bool = True,
    ):
        """初始化 SparseEncoder。
        
        Args:
            min_term_length: 词项的最小字符长度（默认：2）
            lowercase: 是否将词项转换为小写（默认：True）
        
        Raises:
            ValueError: 如果 min_term_length < 1
        """
        if min_term_length < 1:
            raise ValueError(f"min_term_length must be >= 1, got {min_term_length}")
        
        self.min_term_length = min_term_length
        self.lowercase = lowercase
    
    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """将分块编码为 BM25 词项统计。
        
        对于每个分块，提取：
        - 词频（term -> count）
        - 文档长度（总词项数）
        - 唯一词项数
        
        Args:
            chunks: 要编码的 Chunk 对象列表
            trace: 可选的 TraceContext 用于可观测性（保留用于 Stage F）
        
        Returns:
            统计字典列表（每个分块一个，顺序相同）。
            每个 dict 包含：chunk_id, term_frequencies, doc_length, unique_terms
        
        Raises:
            ValueError: 如果分块列表为空
            ValueError: 如果任何分块的文本为空
        
        Example:
            >>> chunks = [
            ...     Chunk(id="1", text="machine learning", metadata={}),
            ...     Chunk(id="2", text="deep learning networks", metadata={})
            ... ]
            >>> stats = encoder.encode(chunks)
            >>> len(stats) == len(chunks)  # True
            >>> stats[0]["term_frequencies"]["machine"]  # 1
            >>> stats[1]["doc_length"]  # 3
        """
        if not chunks:
            logger.warning("SparseEncoder: Cannot encode empty chunks list")
            raise ValueError("Cannot encode empty chunks list")
        
        logger.info(f"SparseEncoder: Starting to encode {len(chunks)} chunks")
        
        results = []
        
        for i, chunk in enumerate(chunks):
            # 调试日志：记录 chunk 的详细信息
            logger.info(f"SparseEncoder: Processing chunk {i} - id={chunk.id}, type={type(chunk)}, text_len={len(chunk.text) if chunk.text else 0}")
            
            # 验证分块文本
            if not chunk.text or not chunk.text.strip():
                logger.error(f"SparseEncoder: Chunk at index {i} (id={chunk.id}) has empty text")
                raise ValueError(
                    f"Chunk at index {i} (id={chunk.id}) has empty or whitespace-only text"
                )
            
            # 分词并计数
            chunk_id_str = str(chunk.id)[:20] if chunk.id else "N/A"
            logger.debug(f"SparseEncoder: Tokenizing chunk {i} (id={chunk_id_str}..., length={len(chunk.text)})")
            terms = self._tokenize(chunk.text)
            logger.debug(f"SparseEncoder: Extracted {len(terms)} terms from chunk {i}")
            term_frequencies = Counter(terms)
            
            # 构建 statistics dict
            stat_dict = {
                "chunk_id": chunk.id,
                "term_frequencies": dict(term_frequencies),  # Convert Counter to dict
                "doc_length": len(terms),
                "unique_terms": len(term_frequencies),
            }
            
            logger.debug(f"SparseEncoder: Chunk {i} - doc_length={stat_dict['doc_length']}, unique_terms={stat_dict['unique_terms']}")
            results.append(stat_dict)
        
        logger.info(f"SparseEncoder: Successfully encoded {len(chunks)} chunks")
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """将文本分词为词项。
        
        使用 jieba 进行中文分词，使用正则表达式处理英文。
        这确保了与查询端（QueryProcessor）的一致分词，
        这是 BM25 匹配所必需的。
        
        Args:
            text: 要分词的输入文本
        
        Returns:
            有效词项列表
        """
        tokens: List[str] = []

        # 使用 jieba 分词（同时处理中文和英文）
        raw_tokens = jieba.lcut(text)

        # 清理词项：只保留字母数字和中文字符
        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue
            # 跳过纯标点符号/空白字符
            if re.fullmatch(r'[\s\W]+', token, re.UNICODE):
                continue
            tokens.append(token)
        
        # 如果配置则应用小写
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        
        # 按最小长度过滤
        terms = [t for t in tokens if len(t) >= self.min_term_length]
        
        return terms
    
    def get_corpus_stats(
        self,
        encoded_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """从编码的分块计算语料库级别的统计信息。
        
        BM25Indexer 的工具方法，用于计算：
        - 平均文档长度
        - 文档频率（每个词项出现在多少文档中）
        - 文档总数
        
        Args:
            encoded_chunks: 来自 encode() 的统计字典列表
        
        Returns:
            包含语料库级别统计信息的字典：
            {
                "num_docs": int,
                "avg_doc_length": float,
                "document_frequency": Dict[str, int]  # term -> # docs containing it
            }
        """
        if not encoded_chunks:
            return {
                "num_docs": 0,
                "avg_doc_length": 0.0,
                "document_frequency": {}
            }
        
        num_docs = len(encoded_chunks)
        total_length = sum(chunk["doc_length"] for chunk in encoded_chunks)
        avg_doc_length = total_length / num_docs if num_docs > 0 else 0.0
        
        # 计算每个词的文档频率（DF）
        doc_freq: Dict[str, int] = {}
        for chunk_stats in encoded_chunks:
            # 此分块中的每个唯一词项对 DF 贡献 1
            for term in chunk_stats["term_frequencies"].keys():
                doc_freq[term] = doc_freq.get(term, 0) + 1
        
        return {
            "num_docs": num_docs,
            "avg_doc_length": avg_doc_length,
            "document_frequency": doc_freq,
        }
