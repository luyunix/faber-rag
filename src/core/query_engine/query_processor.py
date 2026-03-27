"""用于预处理用户查询的查询处理器。

此模块提供查询预处理功能，包括：
- 使用基于规则的分词进行关键词提取
- 中英文停用词过滤
- 从查询语法中解析过滤器（例如"collection:docs"）
- 查询规范化和清理

设计原则：
- 规则优先：使用简单、确定性规则以确保可靠性
- 语言感知：支持中英文查询
- 可扩展：易于添加同义词扩展或基于 LLM 的处理
- 配置驱动：停用词和模式可通过设置配置
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Set

import jieba

from src.core.types import ProcessedQuery


# 中文默认停用词
CHINESE_STOPWORDS: Set[str] = {
    # 疑问词
    "如何", "怎么", "怎样", "什么", "哪个", "哪些", "为什么", "为何",
    "谁", "多少", "几", "是否", "能否", "可否",
    # 助词
    "的", "地", "得", "了", "着", "过", "吗", "呢", "吧", "啊", "呀",
    # 介词/连词
    "在", "于", "和", "与", "或", "及", "并", "而", "但", "但是",
    "因为", "所以", "如果", "那么", "虽然", "然而",
    # 代词
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那",
    "这个", "那个", "这些", "那些", "这里", "那里",
    # 副词
    "很", "非常", "特别", "更", "最", "都", "也", "还", "又", "再",
    "已", "已经", "正在", "将", "会", "能", "可以", "应该", "必须",
    # 动词(通用)
    "是", "有", "做", "进行", "使用", "通过",
    # 量词
    "个", "种", "类",
    # 标点等
    "？", "。", "！", "，", "、",
}

# 英文默认停用词
ENGLISH_STOPWORDS: Set[str] = {
    # 冠词
    "a", "an", "the",
    # 介词
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "about", "through", "between", "after", "before",
    # 连词
    "and", "or", "but", "if", "then", "because", "while", "although",
    # 代词
    "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "these", "those", "what", "which", "who", "whom", "whose",
    # 助动词
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can",
    # 常用动词
    "get", "use", "make",
    # 疑问词
    "how", "why", "when", "where",
    # 其他
    "not", "no", "yes", "so", "very", "just", "also", "too",
}

# 组合的默认停用词
DEFAULT_STOPWORDS: Set[str] = CHINESE_STOPWORDS | ENGLISH_STOPWORDS

# 过滤器语法模式：key:value
FILTER_PATTERN: Pattern = re.compile(r'(\w+):([^\s]+)')


@dataclass
class QueryProcessorConfig:
    """QueryProcessor 的配置。

    属性:
        stopwords: 要过滤的词集合
        min_keyword_length: 关键词包含的最小长度
        max_keywords: 要提取的最大关键词数
        enable_filter_parsing: 是否从查询中解析过滤器语法
    """
    stopwords: Set[str] = field(default_factory=lambda: DEFAULT_STOPWORDS.copy())
    min_keyword_length: int = 1
    max_keywords: int = 20
    enable_filter_parsing: bool = True


class QueryProcessor:
    """预处理用户查询以进行检索。

    提取关键词、过滤停用词并解析过滤器语法，
    以准备用于稠密和稀疏检索器的查询。

    示例:
        >>> processor = QueryProcessor()
        >>> result = processor.process("如何配置 Azure OpenAI？")
        >>> print(result.keywords)
        ['配置', 'Azure', 'OpenAI']
    """

    def __init__(self, config: Optional[QueryProcessorConfig] = None):
        """初始化 QueryProcessor。

        参数:
            config: 可选配置。未提供时使用默认值。
        """
        self.config = config or QueryProcessorConfig()

    def process(self, query: str) -> ProcessedQuery:
        """将用户查询处理为结构化格式。

        参数:
            query: 原始用户查询字符串

        返回:
            包含提取出的关键词和过滤器的 ProcessedQuery
        """
        if not query or not query.strip():
            return ProcessedQuery(
                original_query=query or "",
                keywords=[],
                filters={}
            )

        # 规范化查询
        normalized = self._normalize(query)

        # 从查询中提取过滤器（例如 "collection:docs"）
        filters, query_without_filters = self._extract_filters(normalized)

        # 分词并提取关键词
        tokens = self._tokenize(query_without_filters)

        # 过滤停用词并应用约束
        keywords = self._filter_keywords(tokens)

        return ProcessedQuery(
            original_query=query,
            keywords=keywords,
            filters=filters
        )

    def _normalize(self, query: str) -> str:
        """规范化查询字符串。

        - 去除首尾空白字符
        - 规范化 Unicode
        - 统一格式

        参数:
            query: 原始查询字符串

        返回:
            规范化后的查询字符串
        """
        # 去除空白字符并规范化
        normalized = " ".join(query.split())
        return normalized

    def _extract_filters(self, query: str) -> tuple[Dict[str, Any], str]:
        """从查询中提取过滤器语法。

        支持如下语法："collection:api-docs keyword1 keyword2"

        参数:
            query: 规范化后的查询字符串

        返回:
            (过滤器字典, 不含过滤器语法的查询) 的元组
        """
        if not self.config.enable_filter_parsing:
            return {}, query

        filters: Dict[str, Any] = {}

        # 查找所有过滤器模式
        matches = FILTER_PATTERN.findall(query)
        for key, value in matches:
            # 支持常用过滤器键
            key_lower = key.lower()
            if key_lower in ("collection", "col", "c"):
                filters["collection"] = value
            elif key_lower in ("type", "doc_type", "t"):
                filters["doc_type"] = value
            elif key_lower in ("source", "src", "s"):
                filters["source_path"] = value
            elif key_lower in ("tag", "tags"):
                # 标签可以是逗号分隔的
                if "tags" not in filters:
                    filters["tags"] = []
                filters["tags"].extend(value.split(","))
            else:
                # 通用过滤器
                filters[key] = value

        # 从查询中移除过滤器模式
        query_without_filters = FILTER_PATTERN.sub("", query).strip()
        query_without_filters = " ".join(query_without_filters.split())

        return filters, query_without_filters

    def _tokenize(self, text: str) -> List[str]:
        """将文本分词为词/术语。

        对中文文本分词使用 jieba，与索引端的分词器（SparseEncoder）一致，
        以便 BM25 匹配能正常工作。
        英文文本由 jieba 原生处理（保持原样）。

        参数:
            text: 要分词的文本

        返回:
            分词列表
        """
        tokens: List[str] = []

        # 使用 jieba 进行分词（支持中文 + 保持英文不变）
        raw_tokens = jieba.lcut(text)

        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue
            # 跳过纯标点符号 / 空白字符
            if re.fullmatch(r'[\s\W]+', token, re.UNICODE):
                continue
            tokens.append(token)

        return tokens

    def _filter_keywords(self, tokens: List[str]) -> List[str]:
        """过滤分词以获得有意义的关键词。

        - 移除停用词
        - 应用最小长度约束
        - 去重同时保持顺序
        - 应用最大数量限制

        参数:
            tokens: 分词列表

        返回:
            过滤后的关键词列表
        """
        seen: Set[str] = set()
        keywords: List[str] = []

        for token in tokens:
            # 用于比较时统一小写
            token_lower = token.lower()

            # 如果已经出现过则跳过（不区分大小写去重）
            if token_lower in seen:
                continue

            # 跳过停用词（检查原始形式和小写形式）
            if token in self.config.stopwords or token_lower in self.config.stopwords:
                continue

            # 如果太短则跳过
            if len(token) < self.config.min_keyword_length:
                continue

            # 添加关键词（保留原始大小写）
            seen.add(token_lower)
            keywords.append(token)

            # 如果数量足够则停止
            if len(keywords) >= self.config.max_keywords:
                break

        return keywords

    def add_stopwords(self, words: Set[str]) -> None:
        """向停用词集合添加词。

        参数:
            words: 要添加的词集合
        """
        self.config.stopwords.update(words)

    def remove_stopwords(self, words: Set[str]) -> None:
        """从停用词集合中移除词。

        参数:
            words: 要移除的词集合
        """
        self.config.stopwords -= words


def create_query_processor(
    stopwords: Optional[Set[str]] = None,
    min_keyword_length: int = 1,
    max_keywords: int = 20,
    enable_filter_parsing: bool = True
) -> QueryProcessor:
    """用于创建 QueryProcessor 的工厂函数。

    参数:
        stopwords: 自定义停用词集合。为 None 时使用默认停用词。
        min_keyword_length: 最小关键词长度
        max_keywords: 要提取的最大关键词数
        enable_filter_parsing: 是否解析过滤器语法

    返回:
        配置好的 QueryProcessor 实例
    """
    config = QueryProcessorConfig(
        stopwords=stopwords if stopwords is not None else DEFAULT_STOPWORDS.copy(),
        min_keyword_length=min_keyword_length,
        max_keywords=max_keywords,
        enable_filter_parsing=enable_filter_parsing
    )
    return QueryProcessor(config)
