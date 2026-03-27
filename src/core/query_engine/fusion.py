"""倒数排名融合（RRF），用于组合多个检索结果。

本模块实现 RRF 融合算法，将来自稠密和稀疏检索器的排名列表
合并为统一的排名。RRF 是一种简单而有效的
doc 排名聚合方法，不需要分数归一化。

参考：
    Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
    "Reciprocal rank fusion outperforms condorcet and individual rank learning methods."
    SIGIR '09.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)


class RRFFusion:
    """用于合并多个排名列表的倒数排名融合（RRF）。

    RRF 使用以下公式合并来自多个来源的排名：
        RRF_score(d) = Σ 1 / (k + rank(d))

    其中：
        - d 是文档（chunk）
        - k 是平滑常数（通常为 60）
        - rank(d) 是文档 d 在排名列表中的排名（从 1 开始）

    关键特性：
    - 确定性：相同输入总是产生相同输出排序
    - 无视分数：仅使用排名位置，不使用原始分数
    - 无需归一化：适用于异构分数尺度
    - 处理缺失文档：仅在单个列表中的文档仍能贡献

    应用的设计原则：
    - 配置驱动：k 参数可配置（默认：60）
    - 类型安全：返回标准化的 RetrievalResult 对象
    - 确定性：通过 chunk_id 进行稳定排序和打破平局
    - 可观察：记录融合过程用于调试

    属性：
        k: RRF 公式的平滑常数（默认：60）
           较高的 k 给予低排名文档更多权重

    示例：
        >>> fusion = RRFFusion(k=60)
        >>> dense_results = [
        ...     RetrievalResult(chunk_id="a", score=0.9, text="...", metadata={}),
        ...     RetrievalResult(chunk_id="b", score=0.8, text="...", metadata={}),
        ... ]
        >>> sparse_results = [
        ...     RetrievalResult(chunk_id="b", score=5.2, text="...", metadata={}),
        ...     RetrievalResult(chunk_id="c", score=4.1, text="...", metadata={}),
        ... ]
        >>> fused = fusion.fuse([dense_results, sparse_results], top_k=5)
    """

    # 原始 RRF 论文推荐的默认平滑常数
    DEFAULT_K = 60

    def __init__(self, k: int = DEFAULT_K) -> None:
        """使用可配置的平滑常数初始化 RRF 融合。

        参数：
            k: RRF 公式的平滑常数（默认：60）
               - 必须是正整数
               - 较高的值降低排名差异的重要性
               - 常用值：60（原始论文）、20、100

        异常：
            ValueError: 如果 k 不是正整数
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        self.k = k
        logger.info(f"RRFFusion 已初始化，k={k}")

    def fuse(
        self,
        ranking_lists: List[List[RetrievalResult]],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """使用倒数排名融合（RRF）合并多个排名列表。

        参数：
            ranking_lists: 排名列表的列表，每个包含 RetrievalResult
                           对象按相关性降序排列
                           通常为 [dense_results, sparse_results]
            top_k: 返回的最大结果数量，为 None 则返回全部
            trace: 可选的 TraceContext 用于可观测性（为 Stage F 预留）

        返回：
            RetrievalResult 对象列表，按融合后的 RRF 分数降序排列
            score 字段包含 RRF 分数，不是原始检索分数
            每个 chunk 的首次出现保留 text 和 metadata

        异常：
            ValueError: ranking_lists 为空

        注意：
            - 在多个列表中出现的文档从所有列表获得贡献
            - 仅在单个列表中出现的文档仍获得 RRF 分数
            - 打破平局：RRF 分数相同时，按 chunk_id 排序以保证稳定性

        示例：
            >>> fusion = RRFFusion(k=60)
            >>> fused = fusion.fuse([dense_results, sparse_results], top_k=10)
            >>> for r in fused:
            ...     print(f"[RRF={r.score:.4f}] {r.chunk_id}")
        """
        if not ranking_lists:
            raise ValueError("ranking_lists 不能为空")

        # 过滤掉空列表
        non_empty_lists = [lst for lst in ranking_lists if lst]

        if not non_empty_lists:
            logger.debug("所有排名列表都为空，返回空结果")
            return []

        logger.debug(
            f"融合 {len(non_empty_lists)} 个排名列表，"
            f"大小为 {[len(lst) for lst in non_empty_lists]}"
        )

        # 步骤 1: 为每个唯一 chunk 计算 RRF 分数
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, RetrievalResult] = {}  # 保留 text/metadata

        for list_idx, ranking_list in enumerate(non_empty_lists):
            for rank, result in enumerate(ranking_list, start=1):
                chunk_id = result.chunk_id

                # 计算 RRF 贡献：1 / (k + rank)
                rrf_contribution = 1.0 / (self.k + rank)

                # 累加分数
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    # 存储首次出现的数据（text、metadata）
                    chunk_data[chunk_id] = result

                rrf_scores[chunk_id] += rrf_contribution

        logger.debug(f"为 {len(rrf_scores)} 个唯一 chunk 计算了 RRF 分数")

        # 步骤 2: 使用 RRF 分数创建融合结果
        fused_results = []
        for chunk_id, rrf_score in rrf_scores.items():
            original = chunk_data[chunk_id]
            fused_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=rrf_score,
                    text=original.text,
                    metadata=original.metadata.copy(),
                )
            )

        # 步骤 3: 按 RRF 分数降序排列，然后按 chunk_id 以保证稳定性
        fused_results.sort(key=lambda r: (-r.score, r.chunk_id))

        # 步骤 4: 如果指定了 top_k 则应用限制
        if top_k is not None and top_k > 0:
            fused_results = fused_results[:top_k]

        logger.debug(
            f"融合完成：{len(fused_results)} 个结果 "
            f"(top_k={top_k if top_k else 'all'})"
        )

        return fused_results

    def fuse_with_weights(
        self,
        ranking_lists: List[List[RetrievalResult]],
        weights: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """使用可选的每个列表权重融合多个排名列表。

        这是 fuse() 的扩展版本，允许对不同的
        排名来源进行加权。例如，为语义查询的稠密检索
        或关键词查询的稀疏检索赋予更高权重。

        参数：
            ranking_lists: 排名列表的列表，每个包含 RetrievalResult 对象
            weights: 每个排名列表的可选权重列表（默认：统一权重）
                     如果提供，长度必须与 ranking_lists 相同
                     权重与 RRF 贡献相乘
            top_k: 返回的最大结果数量，为 None 则返回全部
            trace: 可选的 TraceContext 用于可观测性（为 Stage F 预留）

        返回：
            RetrievalResult 对象列表，按加权 RRF 分数降序排列

        异常：
            ValueError: ranking_lists 为空或 weights 长度不匹配

        示例：
            >>> fusion = RRFFusion(k=60)
            >>> # 给稠密结果 1.5 倍权重
            >>> fused = fusion.fuse_with_weights(
            ...     [dense_results, sparse_results],
            ...     weights=[1.5, 1.0],
            ...     top_k=10
            ... )
        """
        if not ranking_lists:
            raise ValueError("ranking_lists 不能为空")

        # 默认使用统一权重
        if weights is None:
            weights = [1.0] * len(ranking_lists)

        if len(weights) != len(ranking_lists):
            raise ValueError(
                f"weights 长度 ({len(weights)}) 必须匹配 "
                f"ranking_lists 长度 ({len(ranking_lists)})"
            )

        # 验证权重
        for i, w in enumerate(weights):
            if not isinstance(w, (int, float)) or w < 0:
                raise ValueError(f"索引 {i} 处的权重必须为非负数，得到 {w}")

        # 过滤掉空列表（保持权重对齐）
        filtered = [
            (lst, w) for lst, w in zip(ranking_lists, weights) if lst
        ]

        if not filtered:
            logger.debug("所有排名列表都为空，返回空结果")
            return []

        non_empty_lists, filtered_weights = zip(*filtered)

        logger.debug(
            f"融合 {len(non_empty_lists)} 个排名列表，"
            f"weights={list(filtered_weights)}"
        )

        # 计算加权 RRF 分数
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, RetrievalResult] = {}

        for list_idx, (ranking_list, weight) in enumerate(zip(non_empty_lists, filtered_weights)):
            for rank, result in enumerate(ranking_list, start=1):
                chunk_id = result.chunk_id

                # 加权 RRF 贡献
                rrf_contribution = weight * (1.0 / (self.k + rank))

                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    chunk_data[chunk_id] = result

                rrf_scores[chunk_id] += rrf_contribution

        # 创建并排序结果
        fused_results = [
            RetrievalResult(
                chunk_id=chunk_id,
                score=rrf_score,
                text=chunk_data[chunk_id].text,
                metadata=chunk_data[chunk_id].metadata.copy(),
            )
            for chunk_id, rrf_score in rrf_scores.items()
        ]

        fused_results.sort(key=lambda r: (-r.score, r.chunk_id))

        if top_k is not None and top_k > 0:
            fused_results = fused_results[:top_k]

        return fused_results


def rrf_score(rank: int, k: int = RRFFusion.DEFAULT_K) -> float:
    """计算单个排名位置的 RRF 分数贡献。

    这是一个用于计算单个 RRF 贡献的工具函数。

    参数：
        rank: 从 1 开始的排名位置（1 = 最高排名）
        k: 平滑常数（默认：60）

    返回：
        RRF 分数贡献：1 / (k + rank)

    异常：
        ValueError: 如果 rank 不是正整数或 k 不是正数

    示例：
        >>> rrf_score(1, k=60)  # 最高排名文档
        0.01639344262295082
        >>> rrf_score(10, k=60)  # 第 10 名文档
        0.014285714285714285
    """
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"rank must be a positive integer, got {rank}")
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    return 1.0 / (k + rank)
