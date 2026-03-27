"""查询追踪页面 – 浏览查询追踪历史及阶段瀑布图。

布局：
1. 可选的关键词搜索过滤
2. 追踪列表（按时间倒序，过滤 trace_type=="query"）
3. 详情视图：阶段瀑布图 + Dense vs Sparse 对比 + Rerank delta
4. 每个追踪的 Ragas 评估按钮（LLM-as-Judge 评分）
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService

logger = logging.getLogger(__name__)


def _format_timestamp(iso_timestamp: str) -> str:
    """将 ISO 格式时间戳格式化为易读的中文格式（本地时间）。
    
    Args:
        iso_timestamp: ISO 格式时间戳，如 "2026-03-27T20:53:25.123+00:00"
    
    Returns:
        格式化后的本地时间字符串，如 "2026-03-28 04:53:25"
    """
    if not iso_timestamp or iso_timestamp == "—":
        return "—"
    
    try:
        from datetime import datetime
        
        # 解析 ISO 格式时间戳
        # 处理带有时区的时间戳（如 +00:00）
        if "+" in iso_timestamp or iso_timestamp.endswith("Z"):
            # 替换 Z 为 +00:00
            iso_timestamp = iso_timestamp.replace("Z", "+00:00")
            
            # 解析带时区的时间
            dt = datetime.fromisoformat(iso_timestamp)
            
            # 转换为本地时间
            local_dt = dt.astimezone()
            
            # 格式化为易读的格式
            return local_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # 没有时区信息，直接替换 T 为空格
            return iso_timestamp.replace("T", " ").split(".")[0]
    except Exception as e:
        logger.warning(f"时间格式化失败：{iso_timestamp}, 错误：{e}")
        # 如果解析失败，返回原始值（去掉 T）
        return iso_timestamp.replace("T", " ").split("+")[0].split(".")[0]


def render() -> None:
    """渲染查询追踪页面。"""
    st.header("🔎 查询历史")

    svc = TraceService()
    traces = svc.list_traces(trace_type="query")

    if not traces:
        st.info("暂无查询追踪。请先运行查询！")
        return

    # ── 关键词过滤 ────────────────────────────────────────
    keyword = st.text_input(
        "搜索查询关键词",
        value="",
        key="qt_keyword",
    )
    if keyword.strip():
        kw = keyword.strip().lower()
        traces = [
            t
            for t in traces
            if kw in str(t.get("metadata", {})).lower()
            or kw in str(t.get("stages", [])).lower()
        ]

    st.subheader(f"📋 查询历史 ({len(traces)})")

    for idx, trace in enumerate(traces):
        trace_id = trace.get("trace_id", "unknown")
        started = trace.get("started_at", "—")
        total_ms = trace.get("elapsed_ms")
        total_label = f"{total_ms:.0f} ms" if total_ms is not None else "—"
        meta = trace.get("metadata", {})
        query_text = meta.get("query", "")
        source = meta.get("source", "unknown")

        # ── 展开器标题：显示查询文本 ───────────────────
        query_preview = (
            query_text[:40] + "…" if len(query_text) > 40 else query_text
        ) if query_text else "—"
        started_formatted = _format_timestamp(started)
        expander_title = (
            f"🔍 \"{query_preview}\"  ·  {total_label}  ·  {started_formatted}"
        )

        with st.expander(expander_title, expanded=(idx == 0)):
            # ── 1. 查询概览 ─────────────────────────────
            st.markdown("#### 💬 查询")
            col_q, col_meta = st.columns([3, 1])
            with col_q:
                st.markdown(f"> {query_text}")
            with col_meta:
                source_emoji = "🤖" if source == "mcp" else "📡"
                st.markdown(f"**来源:** {source_emoji} `{source}`")
                st.markdown(f"**Top-K:** `{meta.get('top_k', '—')}`")
                st.markdown(f"**Collection:** `{meta.get('collection', '—')}`")

            st.divider()

            # ── 2. 概览指标 ───────────────────────────
            timings = svc.get_stage_timings(trace)
            stages_by_name = {t["stage_name"]: t for t in timings}

            dense_d = (stages_by_name.get("dense_retrieval", {}).get("data") or {})
            sparse_d = (stages_by_name.get("sparse_retrieval", {}).get("data") or {})
            fusion_d = (stages_by_name.get("fusion", {}).get("data") or {})
            rerank_d = (stages_by_name.get("rerank", {}).get("data") or {})

            dense_count = dense_d.get("result_count", 0)
            sparse_count = sparse_d.get("result_count", 0)
            fusion_count = fusion_d.get("result_count", 0)
            rerank_count = rerank_d.get("output_count", 0)

            rc1, rc2, rc3, rc4, rc5 = st.columns(5)
            with rc1:
                st.metric("Dense Hits", dense_count)
            with rc2:
                st.metric("Sparse Hits", sparse_count)
            with rc3:
                st.metric("Fused", fusion_count or (dense_count + sparse_count))
            with rc4:
                st.metric("After Rerank", rerank_count if rerank_d else "—")
            with rc5:
                st.metric("Total Time", total_label)

            # ── 诊断提示 ──────────────────────────────
            _render_diagnostics(
                stages_by_name, dense_d, sparse_d, fusion_d, rerank_d,
                dense_count, sparse_count,
            )

            st.divider()

            # ── 3. 阶段计时瀑布图 ───────────────────
            main_stage_names = ("query_processing", "dense_retrieval", "sparse_retrieval", "fusion", "rerank")
            main_timings = [t for t in timings if t["stage_name"] in main_stage_names]
            if main_timings:
                st.markdown("#### ⏱️ 阶段计时")
                chart_data = {t["stage_name"]: t["elapsed_ms"] for t in main_timings}
                st.bar_chart(chart_data, horizontal=True)
                st.table([
                    {
                        "阶段": t["stage_name"],
                        "耗时 (ms)": round(t["elapsed_ms"], 2),
                    }
                    for t in main_timings
                ])

            st.divider()

            # ── 4. 阶段详情标签页 ───────────────────
            st.markdown("#### 🔍 阶段详情")

            tab_defs = []
            if "query_processing" in stages_by_name:
                tab_defs.append(("🔤 Query Processing", "query_processing"))
            if "dense_retrieval" in stages_by_name:
                tab_defs.append(("🟦 Dense Retrieval", "dense_retrieval"))
            if "sparse_retrieval" in stages_by_name:
                tab_defs.append(("🟨 Sparse Retrieval", "sparse_retrieval"))
            if "fusion" in stages_by_name:
                tab_defs.append(("🟩 Fusion (RRF)", "fusion"))
            if "rerank" in stages_by_name:
                tab_defs.append(("🟪 Rerank", "rerank"))

            if tab_defs:
                tabs = st.tabs([label for label, _ in tab_defs])
                for tab, (label, key) in zip(tabs, tab_defs):
                    with tab:
                        stage = stages_by_name[key]
                        data = stage.get("data", {})
                        elapsed = stage.get("elapsed_ms")
                        if elapsed is not None:
                            st.caption(f"⏱️ {elapsed:.1f} ms")

                        if key == "query_processing":
                            _render_query_processing_stage(data)
                        elif key == "dense_retrieval":
                            _render_retrieval_stage(data, "Dense", trace_idx=idx)
                        elif key == "sparse_retrieval":
                            _render_retrieval_stage(data, "Sparse", trace_idx=idx)
                        elif key == "fusion":
                            _render_fusion_stage(data, trace_idx=idx)
                        elif key == "rerank":
                            _render_rerank_stage(data, trace_idx=idx)
                st.info("暂无阶段详情。")

            # ── 5. Ragas 评估按钮 ───────────────────
            _render_evaluate_button(trace, idx)


def _render_diagnostics(
    stages_by_name: Dict[str, Any],
    dense_d: Dict[str, Any],
    sparse_d: Dict[str, Any],
    fusion_d: Dict[str, Any],
    rerank_d: Dict[str, Any],
    dense_count: int,
    sparse_count: int,
) -> None:
    """渲染关于缺失或错误流水线阶段的诊断提示。"""
    hints: list = []
    
    # 稠密错误
    dense_err = dense_d.get("error", "")
    if dense_err:
        hints.append(("error", f"**Dense Retrieval 失败:** {dense_err}"))
    elif dense_count == 0 and "dense_retrieval" in stages_by_name:
        hints.append(("warning", "Dense Retrieval 返回 **0 结果**。检查集合是否有索引数据。"))

    # 稀疏错误/为空
    sparse_err = sparse_d.get("error", "")
    if sparse_err:
        hints.append(("error", f"**Sparse Retrieval 失败:** {sparse_err}"))
    elif sparse_count == 0 and "sparse_retrieval" in stages_by_name:
        hints.append((
            "warning",
            "Sparse (BM25) Retrieval 返回 **0 结果**。"
            "BM25 index may be empty or not yet built for this collection.",
        ))

    # 融合缺失
    if "fusion" not in stages_by_name:
        if dense_count > 0 and sparse_count > 0:
            hints.append(("info", "即使两个检索器都返回了结果，Fusion 阶段也未被记录。"))
        elif dense_count == 0 or sparse_count == 0:
            only_source = "Dense" if dense_count > 0 else ("Sparse" if sparse_count > 0 else "两者都")
            hints.append((
                "info",
                f"**Fusion (RRF) 跳过:** 只有 {only_source} 检索返回了结果。"
                "Fusion 需要 Dense 和 Sparse 两个结果才能合并。",
            ))

    # Rerank 缺失
    if "rerank" not in stages_by_name:
        if dense_count > 0 or sparse_count > 0:
            hints.append((
                "info",
                "**Rerank 跳过:** reranker 未启用或未配置。"
                "在 settings.yaml 中启用 `reranker` 以应用基于 LLM 的重排序。",
            ))

    # 所有结果为空
    if dense_count == 0 and sparse_count == 0:
        hints.append((
            "warning",
            "**未找到结果。** 集合可能为空，或者查询与任何索引内容不匹配。"
            "请先尝试处理文档。",
        ))

    # 渲染提示
    for level, msg in hints:
        if level == "error":
            st.error(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.info(msg)


def _render_evaluate_button(trace: Dict[str, Any], idx: int) -> None:
    """为单个查询追踪渲染 Ragas 评估按钮。

    重新运行存储的查询并使用 RagasEvaluator（LLM-as-Judge）进行评估。
    仅在追踪元数据中有查询文本时才有效。
    """
    meta = trace.get("metadata", {})
    query = meta.get("query", "")
    if not query:
        return

    st.divider()
    st.markdown("#### 📏 Ragas 评估")
    st.caption(
        "RAGAS 需要 **Query + Retrieved Context + Answer** 三要素来评估。"
        "日志中仅包含 Query 和检索到的上下文，请在下方输入实际回答后再运行评估。"
    )

    # 答案输入框 — 用户提供实际的生成的答案
    answer_key = f"eval_answer_{idx}"
    user_answer = st.text_area(
        "✏️ 生成的回答",
        value=st.session_state.get(answer_key, ""),
        height=120,
        key=answer_key,
        placeholder="请输入系统生成的回答，或粘贴 LLM 的实际输出…",
        help=(
            "Ragas 使用 LLM-as-Judge 评估回答质量。"
            "faithfulness 衡量回答是否忠于检索到的上下文，"
            "answer_relevancy 衡量回答与问题的相关性。"
            "如果不填写回答，将无法获得有意义的评估结果。"
        ),
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        clicked = st.button(
            "📏 Ragas 评估",
            key=f"eval_trace_{idx}",
            help="重新运行此查询并使用 Ragas 评分（LLM-as-Judge）",
            disabled=not user_answer.strip(),
        )
    with col_info:
        if not user_answer.strip():
            st.warning("⚠️ 请先在上方输入回答内容，再运行 Ragas 评估。")
        else:
            st.caption(
                "使用 Ragas 评估 faithfulness（忠实度）、answer relevancy（回答相关性）"
                "和 context precision（上下文精度）。需要调用 LLM — 可能需要几秒钟。"
            )

    # 从 session state 显示之前的结果
    result_key = f"eval_result_{idx}"
    if result_key in st.session_state and not clicked:
        _display_eval_metrics(st.session_state[result_key])

    if clicked:
        with st.spinner("运行 Ragas 评估…"):
            result = _evaluate_single_trace(query, meta, user_answer=user_answer.strip())
        st.session_state[result_key] = result
        _display_eval_metrics(result)


def _evaluate_single_trace(
    query: str,
    meta: Dict[str, Any],
    user_answer: Optional[str] = None,
) -> Dict[str, Any]:
    """重新运行检索并使用 Ragas 评估单个查询。

    返回包含 'metrics'（评分字典）或 'error'（字符串）的字典。
    """
    try:
        from dataclasses import replace as dc_replace

        from src.core.settings import load_settings, EvaluationSettings
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory

        settings = load_settings()

        # 覆盖评估设置以强制使用 Ragas（frozen dataclass，使用 replace）
        ragas_eval = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=["faithfulness", "answer_relevancy", "context_precision"],
        )
        settings = dc_replace(settings, evaluation=ragas_eval)
        evaluator = EvaluatorFactory.create(settings)

        # 重新运行检索
        collection = meta.get("collection", "default")
        top_k = meta.get("top_k", 10)
        chunks = _retrieve_chunks(settings, query, top_k, collection)

        if not chunks:
            return {"error": "未检索到片段 — 数据是否已索引？"}

        # 使用用户提供的答案；仅在万不得已时回退到片段拼接
        # （产生不太有意义的 RAGAS 评分）。
        if user_answer:
            answer = user_answer
        else:
            _MAX_ANSWER_CHARS = 1500
            texts = []
            for c in chunks:
                if hasattr(c, "text"):
                    texts.append(c.text)
                elif isinstance(c, dict):
                    texts.append(c.get("text", str(c)))
                else:
                    texts.append(str(c))
            answer = " ".join(texts[:3])
            if len(answer) > _MAX_ANSWER_CHARS:
                answer = answer[:_MAX_ANSWER_CHARS]

        # 评估
        metrics = evaluator.evaluate(
            query=query,
            retrieved_chunks=chunks,
            generated_answer=answer,
        )
        return {"metrics": metrics, "answer_used": answer}

    except ImportError as exc:
        return {"error": f"Ragas 未安装：{exc}"}
    except Exception as exc:
        logger.exception("Ragas evaluation failed")
        return {"error": str(exc)}


def _retrieve_chunks(
    settings: Any,
    query: str,
    top_k: int,
    collection: str,
) -> list:
    """重新运行 HybridSearch + Rerank 以检索用于评估的片段。"""
    try:
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.core.query_engine.reranker import create_core_reranker
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        vector_store = VectorStoreFactory.create(
            settings, collection_name=collection,
        )
        embedding_client = EmbeddingFactory.create(settings)
        dense_retriever = create_dense_retriever(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )
        from src.core.settings import resolve_path
        bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
        sparse_retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection
        query_processor = QueryProcessor()
        hybrid_search = create_hybrid_search(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )

        # 如果启用重排序则检索更多候选结果
        reranker = create_core_reranker(settings=settings)
        initial_top_k = top_k * 2 if reranker.is_enabled else top_k

        results = hybrid_search.search(query=query, top_k=initial_top_k)
        results = results if isinstance(results, list) else results.results

        # 如果启用则应用重排序
        if reranker.is_enabled and results:
            rerank_result = reranker.rerank(query=query, results=results, top_k=top_k)
            results = rerank_result.results

        return results
    except Exception as exc:
        logger.warning("Retrieval for evaluation failed: %s", exc)
        return []


def _display_eval_metrics(result: Dict[str, Any]) -> None:
    """Display evaluation result (metrics or error)."""
    if "error" in result:
        st.error(f"❌ Evaluation failed: {result['error']}")
        return

    metrics = result.get("metrics", {})
    if not metrics:
        st.warning("No metrics returned.")
        return

    st.markdown("**📏 Ragas Scores**")
    cols = st.columns(min(len(metrics), 4))
    for i, (name, value) in enumerate(sorted(metrics.items())):
        with cols[i % len(cols)]:
            st.metric(
                label=name.replace("_", " ").title(),
                value=f"{value:.4f}",
            )


def _extract_pipeline_chunks(
    timings: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract chunk lists from each pipeline stage."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    for stage in timings:
        name = stage.get("stage_name", "")
        data = stage.get("data") or {}
        chunks = data.get("chunks")
        if chunks and isinstance(chunks, list):
            result[name] = chunks
    final = meta.get("final_results") or meta.get("results")
    if final and isinstance(final, list):
        result["final"] = final
    return result


# ═══════════════════════════════════════════════════════════════
# 每个阶段的渲染器
# ═══════════════════════════════════════════════════════════════

def _render_query_processing_stage(data: Dict[str, Any]) -> None:
    """Render Query Processing stage: original query → keywords."""
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original Query**")
        st.info(data.get("original_query", "—"))
    with c2:
        st.markdown("**Method**")
        st.code(data.get("method", "—"))

    keywords = data.get("keywords", [])
    if keywords:
        st.markdown("**Extracted Keywords**")
        st.markdown(" · ".join(f"`{kw}`" for kw in keywords))
    else:
        st.warning("No keywords extracted.")


def _render_retrieval_stage(data: Dict[str, Any], label: str, *, trace_idx: int = 0) -> None:
    """Render Dense or Sparse retrieval stage: method, counts, chunk list."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Method", data.get("method", "—"))
    with c2:
        extra = data.get("provider", data.get("keyword_count", "—"))
        extra_label = "Provider" if "provider" in data else "Keywords"
        st.metric(extra_label, extra)
    with c3:
        st.metric("Results", data.get("result_count", 0))

    st.markdown(f"**Top-K requested:** `{data.get('top_k', '—')}`")

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix=f"{label.lower().replace(' ', '_')}_chunk_{trace_idx}")
    else:
        st.info(f"No {label.lower()} results returned.")


def _render_fusion_stage(data: Dict[str, Any], *, trace_idx: int = 0) -> None:
    """Render Fusion (RRF) stage: input lists, fused result count, chunk list."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Method", data.get("method", "rrf"))
    with c2:
        st.metric("Input Lists", data.get("input_lists", "—"))
    with c3:
        st.metric("Fused Results", data.get("result_count", 0))

    st.markdown(f"**Top-K:** `{data.get('top_k', '—')}`")

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix=f"fusion_chunk_{trace_idx}")
    else:
        st.info("No fusion results.")


def _render_rerank_stage(data: Dict[str, Any], *, trace_idx: int = 0) -> None:
    """Render Rerank stage: method, input/output counts, reranked chunk list."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Method", data.get("method", "—"))
    with c2:
        st.metric("Provider", data.get("provider", "—"))
    with c3:
        st.metric("Input", data.get("input_count", "—"))
    with c4:
        st.metric("Output", data.get("output_count", "—"))

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix=f"rerank_chunk_{trace_idx}")
    else:
        st.info("No reranked results.")


def _render_chunk_list(chunks: List[Dict[str, Any]], prefix: str = "chunk") -> None:
    """Render a list of chunk dicts as a compact, readable table with expandable text."""
    for ci, chunk in enumerate(chunks):
        score = chunk.get("score", 0)
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", "")
        source = chunk.get("source", "")
        title = chunk.get("title", "")

        # 颜色编码的分数指示器
        if score >= 0.8:
            score_bar = "🟢"
        elif score >= 0.5:
            score_bar = "🟡"
        else:
            score_bar = "🔴"

        header = f"{score_bar} **#{ci + 1}** — Score: `{score:.4f}`"
        if title:
            header += f" — {title}"

        with st.expander(header, expanded=False):
            cols = st.columns([2, 3])
            with cols[0]:
                st.caption(f"Chunk ID: `{chunk_id}`")
            with cols[1]:
                if source:
                    st.caption(f"Source: `{source}`")
            # 显示片段文本（可滚动）
            if text:
                st.text_area(
                    f"{prefix}_{ci}",
                    value=text,
                    height=max(80, min(len(text) // 2, 400)),
                    disabled=True,
                    label_visibility="collapsed",
                )
            else:
                st.caption("_No text available_")


def _find_stage(timings, name):
    """Find a stage dict by name, or None."""
    for t in timings:
        if t["stage_name"] == name:
            return t
    return None
