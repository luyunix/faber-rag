"""评估面板页面 – 运行评估并查看指标。

布局：
1. 配置区域：选择评估后端、golden test set、top_k
2. 运行按钮及进度指示器
3. 结果区域：汇总指标、每个查询的详情表
4. 可选：历史评估结果对比
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

# 默认 golden test set 位置
DEFAULT_GOLDEN_SET = Path("tests/fixtures/golden_test_set.json")
# 评估结果历史文件
EVAL_HISTORY_PATH = Path("logs/eval_history.jsonl")


def render() -> None:
    """渲染评估面板页面。"""
    st.header("📏 评估面板")
    st.markdown(
        "基于 **golden test set** 运行评估，以衡量检索和生成质量。"
        "结果包括每个查询的详情和汇总指标。"
    )

    # ── 配置区域 ─────────────────────────────────────
    st.subheader("⚙️ 配置")

    col1, col2, col3 = st.columns(3)

    with col1:
        backend = st.selectbox(
            "评估后端",
            options=["custom", "ragas", "composite"],
            index=0,
            key="eval_backend",
            help="选择使用哪个评估后端。",
        )

    # 根据选择的后端显示信息/警告
    if backend in ("custom", "composite"):
        st.info(
            "ℹ️ **Custom Evaluator** 尚未完成数据集准备，当前仅为预留接口。"
            "Custom Evaluator 需要在 Golden Test Set 中填写 `expected_chunk_ids` "
            "作为 ground truth 才能计算 hit_rate / MRR 指标。"
            "目前建议使用 **ragas** 后端进行评估。",
            icon="🚧",
        )

    with col2:
        top_k = st.number_input(
            "Top-K（返回数量）",
            min_value=1,
            max_value=50,
            value=10,
            key="eval_top_k",
            help="每个查询检索的片段数量。",
        )

    with col3:
        collection = st.text_input(
            "集合（可选）",
            value="",
            key="eval_collection",
            help="限制检索到特定的集合。",
        )

    # Golden test set 文件选择
    golden_path_str = st.text_input(
        "Golden Test Set 路径",
        value=str(DEFAULT_GOLDEN_SET),
        key="eval_golden_path",
        help="golden_test_set.json 文件的路径。",
    )
    golden_path = Path(golden_path_str)

    # 验证 golden set 是否存在
    if not golden_path.exists():
        st.warning(
            f"⚠️ **未找到 Golden test set:** `{golden_path}`。"
            "创建一个包含测试查询和预期结果的 JSON 文件。"
            "参考 `tests/fixtures/golden_test_set.json` 了解格式。"
        )

    # ── 答案输入区域（针对 Ragas）───────────────────────────
    user_answers: Dict[int, str] = {}
    if backend == "ragas" and golden_path.exists():
        st.divider()
        st.subheader("✏️ Provide Answers (回答输入)")
        st.caption(
            "**RAGAS 需要 Query + Context + Answer 三要素来评估。**"
            "日志中仅包含 Query 和检索到的上下文（Context），"
            "请为每个测试用例填写实际的系统回答（Answer），"
            "以便获得有意义的 faithfulness 和 answer_relevancy 评分。"
        )
        try:
            _test_cases = _load_golden_queries(golden_path)
            for tc_idx, tc in enumerate(_test_cases):
                ans_key = f"eval_answer_tc_{tc_idx}"
                default_val = tc.get("reference_answer", "")
                q_preview = tc["query"][:60] + ("…" if len(tc["query"]) > 60 else "")
                user_ans = st.text_area(
                    f"Q{tc_idx + 1}: {q_preview}",
                    value=st.session_state.get(ans_key, default_val),
                    height=80,
                    key=ans_key,
                    placeholder="请输入该问题对应的系统回答…",
                    help=(
                        f"Query: {tc['query']}\n\n"
                        "填写 LLM 生成的回答或期望的回答文本。"
                        "Ragas 会基于此评估 faithfulness（忠实度）和 answer_relevancy（相关性）。"
                    ),
                )
                if user_ans.strip():
                    user_answers[tc_idx] = user_ans.strip()

            # 显示填写状态
            filled = len(user_answers)
            total = len(_test_cases)
            if filled < total:
                st.warning(f"⚠️ 已填写 {filled}/{total} 个回答。未填写的用例将使用检索片段拼接作为回答（评估结果可能不准确）。")
            else:
                st.success(f"✅ 所有 {total} 个回答已填写。")
        except Exception as exc:
            st.warning(f"无法加载测试用例预览: {exc}")

    # ── 运行评估 ────────────────────────────────────────
    st.divider()

    run_clicked = st.button(
        "▶️  运行评估",
        type="primary",
        key="eval_run_btn",
        disabled=not golden_path.exists(),
    )

    if run_clicked:
        _run_evaluation(
            backend=backend,
            golden_path=golden_path,
            top_k=int(top_k),
            collection=collection.strip() or None,
            user_answers=user_answers if user_answers else None,
        )

    # ── 历史结果 ───────────────────────────────────────
    st.divider()
    _render_history()


def _run_evaluation(
    backend: str,
    golden_path: Path,
    top_k: int,
    collection: Optional[str],
    user_answers: Optional[Dict[int, str]] = None,
) -> None:
    """执行评估运行并显示结果。

    尝试加载评估器，运行 golden test set，并显示汇总和每个查询的指标。
    失败时会显示友好的错误消息。
    """
    with st.spinner("加载中评估器并运行评估…"):
        try:
            report_dict = _execute_evaluation(
                backend=backend,
                golden_path=golden_path,
                top_k=top_k,
                collection=collection,
                user_answers=user_answers,
            )
        except Exception as exc:
            st.error(f"❌ 评估失败：{exc}")
            logger.exception("Evaluation failed")
            return

    # ── 显示结果 ───────────────────────────────────────
    st.success("✅ 评估完成！")

    _render_aggregate_metrics(report_dict)
    _render_query_details(report_dict)

    # 保存到历史记录
    _save_to_history(report_dict)


def _execute_evaluation(
    backend: str,
    golden_path: Path,
    top_k: int,
    collection: Optional[str],
    user_answers: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """运行评估流程并返回报告字典。

    此函数延迟导入重型依赖以保持仪表板响应性。
    """
    from dataclasses import replace as dc_replace

    from src.core.settings import load_settings
    from src.libs.evaluator.evaluator_factory import EvaluatorFactory
    from src.observability.evaluation.eval_runner import EvalRunner, load_test_set

    settings = load_settings()

    # 从 UI 选择覆盖评估器提供程序 — 构建一个新的完整    # Settings 对象，以便 RagasEvaluator 仍能访问 .llm / .embedding。
    eval_settings = settings.evaluation
    overridden_eval = type(eval_settings)(
        enabled=True,
        provider=backend,
        metrics=eval_settings.metrics if hasattr(eval_settings, "metrics") else [],
    )
    # 仅在完整设置中替换 evaluation 子配置    settings_with_override = dc_replace(settings, evaluation=overridden_eval)

    evaluator = EvaluatorFactory.create(settings_with_override)

    # 尝试创建 HybridSearch（可选 - 无需配置也可工作）    target_collection = collection or "default"
    hybrid_search = _try_create_hybrid_search(settings, target_collection)

    # 如果启用则创建 reranker    reranker = None
    try:
        from src.core.query_engine.reranker import create_core_reranker
        reranker = create_core_reranker(settings=settings)
        if not reranker.is_enabled:
            reranker = None
    except Exception as exc:
        logger.warning("Could not create reranker: %s", exc)

    # 构建 answer_override 映射：index → 用户提供的答案文本    # EvalRunner 将使用这些而不是自动生成片段。
    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
        answer_overrides=user_answers,
        reranker=reranker,
    )

    report = runner.run(
        test_set_path=golden_path,
        top_k=top_k,
        collection=collection,
    )

    return report.to_dict()


def _try_create_hybrid_search(settings: Any, collection: str = "default") -> Any:
    """尝试创建 HybridSearch 实例。

    如果所需依赖不可用则返回 None（例如，没有索引数据）。
    """
    try:
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
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
        bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
        sparse_retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection

        query_processor = QueryProcessor()
        return create_hybrid_search(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )
    except Exception as exc:
        logger.warning("Could not create HybridSearch: %s", exc)
        return None


def _render_aggregate_metrics(report: Dict[str, Any]) -> None:
    """以指标卡片形式显示汇总指标。"""
    st.subheader("📊 汇总指标")

    agg = report.get("aggregate_metrics", {})

    if not agg:
        st.info("暂无汇总指标。")
        return

    cols = st.columns(min(len(agg), 4))
    for idx, (name, value) in enumerate(sorted(agg.items())):
        with cols[idx % len(cols)]:
            st.metric(
                label=name.replace("_", " ").title(),
                value=f"{value:.4f}",
            )

    st.caption(
        f"评估器：**{report.get('evaluator_name', '—')}** · "
        f"查询数：**{report.get('query_count', 0)}** · "
        f"总耗时：**{report.get('total_elapsed_ms', 0):.0f} ms**"
    )


def _render_query_details(report: Dict[str, Any]) -> None:
    """在可展开的表格中显示每个查询的评估结果。"""
    st.subheader("🔍 查询详情")

    query_results = report.get("query_results", [])
    if not query_results:
        st.info("暂无查询详情。")
        return

    for idx, qr in enumerate(query_results):
        query = qr.get("query", "—")
        elapsed = qr.get("elapsed_ms", 0)
        metrics = qr.get("metrics", {})

        # 为展开器标签构建指标摘要
        metric_summary = " · ".join(
            f"{k}: {v:.3f}" for k, v in sorted(metrics.items())
        )
        if not metric_summary:
            metric_summary = "无指标"

        with st.expander(
            f"**Q{idx + 1}**: {query[:80]} — {elapsed:.0f} ms — {metric_summary}",
            expanded=False,
        ):
            # 指标
            if metrics:
                mcols = st.columns(min(len(metrics), 4))
                for midx, (mname, mval) in enumerate(sorted(metrics.items())):
                    with mcols[midx % len(mcols)]:
                        st.metric(mname, f"{mval:.4f}")

            # 检索到的片段
            chunks = qr.get("retrieved_chunk_ids", [])
            if chunks:
                st.markdown(f"**检索到的片段** ({len(chunks)})：")
                st.code(", ".join(chunks[:20]), language=None)

            # 生成的答案
            answer = qr.get("generated_answer")
            if answer:
                st.markdown("**生成的答案:**")
                st.text(answer[:500])


def _render_history() -> None:
    """显示历史评估结果以供对比。"""
    st.subheader("📈 评估历史")

    history = _load_history()
    if not history:
        st.info(
            "**暂无评估历史**。 "
            "配置上方的评估器并点击 **运行评估** 开始。"
            "结果将保存在这里以便跨次运行对比。"
        )
        return

    # 显示最近运行的结果表格
    rows = []
    for entry in history[-10:]:  # last 10 runs
        rows.append(
            {
                "Timestamp": entry.get("timestamp", "—"),
                "Evaluator": entry.get("evaluator_name", "—"),
                "Queries": entry.get("query_count", 0),
                "Time (ms)": round(entry.get("total_elapsed_ms", 0)),
                **{
                    k: round(v, 4)
                    for k, v in entry.get("aggregate_metrics", {}).items()
                },
            }
        )

    st.dataframe(rows, use_container_width=True)


def _save_to_history(report: Dict[str, Any]) -> None:
    """将评估报告追加到历史文件。"""
    try:
        EVAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **report,
        }
        with EVAL_HISTORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to save evaluation history: %s", exc)


def _load_history() -> List[Dict[str, Any]]:
    """从 JSONL 文件加载评估历史。"""
    if not EVAL_HISTORY_PATH.exists():
        return []

    entries: List[Dict[str, Any]] = []
    try:
        with EVAL_HISTORY_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:
        logger.warning("Failed to load evaluation history: %s", exc)

    return entries


def _load_golden_queries(golden_path: Path) -> List[Dict[str, Any]]:
    """从 golden test set 加载测试用例以在 UI 中显示。

    返回至少包含 'query' 键的字典列表，可选包含 'reference_answer' 键。
    """
    with golden_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("test_cases", [])
