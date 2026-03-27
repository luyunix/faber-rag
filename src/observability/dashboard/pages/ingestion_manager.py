"""文档处理页面 – 上传文件、触发处理、删除文档。

布局：
1. 文件上传器 + 集合选择器
2. 处理按钮 → 进度条（使用 on_progress 回调）
3. 文档列表及删除按钮
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


def _run_ingestion(
    uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile",
    collection: str,
    progress_bar: "st.delta_generator.DeltaGenerator",
    status_text: "st.delta_generator.DeltaGenerator",
) -> None:
    """将上传的文件保存到临时位置并运行处理流程。"""
    from src.core.settings import load_settings
    from src.core.trace import TraceContext, TraceCollector
    from src.ingestion.pipeline import IngestionPipeline
    import logging
    import sys

    settings = load_settings()
    
    # 确保日志输出到控制台（Streamlit 环境）
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            stream=sys.stdout,
        )

    # 将上传的文件写入临时位置
    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    _STAGE_LABELS = {
        "integrity": "🔍 检查文件完整性…",
        "load": "📄 加载文档…",
        "split": "✂️ 分块中…",
        "transform": "🔄 转换片段（LLM 优化 + 增强）…",
        "embed": "🔢 编码向量…",
        "upsert": "💾 存储到数据库…",
    }

    def on_progress(stage: str, current: int, total: int) -> None:
        frac = (current - 1) / total  # stage just started, show partial progress
        label = _STAGE_LABELS.get(stage, stage)
        progress_bar.progress(frac, text=f"[{current}/{total}] {label}")
        status_text.caption(label)

    trace = TraceContext(trace_type="ingestion")
    trace.metadata["source_path"] = uploaded_file.name
    trace.metadata["collection"] = collection
    trace.metadata["source"] = "dashboard"

    try:
        pipeline = IngestionPipeline(settings, collection=collection)
        result = pipeline.run(
            file_path=tmp_path,
            trace=trace,
            on_progress=on_progress,
        )
        
        # 记录 source_hash (即 doc_id) 到 trace metadata
        if result and hasattr(result, 'doc_id'):
            trace.metadata["source_hash"] = result.doc_id
        
        progress_bar.progress(1.0, text="✅ 完成")
        status_text.success(f"成功将 **{uploaded_file.name}** 处理到集合 **{collection}** 中。")
    except Exception as exc:
        status_text.error(f"处理失败：{exc}")
    finally:
        TraceCollector().collect(trace)
        # 清理临时文件
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def render() -> None:
    """渲染文档处理页面。"""
    # 页面标题
    st.title("📥 文档处理")
    st.markdown(
        """
        <div style='background-color: #ffffff; padding: 20px; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <p style='font-size: 16px; color: #555;'>
                上传并处理文档到知识库系统。支持 PDF、TXT、MD、DOCX 格式，自动进行分块、向量编码和存储。
                处理完成后，可在下方查看每个文档的详细处理历史。
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── 上传区域 ────────────────────────────────────────
    st.subheader("📤 上传并处理")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader(
            "选择要处理的文件",
            type=["pdf", "txt", "md", "docx"],
            key="ingest_uploader",
            help="支持的文件格式：PDF、TXT、MD、DOCX"
        )
    with col2:
        collection = st.text_input(
            "集合名称", 
            value="default", 
            key="ingest_collection",
            placeholder="default",
            help="文档将存储在此集合中"
        )

    if uploaded is not None:
        if st.button("🚀 开始处理", key="btn_ingest", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="准备中…")
            status_text = st.empty()
            _run_ingestion(uploaded, collection.strip() or "default", progress_bar, status_text)

    st.divider()

    # ── 文档列表及处理历史 ───────────────────────────────
    st.subheader("📚 文档列表与处理历史")

    try:
        svc = DataService()
        docs = svc.list_documents()
    except Exception as exc:
        st.error(f"加载文档失败：{exc}")
        return

    if not docs:
        st.info(
            "**尚未处理任何文档。** "
            "请上传 PDF、TXT、MD 或 DOCX 文件，然后点击 \"开始处理\"。"
        )
        return

    # 加载处理历史
    from src.observability.dashboard.services.trace_service import TraceService
    trace_svc = TraceService()
    ingestion_traces = trace_svc.list_traces(trace_type="ingestion")
    
    # 构建 source_hash 到 trace 的映射 (使用 hash 值匹配，更准确)
    trace_map = {}
    for trace in ingestion_traces:
        # 尝试从 metadata 中获取 source_hash 或 source_path
        meta = trace.get("metadata", {})
        source_hash = meta.get("source_hash")
        if source_hash:
            trace_map[source_hash] = trace

    # 美化文档列表
    for idx, doc in enumerate(docs):
        with st.container():
            # 使用白色卡片展示文档信息
            st.markdown(
                f"""
                <div style='background-color: #ffffff; padding: 15px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 4px solid #667eea;'>
                    <h4 style='margin: 0 0 10px 0; color: #333;'>📄 {doc['source_path']}</h4>
                    <p style='margin: 5px 0; color: #666;'>
                        <strong>集合:</strong> <code style='background-color: #f0f2f6; padding: 2px 6px; border-radius: 3px;'>{doc.get('collection', '—')}</code> ·
                        <strong>片段:</strong> <span style='color: #667eea;'>{doc['chunk_count']}</span> ·
                        <strong>图片:</strong> <span style='color: #764ba2;'>{doc['image_count']}</span>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # 操作按钮区域
            col_del, col_space = st.columns([1, 5])
            with col_del:
                if st.button("🗑️ 删除", key=f"del_{idx}", use_container_width=True):
                    try:
                        result = svc.delete_document(
                            source_path=doc["source_path"],
                            collection=doc.get("collection", "default"),
                            source_hash=doc.get("source_hash"),
                        )
                        if result.success:
                            st.success(
                                f"✅ 已删除：{result.chunks_deleted} 个片段，"
                                f"{result.images_deleted} 张图片已移除。"
                            )
                            st.rerun()
                        else:
                            st.warning(f"⚠️ 部分删除失败。错误：{result.errors}")
                    except Exception as exc:
                        st.error(f"❌ 删除失败：{exc}")
            
            # 处理历史详情 - 使用 source_hash 匹配
            doc_hash = doc.get("source_hash")
            
            if doc_hash and doc_hash in trace_map:
                trace = trace_map[doc_hash]
                with st.expander("📊 查看处理历史", expanded=False):
                    _render_trace_details(trace, trace_svc)
            else:
                st.caption("💡 暂无处理历史记录")


def _render_trace_details(trace: dict, trace_svc: TraceService) -> None:
    """渲染单个处理追踪的详情。"""
    from src.observability.dashboard.services.trace_service import TraceService
    
    trace_id = trace.get("trace_id", "unknown")
    started = trace.get("started_at", "—")
    total_ms = trace.get("elapsed_ms")
    total_label = f"{total_ms:.0f} ms" if total_ms is not None else "—"
    meta = trace.get("metadata", {})
    source_path = meta.get("source_path", "—")
    
    # 概览
    st.caption(f"处理时间：{started[:19]} · 总耗时：{total_label}")
    
    timings = trace_svc.get_stage_timings(trace)
    stages_by_name = {t["stage_name"]: t for t in timings}
    
    # 阶段耗时瀑布图
    main_stages = [
        t for t in timings
        if t["stage_name"] in ("load", "split", "transform", "embed", "upsert")
    ]
    if main_stages:
        st.markdown("#### ⏱️ 阶段耗时")
        chart_data = {t["stage_name"]: t["elapsed_ms"] for t in main_stages}
        st.bar_chart(chart_data, horizontal=True)
    
    # 每个阶段的详情
    st.markdown("#### 🔍 阶段详情")
    
    tab_defs = []
    if "load" in stages_by_name:
        tab_defs.append(("📄 Load", "load"))
    if "split" in stages_by_name:
        tab_defs.append(("✂️ Split", "split"))
    if "transform" in stages_by_name:
        tab_defs.append(("🔄 Transform", "transform"))
    if "embed" in stages_by_name:
        tab_defs.append(("🔢 Embed", "embed"))
    if "upsert" in stages_by_name:
        tab_defs.append(("💾 Upsert", "upsert"))
    
    if tab_defs:
        tabs = st.tabs([label for label, _ in tab_defs])
        for tab, (label, key) in zip(tabs, tab_defs):
            with tab:
                stage = stages_by_name[key]
                data = stage.get("data", {})
                elapsed = stage.get("elapsed_ms")
                if elapsed is not None:
                    st.caption(f"⏱️ {elapsed:.1f} ms")
                
                if key == "load":
                    _render_load_stage(data, trace_id)
                elif key == "split":
                    _render_split_stage(data, trace_id)
                elif key == "transform":
                    _render_transform_stage(data, trace_id)
                elif key == "embed":
                    _render_embed_stage(data)
                elif key == "upsert":
                    _render_upsert_stage(data)


def _render_load_stage(data: dict, trace_idx: str) -> None:
    """Load 阶段：原始文档预览。"""
    c1, c2, c3 = st.columns(3)
    with c1:
        doc_id = data.get("doc_id", "—")
        st.metric("文档 ID", doc_id[:16] if doc_id else "—")
    with c2:
        st.metric("文本长度", f"{data.get('text_length', 0):,} 字符")
    with c3:
        st.metric("图片数量", data.get("image_count", 0))
    
    preview = data.get("text_preview", "")
    if preview:
        st.markdown("**📄 原始文档文本**")
        st.text_area(
            "raw_text",
            value=preview,
            height=max(120, min(len(preview) // 2, 600)),
            disabled=True,
            label_visibility="collapsed",
            key=f"load_raw_{trace_idx}",
        )
    else:
        st.info("暂无文本预览")


def _render_split_stage(data: dict, trace_idx: str) -> None:
    """Split 阶段：分块列表。"""
    c1, c2 = st.columns(2)
    with c1:
        st.metric("分块数量", data.get("chunk_count", 0))
    with c2:
        st.metric("平均大小", f"{data.get('avg_chunk_size', 0)} 字符")
    
    chunks = data.get("chunks", [])
    if chunks:
        st.markdown("**✂️ 分块详情**")
        for i, chunk in enumerate(chunks):
            char_len = chunk.get("char_len", 0)
            chunk_id = chunk.get("chunk_id", "")
            text = chunk.get("text", "")
            header = f"📝 **Chunk #{i+1}** — `{chunk_id[:20] if chunk_id else 'N/A'}` — {char_len} 字符"
            with st.expander(header, expanded=(i < 2)):
                st.text_area(
                    f"split_{i}",
                    value=text,
                    height=max(100, min(len(text) // 2, 500)),
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"split_{trace_idx}_{i}",
                )
    else:
        st.info("暂无分块数据")


def _render_transform_stage(data: dict, trace_idx: str) -> None:
    """Transform 阶段：优化前后对比 + 元数据增强。"""
    # 汇总指标
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "优化 (LLM / 规则)",
            f"{data.get('refined_by_llm', 0)} / {data.get('refined_by_rule', 0)}"
        )
    with c2:
        st.metric(
            "增强 (LLM / 规则)",
            f"{data.get('enriched_by_llm', 0)} / {data.get('enriched_by_rule', 0)}"
        )
    with c3:
        st.metric("图片描述", data.get("captioned_chunks", 0))
    
    chunks = data.get("chunks", [])
    if chunks:
        st.markdown("**🔄 每个片段的优化详情**")
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", "")
            refined_by = chunk.get("refined_by", "")
            enriched_by = chunk.get("enriched_by", "")
            title = chunk.get("title", "")
            tags = chunk.get("tags", [])
            summary = chunk.get("summary", "")
            text_before = chunk.get("text_before", "")
            text_after = chunk.get("text_after", "")
            
            badge_parts = []
            if refined_by:
                badge_parts.append(f"优化:`{refined_by}`")
            if enriched_by:
                badge_parts.append(f"增强:`{enriched_by}`")
            badges = " · ".join(badge_parts)
            
            header = f"🔄 **Chunk #{i+1}** — `{chunk_id[:20] if chunk_id else 'N/A'}` — {badges}"
            with st.expander(header, expanded=(i == 0)):
                # 元数据
                if title or tags or summary:
                    st.markdown("**📋 增强后的元数据**")
                    meta_cols = st.columns(3)
                    with meta_cols[0]:
                        st.markdown(f"**标题:** {title}" if title else "_无标题_")
                    with meta_cols[1]:
                        if tags:
                            st.markdown("**标签:** " + ", ".join(f"`{t}`" for t in tags))
                        else:
                            st.markdown("_无标签_")
                    with meta_cols[2]:
                        if summary:
                            st.markdown(f"**摘要:** {summary}")
                
                # 前后对比
                if text_before or text_after:
                    st.markdown("**📝 文本对比**")
                    _max_len = max(len(text_before or ""), len(text_after or ""))
                    _h = max(150, min(_max_len // 2, 600))
                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.markdown("*优化前:*")
                        st.text_area(
                            f"before_{i}",
                            value=text_before if text_before else "(空)",
                            height=_h,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"transform_before_{trace_idx}_{i}",
                        )
                    with col_after:
                        st.markdown("*优化后:*")
                        st.text_area(
                            f"after_{i}",
                            value=text_after if text_after else "(空)",
                            height=_h,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"transform_after_{trace_idx}_{i}",
                        )
    else:
        st.info("暂无优化数据")


def _render_embed_stage(data: dict) -> None:
    """Embed 阶段：双路编码详情。"""
    # 概览
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Dense 向量", data.get("dense_vector_count", 0))
    with c2:
        st.metric("维度", data.get("dense_dimension", 0))
    with c3:
        st.metric("Sparse 文档", data.get("sparse_doc_count", 0))
    with c4:
        st.metric("方法", data.get("method", "—"))
    
    chunks = data.get("chunks", [])
    if not chunks:
        st.info("暂无编码数据")
        return
    
    # 双路编码表
    st.markdown("---")
    dense_tab, sparse_tab = st.tabs(["🟦 Dense 编码", "🟨 Sparse 编码 (BM25)"])
    
    with dense_tab:
        st.markdown("每个片段 → **浮点向量** (通过 embedding 模型)")
        dense_rows = []
        for i, chunk in enumerate(chunks):
            char_len = chunk.get("char_len", 0)
            dense_rows.append({
                "#": i + 1,
                "Chunk ID": chunk.get("chunk_id", ""),
                "字符数": char_len,
                "预估 Token": max(1, char_len // 3),
                "向量维度": chunk.get("dense_dim", data.get("dense_dimension", "—")),
            })
        st.table(dense_rows)
    
    with sparse_tab:
        st.markdown("每个片段 → **词频统计** (用于 BM25 索引)")
        sparse_rows = []
        for i, chunk in enumerate(chunks):
            sparse_rows.append({
                "#": i + 1,
                "Chunk ID": chunk.get("chunk_id", ""),
                "文档长度 (词)": chunk.get("doc_length", "—"),
                "唯一词数": chunk.get("unique_terms", "—"),
            })
        st.table(sparse_rows)
        
        # 高频词
        for i, chunk in enumerate(chunks):
            top_terms = chunk.get("top_terms", [])
            if top_terms:
                with st.expander(f"🔤 Chunk {i + 1} — 高频词", expanded=False):
                    term_rows = [{"词": t["term"], "频次": t["freq"]} for t in top_terms]
                    st.table(term_rows)


def _render_upsert_stage(data: dict) -> None:
    """Upsert 阶段：存储详情。"""
    dense_store = data.get("dense_store", {})
    sparse_store = data.get("sparse_store", {})
    image_store = data.get("image_store", {})
    
    # 概览
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Dense 向量", dense_store.get("count", data.get("vector_count", 0)))
    with c2:
        st.metric("Sparse (BM25)", sparse_store.get("count", data.get("bm25_docs", 0)))
    with c3:
        st.metric("图片", image_store.get("count", data.get("images_indexed", 0)))
    
    # Dense store
    if dense_store:
        with st.expander("🟦 Dense Vector Store (ChromaDB)", expanded=True):
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown(f"**后端:** `{dense_store.get('backend', '—')}`")
                st.markdown(f"**集合:** `{dense_store.get('collection', '—')}`")
            with dc2:
                st.markdown(f"**路径:** `{dense_store.get('path', '—')}`")
                st.markdown(f"**向量数:** {dense_store.get('count', 0)}")
    
    # Sparse store
    if sparse_store:
        with st.expander("🟨 Sparse Index (BM25)", expanded=True):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(f"**后端:** `{sparse_store.get('backend', '—')}`")
                st.markdown(f"**集合:** `{sparse_store.get('collection', '—')}`")
            with sc2:
                st.markdown(f"**路径:** `{sparse_store.get('path', '—')}`")
                st.markdown(f"**文档数:** {sparse_store.get('count', 0)}")
    
    # Image store
    if image_store and image_store.get("count", 0) > 0:
        with st.expander(f"🖼️ 图片存储 ({image_store.get('count', 0)} 张图片)", expanded=True):
            st.markdown(f"**后端:** `{image_store.get('backend', '—')}`")
            imgs = image_store.get("images", [])
            if imgs:
                img_rows = [
                    {
                        "图片 ID": img.get("image_id", ""),
                        "页码": img.get("page", 0),
                        "文件": img.get("file_path", ""),
                        "文档 Hash": img.get("doc_hash", "")[:16] + "…",
                    }
                    for img in imgs
                ]
                st.table(img_rows)
    
    # Chunk → Vector 映射
    chunk_mapping = data.get("chunk_mapping", [])
    if chunk_mapping:
        with st.expander(f"🔗 Chunk → Vector 映射 ({len(chunk_mapping)} 条)", expanded=False):
            mapping_rows = [
                {
                    "#": i + 1,
                    "Chunk ID": m.get("chunk_id", ""),
                    "Vector ID": m.get("vector_id", ""),
                    "存储": m.get("store", ""),
                    "集合": m.get("collection", ""),
                }
                for i, m in enumerate(chunk_mapping)
            ]
            st.table(mapping_rows)
