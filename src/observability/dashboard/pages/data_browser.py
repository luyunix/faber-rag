"""数据浏览页面 – 浏览已摄取的文档、片段和图片。

布局：
1. 集合选择器（侧边栏）
2. 文档列表及片段数量
3. 可展开的文档详情 → 带文本和元数据的片段卡片
4. 图片预览画廊
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


def render() -> None:
    """渲染数据浏览页面。"""
    # 页面标题
    st.title("🔍 数据浏览")
    st.markdown(
        """
        <div style='background-color: #ffffff; padding: 20px; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <p style='font-size: 16px; color: #555;'>
                浏览已摄取的文档、片段和图片。查看详细信息、元数据和内容预览。
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        svc = DataService()
    except Exception as exc:
        st.error(f"初始化 DataService 失败：{exc}")
        return

    # ── 集合选择器 ───────────────────────────────────────
    collections = svc.list_collections()
    if "default" not in collections:
        collections.insert(0, "default")
    collection = st.selectbox(
        "集合",
        options=collections,
        index=0,
        key="db_collection_filter",
        help="选择要查看的集合"
    )
    coll_arg = collection if collection else None

    # ── 危险区域：清空所有数据 ───────────────────────────────
    st.divider()
    with st.expander("⚠️ 危险区域", expanded=False):
        st.markdown(
            """
            <div style='background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545;'>
                <h4 style='margin: 0 0 10px 0; color: #dc3545;'>⚠️ 高风险操作</h4>
                <p style='margin: 0; color: #856404;'>
                    此操作将 <strong>永久删除</strong> 所有数据：
                    ChromaDB 集合、BM25 索引、图片、处理历史记录和追踪日志。
                    <strong>此操作无法撤销！</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        col_btn, col_status = st.columns([1, 2])
        with col_btn:
            if st.button("🗑️ 清空所有数据", type="primary", key="btn_clear_all", use_container_width=True):
                st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear"):
            st.error("⚠️ 确定吗？此操作无法撤销！")
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                if st.button("✅ 是，删除所有内容", key="btn_confirm_clear", use_container_width=True):
                    result = svc.reset_all()
                    st.session_state["confirm_clear"] = False
                    if result["errors"]:
                        st.warning(
                            f"⚠️ 清理完成，但有 {len(result['errors'])} 个错误："
                            + "; ".join(result["errors"])
                        )
                    else:
                        st.success(
                            f"✅ 已清空所有数据！"
                            f"{result['collections_deleted']} 个集合已被删除。"
                        )
                    st.rerun()
            with c2:
                if st.button("❌ 取消", key="btn_cancel_clear", use_container_width=True):
                    st.session_state["confirm_clear"] = False
                    st.rerun()

    st.divider()

    # ── 文档列表 ──────────────────────────────────────────────
    try:
        docs = svc.list_documents(coll_arg)
    except Exception as exc:
        st.error(f"加载文档失败：{exc}")
        return

    if not docs:
        st.info(
            "**此集合中没有文档。** "
            "请使用文档处理页面上传并摄取文件，"
            "或从上方下拉菜单选择不同的集合。"
        )
        return

    st.subheader(f"📄 文档 ({len(docs)})")

    for idx, doc in enumerate(docs):
        source_name = Path(doc["source_path"]).name
        label = f"📑 {source_name}  —  {doc['chunk_count']} 个片段 · {doc['image_count']} 张图片"
        with st.expander(label, expanded=(len(docs) == 1)):
            # ── 文档元数据 ─────────────────────────────────
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("片段", doc["chunk_count"])
            col_b.metric("图片", doc["image_count"])
            col_c.metric("集合", doc.get("collection", "—"))
            st.caption(
                f"**来源**: {doc['source_path']}  ·  "
                f"**哈希**: `{doc['source_hash'][:16]}…`  ·  "
                f"**处理时间**: {doc.get('processed_at', '—')}"
            )

            st.divider()

            # ── 片段卡片 ───────────────────────────────────────
            chunks = svc.get_chunks(doc["source_hash"], coll_arg)
            if chunks:
                st.markdown(f"### 📦 片段 ({len(chunks)})")
                for cidx, chunk in enumerate(chunks):
                    text = chunk.get("text", "")
                    meta = chunk.get("metadata", {})
                    chunk_id = chunk["id"]

                    # 从元数据获取标题或使用第一行
                    title = meta.get("title", "")
                    if not title:
                        title = text[:60].replace("\n", " ").strip()
                        if len(text) > 60:
                            title += "…"

                    # 美化片段卡片
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                                <h5 style='margin: 0 0 10px 0; color: #333;'>
                                    📝 Chunk {cidx + 1} · <code style='background-color: #e9ecef; padding: 2px 6px; border-radius: 3px;'>{chunk_id[-16:]}</code> ·
                                    <span style='color: #667eea;'>{len(text)} chars</span>
                                </h5>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        # 显示实际的片段文本（可滚动）
                        _height = max(120, min(len(text) // 2, 600))
                        st.text_area(
                            "内容",
                            value=text,
                            height=_height,
                            disabled=True,
                            key=f"chunk_text_{idx}_{cidx}",
                            label_visibility="collapsed",
                        )
                        # 可展开的元数据
                        with st.expander("📋 元数据", expanded=False):
                            st.json(meta)
            else:
                st.caption("在向量存储中未找到该文档的片段。")

            # ── 图片预览 ─────────────────────────────────────
            images = svc.get_images(doc["source_hash"], coll_arg)
            if images:
                st.divider()
                st.markdown(f"### 🖼️ 图片 ({len(images)})")
                img_cols = st.columns(min(len(images), 4))
                for iidx, img in enumerate(images):
                    with img_cols[iidx % len(img_cols)]:
                        img_path = Path(img.get("file_path", ""))
                        if img_path.exists():
                            st.image(str(img_path), caption=img["image_id"], use_container_width=True)
                        else:
                            st.caption(f"{img['image_id']} (文件丢失)")
