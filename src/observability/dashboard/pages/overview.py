"""总览页面 – 系统配置和数据统计。

显示:
- 组件配置 (LLM、Embedding、VectorStore 等)
- 集合统计 (文档数量、片段数量、图片数量)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

from src.observability.dashboard.services.config_service import ConfigService


def _safe_collection_stats() -> Dict[str, Any]:
    """尝试从 ChromaDB 加载集合统计信息。

    失败时返回空字典以便页面仍能正常渲染。
    """
    try:
        from src.core.settings import load_settings, resolve_path
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        settings = load_settings()
        persist_dir = str(
            resolve_path(settings.vector_store.persist_directory)
        )
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        stats: Dict[str, Any] = {}
        for col in client.list_collections():
            name = col.name if hasattr(col, "name") else str(col)
            collection = client.get_collection(name)
            stats[name] = {"chunk_count": collection.count()}
        return stats
    except Exception:
        return {}


def render() -> None:
    """渲染总览页面。"""
    # 页面标题
    st.title("📊 系统总览")
    st.markdown(
        """
        <div style='background-color: #ffffff; padding: 20px; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <p style='font-size: 16px; color: #555;'>
                欢迎使用 <strong>Faber RAG</strong> 系统。查看组件配置、数据统计
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── 组件配置卡片 ──────────────────────────────────────────────
    st.subheader("🔧 组件配置")

    try:
        config_service = ConfigService()
        cards = config_service.get_component_cards()
    except Exception as exc:
        st.error(f"加载配置失败：{exc}")
        return

    # 紧凑展示所有组件配置 - 使用多列布局
    cols = st.columns(min(len(cards), 3))
    for idx, card in enumerate(cards):
        with cols[idx % len(cols)]:
            # 组件名称
            st.markdown(f"**{card.name}**")
            
            # 直接显示所有信息
            st.caption(f"Provider: `{card.provider}`")
            st.caption(f"Model: `{card.model}`")
            
            # 显示额外的配置信息 - 直接展开
            if card.extra:
                for k, v in card.extra.items():
                    st.caption(f"{k}: `{v}`")
            
            st.divider()

    # ── 数据统计 ──────────────────────────────────────────────
    st.subheader("📊 数据统计")
    
    # 第一行：文档统计、Chunk 统计、查询量统计
    doc_stats_cols = st.columns(3)
    
    # 文档统计
    with doc_stats_cols[0]:
        try:
            from src.observability.dashboard.services.data_service import DataService
            data_svc = DataService()
            docs = data_svc.list_documents()
            doc_count = len(docs)
            
            st.markdown(
                f"""
                <div style='background-color: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center;'>
                    <h2 style='margin: 0; color: #667eea; font-size: 32px;'>{doc_count}</h2>
                    <p style='margin: 10px 0 0 0; color: #666; font-size: 14px;'>📄 文档总数</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as exc:
            st.error(f"加载文档统计失败：{exc}")
    
    # Chunk 统计
    with doc_stats_cols[1]:
        stats = _safe_collection_stats()
        if stats:
            total_chunks = sum(info.get('chunk_count', 0) for info in stats.values())
            st.markdown(
                f"""
                <div style='background-color: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center;'>
                    <h2 style='margin: 0; color: #764ba2; font-size: 32px;'>{total_chunks:,}</h2>
                    <p style='margin: 10px 0 0 0; color: #666; font-size: 14px;'>📝 Chunk 总数</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("暂无 Chunk 数据")
    
    # 查询量统计
    with doc_stats_cols[2]:
        from src.core.settings import resolve_path
        traces_path = resolve_path("logs/traces.jsonl")
        if traces_path.exists():
            line_count = sum(1 for _ in traces_path.open(encoding="utf-8"))
            if line_count > 0:
                st.markdown(
                    f"""
                    <div style='background-color: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center;'>
                        <h2 style='margin: 0; color: #28a745; font-size: 32px;'>{line_count}</h2>
                        <p style='margin: 10px 0 0 0; color: #666; font-size: 14px;'>🔍 查询总数</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("暂无查询记录")
        else:
            st.info("暂无查询记录")
