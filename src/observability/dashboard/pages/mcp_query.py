"""MCP 查询页面 – 通过 MCP 协议访问 RAG 系统进行查询。

布局：
1. MCP 连接配置（服务器地址、端口）
2. 查询输入区域
3. 查询结果显示
4. 历史记录
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)


async def _call_mcp_tool_async(
    server_url: str,
    tool_name: str,
    tool_args: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """通过 MCP 协议调用工具（异步）。
    
    Args:
        server_url: MCP 服务器 URL
        tool_name: 工具名称
        tool_args: 工具参数
        
    Returns:
        工具调用结果，失败返回 None
    """
    try:
        from src.mcp_server.mcp_client import MCPClient, MCPToolResult
        
        async with MCPClient(server_url=server_url) as client:
            logger.info(f"Calling MCP tool: {tool_name} with args: {tool_args}")
            result: MCPToolResult = await client.call_tool(
                tool_name=tool_name,
                arguments=tool_args,
            )
            
            logger.info(f"MCP tool result: is_error={result.is_error}, content_count={len(result.content)}")
            
            return {
                "content": result.content,
                "is_error": result.is_error,
                "text": result.get_text(),
                "raw": result.raw_response,
            }
    except ImportError as e:
        logger.error(f"MCP SDK 未安装：{e}")
        st.error("MCP SDK 未安装。请运行：`uv pip install mcp`")
        return None
    except Exception as exc:
        logger.exception(f"MCP 调用失败：{exc}")
        st.error(f"MCP 调用失败：{exc}")
        return None


def _call_mcp_tool(
    server_url: str,
    tool_name: str,
    tool_args: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """通过 MCP 协议调用工具（同步包装器）。
    
    Args:
        server_url: MCP 服务器 URL
        tool_name: 工具名称
        tool_args: 工具参数
        
    Returns:
        工具调用结果，失败返回 None
    """
    try:
        import asyncio
        result = asyncio.run(_call_mcp_tool_async(server_url, tool_name, tool_args))
        logger.error(f"_call_mcp_tool returned: {type(result)} - {result}")
        return result
    except Exception as exc:
        logger.exception(f"MCP 调用失败：{exc}")
        st.error(f"MCP 调用失败：{exc}")
        return None


def render() -> None:
    """渲染 MCP 查询页面。"""
    # 页面标题和介绍
    st.title("🔍 MCP 智能查询")
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
            <p style='font-size: 16px; color: #555;'>
                通过 <strong>MCP 协议</strong> 连接到 RAG 系统进行智能查询。
                支持混合检索、语义理解和多模态响应。
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # ── MCP 连接配置 ────────────────────────────────────────
    with st.expander("⚙️ MCP 连接配置", expanded=False):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            server_url = st.text_input(
                "MCP 服务器 URL",
                value="http://localhost:8080",
                help="MCP 服务器的 HTTP 端点地址",
                placeholder="http://localhost:8080"
            )
        with col2:
            timeout = st.number_input(
                "超时时间 (秒)",
                min_value=1,
                max_value=300,
                value=30,
                help="MCP 调用的超时时间"
            )
        with col3:
            # 健康检查按钮
            if st.button("🔍 检查", key="health_check_btn", use_container_width=True):
                try:
                    import httpx
                    response = httpx.get(f"{server_url}/health", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ 服务器在线")
                    else:
                        st.error(f"❌ 服务器响应：{response.status_code}")
                except Exception as e:
                    st.error(f"❌ 无法连接：{e}")
    
    st.divider()
    
    # ── 查询输入区域 ────────────────────────────────────────
    st.subheader("📝 查询输入")
    
    query = st.text_area(
        "输入查询内容",
        height=120,
        placeholder="请输入您的问题，例如：工厂的生产流程是什么？",
        help="输入要查询的问题或关键词",
        key="query_input"
    )
    
    # 高级选项
    with st.expander("🔧 高级选项", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            collection = st.text_input(
                "集合名称",
                value="default",
                help="要查询的集合名称",
                placeholder="default"
            )
            top_k = st.slider(
                "返回结果数量",
                min_value=1,
                max_value=50,
                value=10,
                help="返回的检索结果数量"
            )
        with col2:
            use_rerank = st.checkbox(
                "启用重排序",
                value=False,
                help="是否使用 Rerank 对结果进行重排序"
            )
            st.info("💡 启用重排序可以提高结果质量，但会增加响应时间")
    
    # 查询按钮
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
    col_btn, col_space = st.columns([1, 5])
    with col_btn:
        query_clicked = st.button("🚀 开始查询", type="primary", key="mcp_query_btn", use_container_width=True)
    
    if query_clicked and query:
        # 调用 MCP 工具
        with st.spinner("🔍 正在智能查询中，请稍候..."):
            result = _call_mcp_tool(
                server_url=server_url,
                tool_name="query_knowledge_hub",
                tool_args={
                    "query": query,
                    "top_k": top_k,
                    "collection": collection,
                }
            )
            
            # 调试：显示返回结果
            logger.info(f"MCP call result: {result}")
            
            if result is None:
                st.error("❌ 查询失败：MCP 调用返回 None，请检查服务器日志")
            elif result.get("is_error"):
                st.error(f"❌ 查询失败：{result.get('text', '未知错误')}")
            else:
                # 显示查询结果
                st.success("✅ 查询成功！")
                
                # 美化结果显示
                st.markdown(
                    """
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;'>
                        <h3 style='color: white; margin: 0;'>📄 智能检索结果</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(result.get("text", ""))
                
                # 显示原始响应（调试用）
                with st.expander("🔍 查看原始响应 (JSON)"):
                    st.json(result.get("raw", {}))
    
    st.divider()
    
    # ── 查询历史 ────────────────────────────────────────
    st.subheader("📋 最近查询历史")
    
    # 从 trace 文件加载最近的查询
    from src.observability.dashboard.services.trace_service import TraceService
    
    svc = TraceService()
    query_traces = svc.list_traces(trace_type="query", limit=5)
    
    if query_traces:
        for trace in query_traces:
            started = trace.get("started_at", "—")
            query_text = trace.get("metadata", {}).get("query", "—")
            elapsed = trace.get("elapsed_ms", 0)
            
            # 美化历史查询显示
            with st.expander(f"📄 {query_text[:50]}... · {started[:19]}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("⏱️ 耗时", f"{elapsed:.0f} ms")
                with col2:
                    st.metric("📊 状态", "✅ 完成")
                
                # 显示阶段信息
                stages = svc.get_stage_timings(trace)
                if stages:
                    st.markdown("**⚙️ 处理阶段详情**:")
                    for stage in stages:
                        stage_name = stage.get("stage_name", "unknown")
                        elapsed_ms = stage.get("elapsed_ms", 0)
                        st.progress(min(elapsed_ms / 1000, 1.0))
                        st.caption(f"  • {stage_name}: {elapsed_ms:.0f} ms")
    else:
        st.info("💡 暂无查询历史。请先进行查询操作。")
    
    # ── 使用说明 ────────────────────────────────────────
    st.divider()
    with st.expander("📖 使用说明", expanded=False):
        st.markdown("""
        ### 🚀 快速开始指南
        
        **1. 启动 MCP Server**
        ```bash
        python scripts/start.py
        ```
        
        **2. 配置连接** (可选)
        - 展开 \"MCP 连接配置\"
        - 输入服务器 URL（默认：http://localhost:8080）
        - 点击 \"🔍 检查\" 验证连接
        
        **3. 执行查询**
        - 在查询框中输入问题
        - 点击 \"🚀 开始查询\" 按钮
        - 等待智能检索结果
        
        **4. 查看结果**
        - 查看格式化的智能答案
        - 查看来源引用和置信度
        - 展开 \"🔍 查看原始响应\" 查看技术细节
        
        ---
        
        ### 🛠️ 可用工具
        
        - **🔍 query_knowledge_hub**: 知识库智能查询（混合检索 + 重排序）
        - **📋 list_collections**: 列出所有文档集合
        - **📄 get_document_summary**: 获取文档摘要
        
        ### 💡 最佳实践
        
        - 使用具体、明确的问题
        - 启用重排序以获得更准确的结果
        - 调整 `top_k` 参数控制返回结果数量
        - 使用集合名称限定搜索范围
        """)
