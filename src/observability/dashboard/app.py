"""Modular RAG Dashboard – multi-page Streamlit application.

入口点：``streamlit run src/observability/dashboard/app.py``

页面通过 ``st.navigation()`` 注册，并由 ``pages/`` 下的相应模块渲染。
尚未实现的页面会显示占位消息。
"""

from __future__ import annotations

import logging
import warnings

import streamlit as st

# 抑制 transformers 的警告logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# ── Page definitions ─────────────────────────────────────────────────

def _page_overview() -> None:
    from src.observability.dashboard.pages.overview import render
    render()


def _page_data_browser() -> None:
    from src.observability.dashboard.pages.data_browser import render
    render()


def _page_ingestion_manager() -> None:
    from src.observability.dashboard.pages.ingestion_manager import render
    render()


def _page_query_traces() -> None:
    from src.observability.dashboard.pages.query_traces import render
    render()


def _page_evaluation_panel() -> None:
    from src.observability.dashboard.pages.evaluation_panel import render
    render()


def _page_mcp_query() -> None:
    from src.observability.dashboard.pages.mcp_query import render
    render()


# ── Navigation ───────────────────────────────────────────────────────

pages = [
    st.Page(_page_overview, title="系统总览", icon="📊", default=True),
    st.Page(_page_data_browser, title="数据浏览", icon="🔍"),
    st.Page(_page_ingestion_manager, title="文档处理", icon="📥"),
    st.Page(_page_query_traces, title="查询历史", icon="🔎"),
    st.Page(_page_evaluation_panel, title="评估面板", icon="📏"),
    st.Page(_page_mcp_query, title="MCP 查询", icon="🌐"),
]


def main() -> None:
    st.set_page_config(
        page_title="模块化 RAG 仪表盘",
        page_icon="📊",
        layout="wide",
    )
    
    # 自定义侧边栏样式 - 白色主题
    st.markdown(
        """
        <style>
        /* 侧边栏背景色 */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
        }
        
        /* 侧边栏菜单项 - 默认状态 */
        [data-testid="stSidebar"] .st-emotion-cache-16idsys {
            color: #333333;
        }
        
        /* 侧边栏菜单项 - 悬停状态 */
        [data-testid="stSidebar"] .st-emotion-cache-16idsys:hover {
            background-color: #f0f2f6;
            color: #667eea;
        }
        
        /* 侧边栏菜单项 - 选中状态 */
        [data-testid="stSidebar"] .st-emotion-cache-16idsys.p {
            background-color: #e8f4f8;
            color: #667eea;
            border-left: 3px solid #667eea;
        }
        
        /* 侧边栏分割线 */
        [data-testid="stSidebar"] hr {
            border-color: #e0e0e0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # 配置日志级别，减少不必要的警告
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # 设置 dashboard 页面日志级别为 ERROR 以查看详细错误
    logging.getLogger("src.observability.dashboard.pages.mcp_query").setLevel(logging.ERROR)

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
else:
    # When run directly via `streamlit run app.py`
# 当通过 `streamlit run app.py` 直接运行时
    main()
