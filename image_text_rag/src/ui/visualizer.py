"""
Streamlit 可视化界面 — 多模态 RAG 技术工程 Demo

启动: streamlit run src/ui/visualizer.py
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import streamlit as st

import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.loader import Config
from src.embedding.clip_onnx import CLIPEmbedderONNX as CLIPEmbedder
from src.vectordb.store import VectorStore
from src.generator.llm import Generator
from src.preprocessing.chunker import TextChunker
from src.preprocessing.image_processor import ImagePreprocessor
from src.service.pipeline import RAGPipeline
from src.retriever.engine import RetrievedChunk

st.set_page_config(
    page_title=Config.UI_TITLE,
    page_icon=Config.UI_PAGE_ICON,
    layout="wide",
)

# ══════════════════════════════════════════════════════════════
# 紫色主题 CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp { background: linear-gradient(160deg, #0b0714 0%, #150b24 50%, #0b0714 100%); }
    .main .block-container { position: relative; z-index: 1; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #120720 0%, #0b0714 100%) !important;
        border-right: 1px solid rgba(147,51,234,.2) !important;
    }
    [data-testid="stSidebar"] * { color: #d8b4fe !important; }
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color: #c084fc !important; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size:1.5rem!important; color:#c084fc!important; }
    [data-testid="stSidebar"] hr { border-color:rgba(147,51,234,.2); }
    [data-testid="stSidebar"] .stButton>button {
        background:linear-gradient(135deg,#5b21b6,#7c3aed)!important;color:#faf5ff!important;
        border:1px solid #7c3aed!important;border-radius:10px!important;font-weight:600!important;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background:linear-gradient(135deg,#7c3aed,#8b5cf6)!important;border-color:#a78bfa!important;
        box-shadow:0 0 16px rgba(124,58,237,.4)!important;
    }
    h1{color:#d8b4fe!important;font-weight:700!important;}
    h2,h3{color:#c084fc!important;}
    .stButton>button{background:linear-gradient(135deg,#5b21b6,#7c3aed)!important;color:#faf5ff!important;border:none!important;border-radius:10px!important;font-weight:600!important;}
    .stButton>button:hover{box-shadow:0 0 20px rgba(139,92,246,.4)!important;transform:translateY(-1px);}
    [data-testid="stChatMessage"]{background:rgba(20,10,40,.65)!important;border:1px solid rgba(147,51,234,.12)!important;border-radius:14px!important;}
    [data-testid="stChatInput"] textarea{background:rgba(20,10,40,.7)!important;color:#e9d5ff!important;border:1px solid rgba(147,51,234,.25)!important;border-radius:14px!important;}
    [data-testid="stChatInput"] textarea:focus{border-color:#8b5cf6!important;box-shadow:0 0 16px rgba(139,92,246,.25)!important;}
    [data-testid="stFileUploader"]{background:rgba(20,10,40,.4)!important;border:1px dashed rgba(147,51,234,.25)!important;border-radius:12px!important;}
    .stProgress>div>div{background:linear-gradient(90deg,#7c3aed,#a78bfa)!important;}
    [data-testid="stExpander"]{background:rgba(20,10,40,.4)!important;border:1px solid rgba(147,51,234,.15)!important;border-radius:12px!important;}
    [data-testid="stExpander"] summary{color:#c084fc!important;}
    .stTabs [data-baseweb="tab-list"]{background:rgba(20,10,40,.4)!important;border-radius:12px!important;padding:3px!important;gap:3px!important;}
    .stTabs [data-baseweb="tab"]{color:#a78bfa!important;border-radius:10px!important;padding:8px 20px!important;}
    .stTabs [aria-selected="true"]{background:linear-gradient(135deg,#5b21b6,#7c3aed)!important;color:#faf5ff!important;}
    .card{background:rgba(20,10,40,.5)!important;border:1px solid rgba(147,51,234,.12)!important;border-radius:14px!important;padding:1.25rem!important;margin:.5rem 0!important;}
    .card:hover{border-color:rgba(147,51,234,.3)!important;}
    .score-bar-wrap{background:rgba(147,51,234,.12);border-radius:6px;height:5px;margin:4px 0 8px;overflow:hidden;}
    .score-bar-fill{height:5px;border-radius:6px;}
    .score-high{background:#a78bfa;}.score-mid{background:#c4b5fd;}.score-low{background:#7c3aed;}
    .tag{display:inline-block;padding:2px 10px;border-radius:9999px;font-size:.72rem;font-weight:600;margin-right:8px;}
    .tag-text{background:rgba(139,92,246,.18);color:#a78bfa;}.tag-image{background:rgba(167,139,250,.18);color:#c4b5fd;}
    .tag-pair{background:rgba(196,181,253,.18);color:#ddd6fe;}
    .mode-badge{display:inline-flex;align-items:center;gap:6px;background:rgba(139,92,246,.12);border:1px solid rgba(139,92,246,.2);border-radius:9999px;padding:4px 14px;font-size:.78rem;font-weight:600;color:#a78bfa;}
    .pulse-dot{width:6px;height:6px;border-radius:50%;display:inline-block;}
    .pulse-rag{background:#8b5cf6;box-shadow:0 0 6px #8b5cf6;animation:pulse 1.5s infinite;}
    .pulse-llm{background:#a78bfa;box-shadow:0 0 6px #a78bfa;animation:pulse 1.5s infinite;}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
    .hero{text-align:center;padding:3.5rem 1rem;}
    .hero h2{font-size:2.2rem;font-weight:800;margin-bottom:.5rem;color:#d8b4fe!important;}
    .hero p{font-size:1rem;color:#8b7aad;max-width:540px;margin:0 auto 2rem;line-height:1.7;}
    .quick-card{background:rgba(20,10,40,.4);border:1px solid rgba(147,51,234,.12);border-radius:16px;padding:1.75rem 1.5rem;text-align:center;}
    .quick-card:hover{border-color:#7c3aed;box-shadow:0 6px 24px rgba(124,58,237,.15);transform:translateY(-3px);}
    .quick-card .icon{font-size:2.3rem;margin-bottom:.6rem;}.quick-card .title{font-weight:700;color:#d8b4fe;margin-bottom:.3rem;}
    .quick-card .desc{font-size:.8rem;color:#8b7aad;line-height:1.5;}
    [data-testid="stMetric"]{background:rgba(139,92,246,.06)!important;border-radius:12px!important;padding:.5rem .75rem!important;}
    .kv-table{width:100%;border-collapse:collapse;font-size:.85rem;}
    .kv-table td{padding:6px 12px;border-bottom:1px solid rgba(147,51,234,.1);}
    .kv-table td:first-child{color:#8b7aad;white-space:nowrap;}.kv-table td:last-child{color:#d8b4fe;font-family:monospace;}
    .pipeline-step{display:flex;align-items:center;gap:8px;padding:8px 0;}
    .pipeline-dot{width:10px;height:10px;border-radius:50%;background:#7c3aed;}
    .pipeline-arrow{color:#5b21b6;font-weight:700;}
    .latency{font-family:monospace;color:#8b7aad;font-size:.8rem;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 组件初始化
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def _init_pipeline():
    emb = CLIPEmbedder()
    store = VectorStore(dim=emb.dim, persist_dir=Config.PERSIST_DIR)
    gen = Generator()
    return RAGPipeline(emb, store, gen, TextChunker(), ImagePreprocessor())

def _init_session():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = _init_pipeline()
    if "messages" not in st.session_state:
        st.session_state.messages = []

_init_session()

pipe: RAGPipeline = st.session_state.pipeline

# ══════════════════════════════════════════════════════════════
# 对话处理函数
# ══════════════════════════════════════════════════════════════
def _handle_rag_chat(question: str):
    "RAG 增强对话 — 委托给 RAGPipeline"
    st.session_state.messages.append({"role": "user", "content": question})
    answer, results, timing = pipe.answer(question)
    timing_line = f"\n\n---\n⚡ 检索 {timing.get('search_ms',0):.0f}ms · 生成 {timing.get('llm_ms',0):.0f}ms"
    st.session_state.messages.append({
        "role": "assistant", "content": answer + timing_line,
        "results": results, "timing": timing,
    })
    st.rerun()

def _handle_pure_llm_chat(question: str):
    "纯 LLM 对话 — 委托给 RAGPipeline"
    st.session_state.messages.append({"role": "user", "content": question})
    answer, timing = pipe.answer_pure_llm(question)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer + f"\n\n---\n⚡ LLM 生成 {timing.get('llm_ms',0):.0f}ms",
        "results": [], "timing": timing,
    })
    st.rerun()

# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════
def _score_bar(score: float, w: int = 80) -> str:
    c = "score-high" if score > 0.7 else ("score-mid" if score > 0.4 else "score-low")
    return f'<div class="score-bar-wrap" style="width:{w}px"><div class="score-bar-fill {c}" style="width:{min(score*100,100):.1f}%"></div></div>'

def _type_tag(doc_type: str) -> str:
    return {"image": '<span class="tag tag-image">📷</span>',
            "image_text_pair": '<span class="tag tag-pair">🖼</span>'}.get(doc_type, '<span class="tag tag-text">📄</span>')

def _mode_badge(has_rag: bool) -> str:
    if has_rag:
        return '<span class="mode-badge"><span class="pulse-dot pulse-rag"></span>RAG 增强模式</span>'
    return '<span class="mode-badge"><span class="pulse-dot pulse-llm"></span>纯 LLM 对话</span>'

def _render_result_card(r: RetrievedChunk):
    score_pct = f"{r.score:.1%}"
    c = "#a78bfa" if r.score > 0.7 else ("#c4b5fd" if r.score > 0.4 else "#8b5cf6")
    st.markdown(f"""
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <div>{_type_tag(r.doc_type)} <strong style="font-size:.9rem; color:#d8b4fe;">{r.source}</strong></div>
            <span style="font-weight:700; font-size:1.1rem; color:{c};">{score_pct}</span>
        </div>
        {_score_bar(r.score)}
        <p style="color:#c4b5fd; font-size:.88rem; line-height:1.6; margin-top:6px;">{r.text[:250]}{'...' if len(r.text)>250 else ''}</p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 侧边栏
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📁 知识库管理")
    st.markdown("### 📤 上传文档")
    uploaded_files = st.file_uploader(
        "支持 txt / md / png / jpg / jpeg",
        type=["txt", "md", "png", "jpg", "jpeg"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    if uploaded_files:
        st.caption(f"已选择 **{len(uploaded_files)}** 个文件")
        for f in uploaded_files:
            ext = Path(f.name).suffix.lower()
            icon = "🖼" if ext in (".png", ".jpg", ".jpeg") else "📄"
            st.caption(f"  {icon} {f.name}  `{(f.size/1024):.1f} KB`")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        ingest_btn = st.button("📥 批量入库", use_container_width=True, disabled=not uploaded_files)
    with col_b:
        auto_ingest = st.checkbox("⚡", value=False, help="秒传模式")

    if (ingest_btn or (auto_ingest and uploaded_files)) and uploaded_files:
        added_total = 0
        progress = st.progress(0, "入库中...")
        status_text = st.empty()
        for idx, f in enumerate(uploaded_files):
            ext = Path(f.name).suffix.lower(); source = f.name
            status_text.caption(f"⏳ {source}")
            try:
                if ext in (".png", ".jpg", ".jpeg"):
                    n = pipe.ingest_image(f.read(), source)
                    added_total += n
                elif ext in (".txt", ".md"):
                    text = f.read().decode("utf-8")
                    n = pipe.ingest_text(text, source)
                    added_total += n
            except Exception as e: st.toast(f"❌ {source}: {e}")
            progress.progress((idx + 1) / len(uploaded_files))
        status_text.caption(f"✅ 入库: {added_total} 条向量")
        st.toast(f"✅ 入库: {added_total} 条"); st.rerun()

    st.markdown("---")
    st.markdown("### 📊 系统状态")
    llm_ok = pipe.llm_healthy
    c1, c2, c3 = st.columns(3)
    c1.metric("向量数", pipe.total_vectors)
    c2.metric("维度", pipe.embedder.dim)
    c3.metric("LLM", "🟣" if llm_ok else "⚫", delta="在线" if llm_ok else "离线")

    st.markdown("---")
    if st.button("🗑 清空知识库", use_container_width=True):
        st.session_state.messages = []; st.rerun()

# ══════════════════════════════════════════════════════════════
# 主界面
# ══════════════════════════════════════════════════════════════
tab_chat, tab_browse, tab_tech = st.tabs([
    "💬 智能对话", "🔍 知识库浏览", "🔬 技术仪表盘"
])

# ── TAB 1: 对话 ────────────────────────────────────────────
with tab_chat:
    has_rag = pipe.total_vectors > 0
    st.markdown(f'<div style="display:flex; justify-content:flex-end; margin-bottom:8px;">{_mode_badge(has_rag)}</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown(f"""
        <div class="hero">
            <h2>✨ 多模态 RAG 智能助手</h2>
            <p>基于 CLIP + FAISS + Qwen2.5，支持图文语义检索与智能问答。<br/>
            {'📚 知识库已就绪' if has_rag else '💡 纯 LLM 模式，上传文档启用 RAG'}</p>
            <div style="display:flex; gap:1rem; justify-content:center; max-width:780px; margin:0 auto;">
                <div class="quick-card" style="flex:1;"><div class="icon">📤</div><div class="title">构建知识库</div><div class="desc">上传文档到左侧边栏<br/>自动向量化入库</div></div>
                <div class="quick-card" style="flex:1;"><div class="icon">{'🔍' if has_rag else '🤖'}</div><div class="title">{'RAG 检索增强' if has_rag else 'LLM 自由对话'}</div><div class="desc">{'检索 + LLM 生成' if has_rag else '直接与 Qwen2.5-1.5B 对话'}</div></div>
                <div class="quick-card" style="flex:1;"><div class="icon">🖼</div><div class="title">多模态检索</div><div class="desc">文本 & 图片混合入库<br/>CLIP 跨模态匹配</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"): st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])
                if msg.get("results"):
                    with st.expander(f"📎 检索来源 ({len(msg['results'])} 条)", expanded=False):
                        for r in msg["results"]: _render_result_card(r)

    st.markdown("<br>", unsafe_allow_html=True)
    q = st.chat_input("输入问题，基于知识库回答…" if has_rag else "直接向 LLM 提问…")
    if q:
        if has_rag: _handle_rag_chat(q)
        else: _handle_pure_llm_chat(q)

# ── TAB 2: 知识库浏览 ──────────────────────────────────────
with tab_browse:
    if pipe.total_vectors == 0:
        st.info("📭 知识库为空，请先在侧边栏上传文档。")
    else:
        st.markdown(f"### 📚 知识库总览 · {pipe.total_vectors} 条向量")
        cq, ck = st.columns([3, 1])
        with cq: browse_query = st.text_input("输入关键词检索", key="browse_input")
        with ck: browse_k = st.number_input("Top-K", 1, 50, Config.TOP_K, key="browse_topk")
        if browse_query:
            results = pipe.search(browse_query, top_k=browse_k)
            if results:
                st.caption(f"检索到 {len(results)} 条结果")
                st.bar_chart({r.source[:25]: r.score for r in results}, use_container_width=True)
                for r in results: _render_result_card(r)
            else: st.info("未找到相关内容。")

# ── TAB 3: 技术仪表盘 ──────────────────────────────────────
with tab_tech:
    st.markdown("### 🔬 向量库 & 检索引擎技术仪表盘")

    if pipe.total_vectors == 0:
        st.info("📭 知识库为空。请先在侧边栏上传文档，技术指标将在入库后自动展示。")
        # 仍展示 CLIP 模型基础信息
        with st.expander("🧠 CLIP 嵌入模型规格", expanded=True):
            st.markdown(f"""
| 属性 | 值 |
|---|---|
| 模型 | `{Config.MODEL_NAME}` |
| 向量维度 | {pipe.embedder.dim} |
| 设备 | `cpu` |
| 最大文本长度 | {Config.MAX_TEXT_LENGTH} |
| 架构 | ViT-B/32 (Vision Transformer) |
| 嵌入空间 | L2 归一化 → 余弦相似度 |
            """)
    else:
        # ════ 第一行：核心指标 ─═══
        st.markdown("#### ⚡ 核心指标")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("📦 向量总数", pipe.total_vectors)
        m2.metric("📐 维度", pipe.embedder.dim)
        m3.metric("💾 索引类型", pipe.store.index_type.upper())
        # 估算内存
        vector_mem_mb = pipe.total_vectors * pipe.embedder.dim * 4 / (1024 * 1024)
        m4.metric("🧮 向量内存", f"{vector_mem_mb:.1f} MB")
        m5.metric("🌐 LLM 状态", "🟣" if pipe.llm_healthy else "⚫")

        st.markdown("---")

        # ════ 第二行：向量特征可视化 + 相似度分布 ─═══
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### 🔢 向量特征抽样 (前 20 维)")
            st.caption("从知识库随机抽取一条向量，展示其高维特征。CLIP 向量经过 L2 归一化。")

            raw = [
                r for r in (pipe.store.search(
                    pipe.embedder.text_to_vector("sample query for random"), top_k=1
                ) if pipe.total_vectors > 0 else [])
            ]
            if not raw:
                # Fallback: generate a sample vector from CLIP directly
                sample_vec = pipe.embedder.text_to_vector("人工智能与机器学习")
                st.info("使用 CLIP 实时生成的样例向量（知识库中无可检索向量）")
            else:
                # Attempt to get actual stored vector dimensions via re-encoding
                sample_vec = pipe.embedder.text_to_vector(raw[0].metadata.get("text", "sample")[:200])

            # Show first 20 dims as bar chart
            dims = min(20, len(sample_vec))
            chart_data = {f"d{i}": float(sample_vec[i]) for i in range(dims)}
            st.bar_chart(chart_data, use_container_width=True, height=220)

            # 统计
            st.markdown(f"""
            <table class="kv-table">
            <tr><td>均值 μ</td><td>{np.mean(sample_vec):.6f}</td></tr>
            <tr><td>标准差 σ</td><td>{np.std(sample_vec):.6f}</td></tr>
            <tr><td>L2 范数</td><td>{np.linalg.norm(sample_vec):.6f}</td></tr>
            <tr><td>min / max</td><td>{np.min(sample_vec):.4f} / {np.max(sample_vec):.4f}</td></tr>
            <tr><td>非零维度</td><td>{np.count_nonzero(sample_vec)} / {pipe.embedder.dim}</td></tr>
            </table>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown("#### 📊 语义相似度抽样测试")
            st.caption("用预设测试词检索知识库，观察相似度分布。")

            test_words = ["人工智能", "深度学习", "自然语言处理", "计算机视觉", "数据科学"]
            all_scores = []

            for w in test_words:
                res = pipe.search(w, top_k=5)
                for r in res:
                    all_scores.append(r.score)

            if all_scores:
                st.caption(f"**{len(all_scores)}** 次检索结果 · 中位数: **{np.median(all_scores):.3f}** · 最高: **{np.max(all_scores):.3f}**")
                # Histogram buckets
                hist, edges = np.histogram(all_scores, bins=10, range=(0, 1))
                hist_data = {f"{edges[i]:.1f}-{edges[i+1]:.1f}": int(hist[i]) for i in range(len(hist))}
                st.bar_chart(hist_data, use_container_width=True, height=220)

            st.markdown("##### 🎯 各测试词最高分")
            for w in test_words:
                res = pipe.search(w, top_k=3)
                top = res[0].score if res else 0
                st.markdown(f'{_score_bar(top, 200)} `{top:.3f}` — {w}')

        st.markdown("---")

        # ════ 第三行：检索引擎架构图 + 延迟 ─═══
        st.markdown("#### 🔗 检索引擎架构")

        arch_col1, arch_col2 = st.columns([2, 1])

        with arch_col1:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;font-size:.85rem;padding:1rem;background:rgba(20,10,40,.4);border-radius:12px;border:1px solid rgba(147,51,234,.12);">
                <span style="color:#c084fc;">👤 用户输入</span>
                <span style="color:#5b21b6;">→</span>
                <span style="background:rgba(139,92,246,.15);padding:4px 12px;border-radius:8px;">🔤 CLIP Tokenizer</span>
                <span style="color:#5b21b6;">→</span>
                <span style="background:rgba(139,92,246,.15);padding:4px 12px;border-radius:8px;">🧠 CLIP Text Encoder</span>
                <span style="color:#5b21b6;">→</span>
                <span style="background:rgba(139,92,246,.15);padding:4px 12px;border-radius:8px;">📐 L2 Normalize</span>
                <span style="color:#5b21b6;">→</span>
                <span style="background:rgba(139,92,246,.15);padding:4px 12px;border-radius:8px;">🔍 FAISS FlatIP</span>
                <span style="color:#5b21b6;">→</span>
                <span style="background:rgba(139,92,246,.15);padding:4px 12px;border-radius:8px;">📋 Top-K 召回</span>
                <span style="color:#5b21b6;">→</span>
                <span style="background:rgba(139,92,246,.15);padding:4px 12px;border-radius:8px;">🧩 Context 拼接</span>
                <span style="color:#5b21b6;">→</span>
                <span style="background:rgba(139,92,246,.15);padding:4px 12px;border-radius:8px;">🤖 LLM 生成</span>
                <span style="color:#5b21b6;">→</span>
                <span style="color:#c084fc;">💬 回答</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### ⏱ 检索延迟基准")
            # Quick benchmark
            bench_times = []
            for w in test_words[:3]:
                t0 = time.time()
                pipe.search(w, top_k=5)
                bench_times.append((time.time() - t0) * 1000)
            bench_c1, bench_c2, bench_c3 = st.columns(3)
            bench_c1.metric("平均延迟", f"{np.mean(bench_times):.1f} ms")
            bench_c2.metric("P50", f"{np.median(bench_times):.1f} ms")
            bench_c3.metric("P95", f"{np.percentile(bench_times, 95):.1f} ms")

        with arch_col2:
            st.markdown("##### 🔑 FAISS 索引参数")
            st.markdown(f"""
            <table class="kv-table">
            <tr><td>索引算法</td><td>IndexFlatIP</td></tr>
            <tr><td>相似度度量</td><td>Inner Product (IP)</td></tr>
            <tr><td>等价于</td><td>余弦相似度</td></tr>
            <tr><td>向量总数</td><td>{pipe.total_vectors}</td></tr>
            <tr><td>维度</td><td>{pipe.embedder.dim}</td></tr>
            <tr><td>搜索复杂度</td><td>O(N·{pipe.embedder.dim})</td></tr>
            <tr><td>是否精确</td><td>✅ 精确检索</td></tr>
            <tr><td>是否需训练</td><td>❌ 无需训练</td></tr>
            </table>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ════ 第四行：文档块统计 ─═══
        st.markdown("#### 📏 文档分块统计")

        # Sample chunks via search
        sample_results = pipe.search("人工智能 深度学习 机器学习", top_k=min(20, pipe.total_vectors))
        chunk_lengths = [len(r.text) for r in sample_results]
        doc_types = {}
        for r in sample_results:
            dt = r.doc_type or "text"
            doc_types[dt] = doc_types.get(dt, 0) + 1

        stat_c1, stat_c2, stat_c3 = st.columns(3)
        with stat_c1:
            st.markdown("##### 块长度分布")
            if chunk_lengths:
                st.bar_chart({f"#{i}": l for i, l in enumerate(chunk_lengths[:15])}, use_container_width=True, height=180)
                st.caption(f"平均: {np.mean(chunk_lengths):.0f} 字 · 中位: {np.median(chunk_lengths):.0f} 字 · 最长: {np.max(chunk_lengths)} 字")
        with stat_c2:
            st.markdown("##### 文档类型分布")
            st.bar_chart(doc_types, use_container_width=True, height=180)
        with stat_c3:
            st.markdown("##### 系统规格")
            import platform
            st.markdown(f"""
            <table class="kv-table">
            <tr><td>CLIP</td><td>ViT-B/32</td></tr>
            <tr><td>FAISS</td><td>FlatIP · CPU</td></tr>
            <tr><td>LLM</td><td>{gen.model}</td></tr>
            <tr><td>块大小</td><td>{Config.CHUNK_SIZE} 字</td></tr>
            <tr><td>重叠</td><td>{Config.CHUNK_OVERLAP} 字</td></tr>
            <tr><td>Top-K</td><td>{Config.TOP_K}</td></tr>
            <tr><td>设备</td><td>{platform.node()[:20]}</td></tr>
            </table>
            """, unsafe_allow_html=True)
