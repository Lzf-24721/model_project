"""
检索引擎模块测试 — F5 可调试

按 F5 (或 python test_retriever.py) 直接运行。
"""
import sys
from pathlib import Path

import numpy as np

# ── F5 调试路径 ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.embedding.clip_model import CLIPEmbedder
from src.vectordb.store import VectorStore
from src.retriever.engine import Retriever, RetrievedChunk

_HERE = Path(__file__).resolve().parent

print("=" * 55)
print("检索引擎模块测试")
print("=" * 55)

# ── 1. 初始化依赖 ──────────────────────────────────────────
print("\n── 1. 初始化嵌入器 + 向量库 ──")
embedder = CLIPEmbedder()
store = VectorStore(dim=embedder.dim, persist_dir=str(_HERE / "test_retriever_db"))

# 灌入模拟数据：文本 + 图片描述
print("\n── 2. 灌入测试数据 ──")
documents = [
    {"text": "CLIP（Contrastive Language-Image Pre-training）是OpenAI提出的多模态模型，通过对比学习将图文映射到同一向量空间。", "source": "clip_intro.txt", "type": "text"},
    {"text": "FAISS是Facebook AI Research开发的高效向量相似度搜索库，支持多种索引类型如Flat、IVF、HNSW。", "source": "faiss_guide.txt", "type": "text"},
    {"text": "RAG（检索增强生成）系统结合了信息检索和语言模型生成，先检索相关知识再让LLM基于此生成答案。", "source": "rag_paper.txt", "type": "text"},
    {"text": "[图片] 神经网络架构图 | 尺寸: 800x600", "source": "nn_arch.png", "type": "image"},
    {"text": "Transformer架构由Vaswani等人在2017年提出，核心是自注意力机制，广泛应用于NLP和CV领域。", "source": "transformer.txt", "type": "text"},
    {"text": "[图片] CLIP模型结构示意图 | 尺寸: 1200x800", "source": "clip_diagram.jpg", "type": "image"},
    {"text": "对比学习的核心思想是拉近正样本对、推远负样本对，CLIP使用InfoNCE损失函数进行优化。", "source": "contrastive_learning.txt", "type": "text"},
    {"text": "向量数据库（Vector Database）专门存储和检索高维向量嵌入，广泛应用于语义搜索、推荐系统等领域。", "source": "vectordb_intro.txt", "type": "text"},
]

texts = [d["text"] for d in documents]
vectors = embedder.texts_to_vectors(texts)
metadatas = [
    {"text": d["text"], "source": d["source"], "type": d["type"]}
    for d in documents
]
ids = store.add(vectors, metadatas)
print(f"  灌入 {len(ids)} 条数据, store.count={store.count}")

# ── 3. 创建检索引擎 ──────────────────────────────────────
print("\n── 3. 检索引擎 ──")
retriever = Retriever(embedder, store)
print(f"  dim={retriever.dim}, total_indexed={retriever.total_indexed}")

# ── 4. 文本检索 ──────────────────────────────────────────
print("\n── 4. 文本语义检索 ──")

# 4a. CLIP 相关查询
query1 = "CLIP模型是什么？"
results1 = retriever.search(query1, top_k=3)
print(f"  查询: '{query1}'")
for r in results1:
    print(f"    [{r.score:.3f}] {r.source:30s} | {r.text[:60]}...")

# 4b. 向量数据库查询
query2 = "如何进行高效向量检索？"
results2 = retriever.search(query2, top_k=3)
print(f"\n  查询: '{query2}'")
for r in results2:
    print(f"    [{r.score:.3f}] {r.source:30s} | {r.text[:60]}...")

# 4c. 带阈值过滤
results3 = retriever.search(query1, top_k=10, min_score=0.15)
print(f"\n  查询: '{query1}' (min_score=0.15) → {len(results3)} 条")

# ── 5. 上下文拼接 ──────────────────────────────────────────
print("\n── 5. 上下文拼接 ──")
context = retriever.build_context(results1, max_chunks=3)
print(f"  {context[:300]}...")

# ── 6. 构造 messages ───────────────────────────────────────
print("\n── 6. 构造 LLM messages ──")
messages = retriever.build_messages(query1, results1)
print(f"  system: {messages[0]['content'][:60]}...")
print(f"  user:   {messages[1]['content'][:120]}...")

# ── 7. 空查询兜底 ──────────────────────────────────────────
print("\n── 7. 边界情况 ──")
empty = retriever.search("", top_k=5)
print(f"  空查询 → {len(empty)} 条 (期望 0)")
no_result = retriever.search("量子计算机与黑洞信息悖论", top_k=3)
print(f"  无关查询 '{no_result[0].text[:50] if no_result else '无结果'}'")

# 清理
import shutil
test_db = _HERE / "test_retriever_db"
if test_db.exists():
    shutil.rmtree(test_db)

print("\n" + "=" * 55)
print("检索引擎模块测试完成！")
print("=" * 55)
