"""
向量数据库模块测试 — F5 可调试

按 F5（或 python test_vectordb.py）直接运行。
"""
import sys
from pathlib import Path

import numpy as np

# ── F5 调试路径 ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.vectordb import VectorStore, SearchResult

_HERE = Path(__file__).resolve().parent

print("=" * 55)
print("向量数据库模块测试")
print("=" * 55)

# ── 1. 创建向量库 ──────────────────────────────────────────
print("\n── 1. 创建 VectorStore ──")
store = VectorStore(dim=512, persist_dir=str(_HERE / "test_db"))
print(f"  dim={store.dim}, index_type={store.index_type}, count={store.count}")

# ── 2. 添加向量 ────────────────────────────────────────────
print("\n── 2. 添加向量 ──")
N = 10
np.random.seed(42)
vectors = np.random.randn(N, 512).astype(np.float32)
# L2 归一化（模拟 CLIP 输出）
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

metadatas = [
    {"text": f"文档{i}的内容文本", "source": f"doc_{i}.txt", "chunk_index": 0}
    for i in range(N)
]
ids = store.add(vectors, metadatas)
print(f"  添加 {len(ids)} 条向量，count={store.count}")
print(f"  前3个 ID: {ids[:3]}")

# ── 3. 检索 ────────────────────────────────────────────────
print("\n── 3. 向量检索 ──")

# 用第一条向量作为查询（应找到自己，相似度≈1.0）
query = vectors[0].copy()
results = store.search(query, top_k=3)
print(f"  查询 top_k=3:")
for r in results:
    print(f"    id={r.id[:8]}...  score={r.score:.4f}  meta={r.metadata}")

# 带阈值过滤
results_filtered = store.search(query, top_k=10, min_score=0.5)
print(f"  阈值过滤(min_score=0.5): {len(results_filtered)} 条结果")

# 不相关查询
random_query = np.random.randn(512).astype(np.float32)
random_query = random_query / np.linalg.norm(random_query)
results_random = store.search(random_query, top_k=3)
print(f"  随机查询 top_k=3 最高分: {results_random[0].score:.4f}")

# ── 4. 批量检索 ────────────────────────────────────────────
print("\n── 4. 批量检索 ──")
batch_queries = vectors[:3]
batch_results = store.batch_search(batch_queries, top_k=2)
for i, res in enumerate(batch_results):
    print(f"  查询{i}: top={[(r.id[:6], f'{r.score:.3f}') for r in res]}")

# ── 5. 删除 ────────────────────────────────────────────────
print("\n── 5. 删除 ──")
removed = store.delete([ids[0], ids[1]])
print(f"  删除 {removed} 条, count={store.count}")

# 验证删除后检索
results_after_del = store.search(query, top_k=5)
print(f"  删除后检索 top_k=5: {len(results_after_del)} 条")

# ── 6. 持久化 ──────────────────────────────────────────────
print("\n── 6. 持久化 ──")
store.save()

# 新建实例加载
store2 = VectorStore(dim=512, persist_dir=str(_HERE / "test_db"))
loaded = store2.load()
print(f"  加载成功: {loaded}, count={store2.count}")

# 验证加载后检索一致
results2 = store2.search(query, top_k=3)
print(f"  加载后检索: {len(results2)} 条")
for r in results2:
    print(f"    id={r.id[:8]}...  score={r.score:.4f}")

# ── 7. 清空 ────────────────────────────────────────────────
print("\n── 7. 清空 ──")
store2.clear()
print(f"  count={store2.count}")
results_empty = store2.search(query, top_k=5)
print(f"  清空后检索: {len(results_empty)} 条（期望 0）")

# 清理测试数据
import shutil
test_db = _HERE / "test_db"
if test_db.exists():
    shutil.rmtree(test_db)

print("\n" + "=" * 55)
print("向量数据库模块测试完成！")
print("=" * 55)
