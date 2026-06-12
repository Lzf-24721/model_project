"""
向量数据库模块 — 向量存储、检索、持久化

基于 FAISS 构建，支持:
  - Flat (精确检索) / HNSW / IVF 等多种索引
  - 元数据持久化
  - 增量增删

用法:
    from src.vectordb import VectorStore, SearchResult

    store = VectorStore(dim=512)
    ids = store.add(vectors, metadatas)
    results = store.search(query_vec, top_k=5)
    store.save()
"""

from .store import VectorStore, SearchResult
