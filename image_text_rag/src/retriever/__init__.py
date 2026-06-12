"""
检索引擎模块 — 业务检索中间层

用法:
    from src.retriever import Retriever, RetrievedChunk

    retriever = Retriever(embedder, store)
    results = retriever.search("什么是CLIP?")
    context  = retriever.build_context(results)
"""

from .engine import Retriever, RetrievedChunk
