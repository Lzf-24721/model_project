"""
检索引擎 — 业务检索中间层

职责:
  1. 接收用户自然语言查询
  2. 调用 CLIPEmbedder 编码为归一化向量
  3. 调用 VectorStore 执行相似度检索
  4. 组装检索上下文 → 格式化输出

数据流:
  query (str)
    → embedder.text_to_vector(query)
    → store.search(q_vec, top_k)
    → List[SearchResult]

用法:
    from src.retriever.engine import Retriever

    retriever = Retriever(embedder, store)
    results = retriever.search("什么是CLIP?", top_k=5)
    prompt  = retriever.build_context(results)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── 延迟导入的模块类型 ──
from ..config.loader import Config

TOP_K = Config.TOP_K


@dataclass
class RetrievedChunk:
    """单条检索结果（中间层格式）"""

    text: str
    score: float
    source: str = ""
    chunk_id: str = ""
    doc_type: str = "text"  # text / image / image_text_pair
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_image(self) -> bool:
        return self.doc_type in ("image", "image_text_pair") and self.image_path is not None


class Retriever:
    """
    检索引擎 — 封装 embedding→search→format 全流程

    用法:
        from src.embedding.clip_model import CLIPEmbedder
        from src.vectordb.store import VectorStore
        from src.retriever.engine import Retriever
    """

    def __init__(self, embedder, store):
        """
        Args:
            embedder: CLIPEmbedder 实例
            store:    VectorStore 实例
        """
        self._embedder = embedder
        self._store = store

    @property
    def dim(self) -> int:
        return self._embedder.dim

    @property
    def total_indexed(self) -> int:
        return self._store.count

    # ── 核心检索 ────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        *,
        min_score: float = 0.0,
    ) -> List[RetrievedChunk]:
        """
        语义检索 — 输入自然语言，返回 top_k 最相关片段。

        Args:
            query:     用户问题 / 查询文本
            top_k:     返回结果数
            min_score: 最低相似度阈值（0~1），低于此值过滤

        Returns:
            [RetrievedChunk, ...] 按 score 降序
        """
        if not query or not query.strip():
            return []

        # 1. 文本 → 向量
        q_vec = self._embedder.text_to_vector(query.strip())

        # 2. FAISS 检索
        raw_results = self._store.search(q_vec, top_k=top_k, min_score=min_score)

        # 3. 转换为业务格式
        chunks = []
        for r in raw_results:
            meta = r.metadata or {}
            chunk = RetrievedChunk(
                text=meta.get("text", ""),
                score=r.score,
                source=meta.get("source", meta.get("filename", "")),
                chunk_id=r.id,
                doc_type=meta.get("type", "text"),
                image_path=meta.get("image_path") or meta.get("image_source"),
                metadata=meta,
            )
            chunks.append(chunk)

        return chunks

    # ── 图片检索 ────────────────────────────────────────────

    def search_by_image(
        self,
        image_path: str | Path,
        top_k: int = TOP_K,
    ) -> List[RetrievedChunk]:
        """
        以图搜文/以图搜图 — 用图片向量检索相似内容。

        Args:
            image_path: 查询图片路径
            top_k:     返回结果数

        Returns:
            [RetrievedChunk, ...] 按 score 降序
        """
        img_vec = self._embedder.image_to_vector(Path(image_path))
        raw_results = self._store.search(img_vec, top_k=top_k)
        return [
            RetrievedChunk(
                text=r.metadata.get("text", "") if r.metadata else "",
                score=r.score,
                source=r.metadata.get("source", "") if r.metadata else "",
                chunk_id=r.id,
                doc_type=r.metadata.get("type", "text") if r.metadata else "text",
                image_path=r.metadata.get("image_path", "") if r.metadata else None,
                metadata=r.metadata or {},
            )
            for r in raw_results
        ]

    # ── 上下文拼接 ──────────────────────────────────────────

    def build_context(
        self,
        chunks: List[RetrievedChunk],
        *,
        max_chunks: int = 5,
        include_scores: bool = True,
    ) -> str:
        """
        将检索结果拼接为 LLM 可读的上下文文本。

        Args:
            chunks:       检索结果列表
            max_chunks:   最多拼接的片段数
            include_scores: 是否显示相似度分数

        Returns:
            格式化上下文字符串
        """
        if not chunks:
            return "（未找到相关内容）"

        parts = []
        for i, ch in enumerate(chunks[:max_chunks], 1):
            header = f"[来源 {i}] {ch.source}"
            if include_scores:
                header += f"  (相似度: {ch.score:.3f})"
            if ch.doc_type == "image":
                header += "  📷 图片"
            elif ch.doc_type == "image_text_pair":
                header += "  🖼 图文对"
            parts.append(f"{header}\n{ch.text}")

        return "\n\n".join(parts)

    def build_messages(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        *,
        system_prompt: str | None = None,
    ) -> List[Dict[str, str]]:
        """
        构造 OpenAI 兼容的 messages 列表。

        Args:
            query:         用户问题
            chunks:        检索结果
            system_prompt: 自定义系统提示，None 则用默认

        Returns:
            [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        """
        if system_prompt is None:
            system_prompt = (
                "你是一个基于多模态知识库的问答助手。"
                "请仅根据提供的上下文回答问题。"
                "如果上下文中没有足够信息，请如实告知。"
                "回答时引用具体的来源编号。"
            )

        context = self.build_context(chunks)
        user_content = f"## 参考资料\n{context}\n\n## 问题\n{query}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
