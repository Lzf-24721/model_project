"""
文本分块器 — 将长文本按语义/固定窗口切分为重叠块

策略: 固定窗口 + 重叠（通用稳健方案，符合配置语义）
      chunk_size / chunk_overlap 均从全局 Config 读取，无硬编码
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .document import Chunk, Document

from ..common import load_config, get_logger
Config = load_config()
_log = get_logger(__name__)


class TextChunker:
    """
    文本分块器

    支持两种模式:
      1. 固定窗口 + 重叠 (默认) — chunk_size 字符，overlap 字符重叠
      2. 分隔符感知 — 在固定窗口基础上尽量在句号/换行处切分

    用法:
        chunker = TextChunker(chunk_size=300, overlap=50)
        chunks = chunker.chunk_document(doc)
    """

    # 句子边界正则（优先在这些位置断句）
    _SENTENCE_BOUNDARY = re.compile(r"[。！？\n](?=\S)")

    def __init__(
        self,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ):
        """
        :param chunk_size: 每块最大字符数，None 则读取 Config.CHUNK_SIZE
        :param overlap:    相邻块重叠字符数，None 则读取 Config.CHUNK_OVERLAP
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.overlap = overlap or Config.CHUNK_OVERLAP

        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) 必须小于 chunk_size ({self.chunk_size})"
            )

    # ── 核心切分逻辑 ──────────────────────────────────────

    def chunk_text(self, text: str) -> List[str]:
        """
        将纯文本切分为块字符串列表

        算法: 滑动窗口，优先在句子边界断句，无法找到边界则硬切
        """
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # 非最后一块时，尽量在句子边界切分
            if end < len(text):
                # 在 window 后半段找最佳断点
                search_start = max(start, end - max(self.overlap * 2, 50))
                window = text[search_start:end]
                best_cut = self._find_best_boundary(window)
                end = search_start + best_cut

            chunk = text[start:end]
            chunks.append(chunk)

            # 下一块起始位置 = end - overlap（重叠区）
            start = end - self.overlap

            # 防止无限循环
            if start >= len(text):
                break

        return chunks

    def _find_best_boundary(self, window: str) -> int:
        """
        在 window 中找最佳断句位置
        优先: 句号/感叹号/问号/换行后
        其次: 逗号/分号后
        最后: window 末尾（硬切）
        """
        # 找最后一个强边界
        best = 0
        for m in self._SENTENCE_BOUNDARY.finditer(window):
            best = m.end()
        if best > 0:
            return best
        # 找最后一个逗号/分号
        for i in range(len(window) - 1, -1, -1):
            if window[i] in "，,；;":
                return i + 1
        # 硬切
        return len(window)

    # ── 文档级切分 ──────────────────────────────────────

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """
        将单个 Document 切分为 Chunk 列表
        每个 Chunk 继承文档的 image_path 和 metadata
        """
        text_parts = self.chunk_text(doc.text)
        chunks: List[Chunk] = []
        for i, part in enumerate(text_parts):
            chunks.append(
                Chunk(
                    text=part,
                    doc_id=doc.id,
                    chunk_index=i,
                    image_path=doc.image_path,
                    metadata={
                        **doc.metadata,
                        "chunk_total": len(text_parts),
                    },
                )
            )
        return chunks

    def chunk_documents(self, docs: List[Document]) -> List[Chunk]:
        """批量切分文档 → 扁平化 Chunk 列表"""
        all_chunks: List[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
