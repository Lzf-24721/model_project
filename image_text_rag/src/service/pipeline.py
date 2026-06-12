"""
RAGPipeline — 业务编排层

职责:
  1. 持有所有核心组件 (embedder, store, retriever, generator, chunker, img_proc)
  2. 提供统一入口: ingest_text / ingest_image / answer / search / clear
  3. 从 UI 层抽离所有业务逻辑

用法:
    pipeline = RAGPipeline(embedder, store, generator, chunker, img_proc)
    pipeline.ingest_text("CLIP is a multimodal model.", "clip_intro.txt")
    answer = pipeline.answer("什么是CLIP?")
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..common import get_logger

_log = get_logger(__name__)


class RAGPipeline:
    """多模态 RAG 全链路编排器"""

    def __init__(self, embedder, store, generator, chunker, img_processor):
        from ..retriever.engine import Retriever

        self.embedder = embedder
        self.store = store
        self.generator = generator
        self.chunker = chunker
        self.img_proc = img_processor
        self.retriever = Retriever(embedder, store)

    # ── 入库 ──

    def ingest_text(self, text: str, source: str) -> int:
        """文本入库 → 返回 chunk 数, 0 表示失败"""
        chunks = self.chunker.chunk_text(text)
        if not chunks:
            return 0
        vecs = self.embedder.texts_to_vectors(chunks)
        metas = [
            {"text": c, "source": source, "type": "text", "chunk_index": i}
            for i, c in enumerate(chunks)
        ]
        self.store.add(vecs, metas)
        self.store.save()
        _log.info("文本入库: %s → %d chunks", source, len(chunks))
        return len(chunks)

    def ingest_image(self, image_bytes: bytes, filename: str) -> int:
        """图片入库 → 返回 1 成功 / 0 失败

        v2: 使用 CLIP 视觉编码器 (image_to_vector) 替代文本描述编码。
            文本查询可直接检索图片视觉内容，实现真正的跨模态检索。
        """
        ext = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as t:
            t.write(image_bytes)
            tp = t.name

        info: dict = {}
        img_vec = None
        desc = ""
        try:
            self.img_proc.preprocess(tp, max_size=512, compress_quality=85)
            info = self.img_proc.get_info(tp)
            img_vec = self.embedder.image_to_vector(tp)
            desc = f"[图片] {Path(filename).stem} | {info['width']}x{info['height']}"
        except Exception:
            img_vec = self.embedder.text_to_vector(f"[图片] {Path(filename).stem}")
            desc = f"[图片] {Path(filename).stem}  (损坏)"
        finally:
            Path(tp).unlink(missing_ok=True)

        self.store.add(
            img_vec.reshape(1, -1),
            [{"text": desc, "source": filename, "type": "image",
              "vec_type": "vision", "width": info.get("width", 0),
              "height": info.get("height", 0)}],
        )
        self.store.save()
        _log.info("图片入库(视觉向量): %s", filename)
        return 1

    # ── 检索 ──

    def search(self, query: str, top_k: int | None = None) -> List[Any]:
        return self.retriever.search(query, top_k=top_k)

    def search_by_image(self, image_bytes: bytes, top_k: int | None = None) -> List[Any]:
        """以图搜图/以图搜文 — 用图片视觉向量检索"""
        ext = ".jpg"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as t:
            t.write(image_bytes)
            tp = t.name
        try:
            results = self.retriever.search_by_image(tp, top_k=top_k)
        finally:
            Path(tp).unlink(missing_ok=True)
        return results

    # ── 问答 ──

    def answer(
        self,
        question: str,
        results: List[Any] | None = None,
    ) -> Tuple[str, List[Any], Dict[str, float]]:
        """RAG 问答 → (答案, 检索结果, 延迟统计)

        如果 results is None，自动先检索再生成。
        """
        timing = {}

        if results is None:
            t0 = time.time()
            results = self.search(question)
            timing["search_ms"] = (time.time() - t0) * 1000

        if not results:
            # 降级：纯 LLM
            try:
                t1 = time.time()
                answer = self.generator.answer_with_raw_context(
                    question,
                    "（未检索到相关文档，请基于通用知识回答）",
                    system_prompt="你是一个智能助手。请基于通用知识简要回答。",
                )
                timing["llm_ms"] = (time.time() - t1) * 1000
            except Exception as e:
                answer = f"**⚠ 调用失败**\n\n{type(e).__name__}: {e}"
                timing["llm_ms"] = 0
            return answer, [], timing

        t1 = time.time()
        try:
            answer = self.generator.answer(question, results)
        except Exception as e:
            answer = f"**⚠ LLM 调用失败**\n\n{type(e).__name__}: {e}\n\n### 检索结果\n"
            for i, r in enumerate(results[:3], 1):
                answer += f"\n**{i}.** [{r.source}] ({r.score:.2%})\n> {r.text[:200]}\n"
        timing["llm_ms"] = (time.time() - t1) * 1000

        return answer, results, timing

    def answer_pure_llm(self, question: str) -> Tuple[str, Dict[str, float]]:
        """无 RAG — 直接与 LLM 对话"""
        t0 = time.time()
        try:
            answer = self.generator.answer_with_raw_context(
                question,
                "（无私有知识库上下文，请基于通用知识回答）",
                system_prompt=(
                    "你是一个多模态 RAG 系统的智能助手。当前知识库为空，"
                    "请基于你的通用知识回答。回答简洁专业，使用 markdown 格式。"
                ),
            )
        except Exception as e:
            answer = f"**⚠ LLM 调用失败**\n\n{type(e).__name__}: {e}"
        return answer, {"llm_ms": (time.time() - t0) * 1000}

    # ── 管理 ──

    def clear(self):
        self.store.clear()
        self.store.save()
        _log.info("知识库已清空")

    @property
    def total_vectors(self) -> int:
        return self.store.count

    @property
    def llm_healthy(self) -> bool:
        return self.generator.health_check()
