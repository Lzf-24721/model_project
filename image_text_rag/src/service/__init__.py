"""
业务编排层 — RAG 全链路编排

用法:
    from src.service.pipeline import RAGPipeline
    pipeline = RAGPipeline(embedder, store, generator, chunker, img_proc)
    pipeline.ingest_text(text, "source.txt")
    pipeline.ingest_image(img_bytes, "photo.png")
    answer = pipeline.answer("什么是CLIP?")
"""
from .pipeline import RAGPipeline
