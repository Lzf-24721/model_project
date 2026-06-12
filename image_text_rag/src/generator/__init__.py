"""
问答生成模块 — LLM 调用层

用法:
    from src.generator import Generator

    gen = Generator()
    answer = gen.answer("什么是CLIP?", chunks)
"""

from .llm import Generator, ROLE_DESCRIPTION
