"""
LLM 问答生成 — 基于检索上下文 + OpenAI 兼容 API 生成答案

支持的服务:
  - Ollama (本地):     base_url = "http://localhost:11434/v1"
  - vLLM  (本地):      base_url = "http://localhost:8000/v1"
  - OpenAI (云端):     base_url = "https://api.openai.com/v1"
  - 任意 /chat/completions 兼容服务

数据流:
  用户问题 + 检索上下文
    → build_rag_prompt(question, chunks)
    → POST /chat/completions
    → LLM 生成答案字符串

用法:
    from src.generator.llm import Generator
    gen = Generator()
    answer = gen.answer(question, chunks)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..common import load_config, get_logger
Config = load_config()
_log = get_logger(__name__)


ROLE_DESCRIPTION = (
    "你是一个基于多模态知识库的问答助手。"
    "请仅根据提供的上下文回答问题。"
    "如果上下文中没有足够信息，请如实告知。"
    "回答时引用具体的来源编号。"
)


class Generator:
    """
    LLM 问答生成器 — 组装 prompt → 调用 API → 返回答案

    用法:
        gen = Generator()
        answer = gen.answer("什么是CLIP?", chunks)
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 120,
    ):
        """
        Args:
            base_url:    API 地址，None 则读 Config.LLM_BASE_URL
            model:       模型名，None 则读 Config.LLM_MODEL
            temperature: 温度，None 则读 Config.LLM_TEMPERATURE
            max_tokens:  最大输出 token，None 则读 Config.LLM_MAX_TOKENS
            timeout:     HTTP 超时秒数
        """
        self._base_url = (base_url or Config.LLM_BASE_URL).rstrip("/")
        self._model = model or Config.LLM_MODEL
        self._temperature = temperature or Config.LLM_TEMPERATURE
        self._max_tokens = max_tokens or Config.LLM_MAX_TOKENS
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    # ── 核心生成 ────────────────────────────────────────────

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        底层 chat 调用 — 直接发送 messages 列表。

        Args:
            messages:    OpenAI 格式 [{"role":"system","content":"..."}, ...]
            temperature: 覆盖温度
            max_tokens:  覆盖最大 token

        Returns:
            模型回复文本

        Raises:
            requests.RequestException: 网络/API 错误
        """
        url = f"{self._base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }

        resp = requests.post(url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # ── RAG 回答 ────────────────────────────────────────────

    def answer(
        self,
        question: str,
        chunks: List[Any],
        *,
        additional_context: str = "",
        system_prompt: str = ROLE_DESCRIPTION,
    ) -> str:
        """
        基于检索结果生成 RAG 答案。

        Args:
            question:           用户问题
            chunks:             检索结果列表 (RetrievedChunk / dict / 任意有 text/score/source 的对象)
            additional_context: 额外上下文（如用户附加的说明）
            system_prompt:      系统角色描述

        Returns:
            LLM 生成的答案字符串
        """
        # 拼接上下文
        ctx_parts = []
        for i, ch in enumerate(chunks, 1):
            # 兼容 dict 和对象
            if isinstance(ch, dict):
                src = ch.get("source", "未知")
                txt = ch.get("text", "")
                score = ch.get("score", 0)
            else:
                src = getattr(ch, "source", "未知")
                txt = getattr(ch, "text", "")
                score = getattr(ch, "score", 0)
            ctx_parts.append(f"[来源 {i}] {src} (相似度: {score:.3f})\n{txt}")

        context_block = "\n\n".join(ctx_parts) if ctx_parts else "（无可用上下文）"

        if additional_context:
            context_block = f"{additional_context}\n\n{context_block}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"## 参考资料\n{context_block}\n\n## 问题\n{question}"},
        ]

        return self.chat(messages)

    def answer_with_raw_context(
        self,
        question: str,
        context_text: str,
        *,
        system_prompt: str = ROLE_DESCRIPTION,
    ) -> str:
        """
        直接使用文本上下文生成答案（跳过检索层）。

        Args:
            question:     用户问题
            context_text: 已拼好的上下文文本
            system_prompt: 系统角色描述

        Returns:
            LLM 生成答案
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"## 参考资料\n{context_text}\n\n## 问题\n{question}"},
        ]
        return self.chat(messages)

    # ── 流式生成（可选） ────────────────────────────────────

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """
        流式 chat 调用 — 生成器函数，逐 token yield。

        用法:
            for token in gen.chat_stream(messages):
                print(token, end="", flush=True)
        """
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "stream": True,
        }

        with requests.post(url, json=payload, stream=True, timeout=self._timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    import json
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    # ── 健康检查 ────────────────────────────────────────────

    def health_check(self) -> bool:
        """检测 LLM 服务是否可达 (GET /models)"""
        try:
            resp = requests.get(f"{self._base_url}/models", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        """
        获取服务端可用模型列表。

        Returns:
            模型 ID 列表
        """
        try:
            resp = requests.get(f"{self._base_url}/models", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [m.get("id", "") for m in data.get("data", [])]
        except requests.RequestException:
            return []
