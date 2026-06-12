"""
配置中心

用法:
    from src.config.loader import CONFIG, Config
    print(Config.DEVICE)
    print(CONFIG["embedding"]["model_name"])
"""
from __future__ import annotations

from typing import Any

CONFIG: dict[str, Any] = {
    "project": {
        "name": "multimodal_rag",
        "version": "1.0.0",
    },
    "embedding": {
        "model_name": "openai/clip-vit-base-patch32",
        "device": "cpu",
        "batch_size": 8,
        "max_text_length": 77,
        "hf_endpoint": "https://hf-mirror.com",
        "onnx": {
            "enabled": True,
            "cache_dir": "data/onnx",
            "quantize": True,
            "intra_op_threads": 4,
        },
    },
    "document": {
        "chunk_size": 300,
        "chunk_overlap": 50,
    },
    "vector_db": {
        "index_type": "flat",
        "persist_dir": "data/vector_db",
    },
    "retrieval": {
        "top_k": 5,
    },
    "llm": {
        "provider": "vllm",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "temperature": 0.1,
        "max_tokens": 512,
    },
    "vllm": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "host": "127.0.0.1",
        "port": 8000,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 4096,
        "dtype": "auto",
        "trust_remote_code": True,
        "enforce_eager": False,
        "max_num_seqs": 16,
    },
    "ui": {
        "title": "多模态 RAG 图文检索",
        "page_icon": "🔍",
        "layout": "wide",
        "max_upload_size_mb": 200,
        "show_similarity_chart": True,
    },
}


def get(key: str, default: Any = None) -> Any:
    """获取配置项，支持点号路径: get("embedding.model_name")"""
    keys = key.split(".")
    value = CONFIG
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k, default)
        else:
            return default
    return value


# ── 属性式访问（兼容 from config import Config 的写法）─

class _ConfigNamespace:
    """将 CONFIG dict 展平为属性访问"""

    def __init__(self, data: dict):
        for k, v in data.items():
            if isinstance(v, dict):
                setattr(self, k, _ConfigNamespace(v))
            else:
                setattr(self, k, v)

    def __repr__(self):
        items = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Config({items})"

    # 展平到顶层: Config.MODEL_NAME / Config.EMBEDDING_DEVICE
    # 仅当 short_key 不存在时才设，避免同名 key 后写覆盖前写
    def _flatten(self, data: dict, prefix: str = "", _seen: set | None = None):
        if _seen is None:
            _seen = set()
        for k, v in data.items():
            full_key = f"{prefix}{k}".upper() if prefix else k.upper()
            short_key = k.upper()
            if isinstance(v, dict):
                self._flatten(v, f"{k}_", _seen)
            else:
                setattr(self, full_key, v)
                if short_key not in _seen and not hasattr(self, short_key):
                    setattr(self, short_key, v)
                    _seen.add(short_key)


# 展平 + 嵌套两层访问: Config.DEVICE 或 Config.EMBEDDING_DEVICE
_config_root = _ConfigNamespace(CONFIG)
_config_root._flatten(CONFIG)

# 类型提示
class ConfigType:
    MODEL_NAME: str = CONFIG["embedding"]["model_name"]
    DEVICE: str = CONFIG["embedding"]["device"]
    BATCH_SIZE: int = CONFIG["embedding"]["batch_size"]
    MAX_TEXT_LENGTH: int = CONFIG["embedding"]["max_text_length"]
    TOP_K: int = CONFIG["retrieval"]["top_k"]

Config = _config_root  # type: ignore[assignment]
