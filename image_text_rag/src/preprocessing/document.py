"""
文档/分块数据结构

Document: 原始文档（文本 + 可选图片路径）
Chunk:   分块后的最小检索单元，携带溯源元数据
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    """原始文档 —— 加载器产出物"""

    text: str
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Chunk:
    """分块 —— 入库/检索的最小单元"""

    text: str
    doc_id: str
    chunk_index: int = 0
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.doc_id}_chunk_{self.chunk_index}"
