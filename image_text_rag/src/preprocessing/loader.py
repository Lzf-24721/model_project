"""
文档加载器 — 从文件/目录读取原始文档

支持的格式:
  - 纯文本: .txt, .md, .markdown
  - 图片:   .jpg, .jpeg, .png, .webp, .bmp (作为纯图片文档)
  - 图文对: 同目录下同名不同后缀的 txt+图片 自动配对
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .document import Document

# ── 支持的扩展名 ──
_TEXT_EXTS = {".txt", ".md", ".markdown"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class DocumentLoader:
    """
    文档加载器

    用法:
        loader = DocumentLoader()
        docs = loader.load_directory("data/docs")       # 加载目录下所有文档
        doc  = loader.load_text("data/readme.txt")      # 加载单个文本
        doc  = loader.load_image_text_pair("img.jpg", "caption.txt")
    """

    # ── 单文件加载 ──────────────────────────────────────────

    def load_text(self, file_path: str | Path) -> Document:
        """
        加载纯文本文件

        :param file_path: 文本文件路径 (.txt / .md)
        :return: Document（仅有 text，无 image_path）
        :raises FileNotFoundError, UnicodeDecodeError
        """
        fp = Path(file_path)
        if not fp.exists():
            raise FileNotFoundError(f"文件不存在: {fp}")
        text = fp.read_text(encoding="utf-8")
        return Document(
            text=text,
            metadata={"source": str(fp.resolve()), "type": "text", "filename": fp.name},
        )

    def load_image(self, file_path: str | Path) -> Document:
        """
        加载图片作为独立文档（文本字段记录图片路径）

        :param file_path: 图片文件路径
        :return: Document（image_path 指向图片，text 为路径描述）
        """
        fp = Path(file_path)
        if not fp.exists():
            raise FileNotFoundError(f"图片不存在: {fp}")
        return Document(
            text=f"[图片] {fp.name}",
            image_path=str(fp.resolve()),
            metadata={
                "source": str(fp.resolve()),
                "type": "image",
                "filename": fp.name,
                "suffix": fp.suffix.lower(),
            },
        )

    def load_image_text_pair(
        self, image_path: str | Path, text_path: str | Path
    ) -> Document:
        """
        显式指定图文对

        :param image_path: 图片路径
        :param text_path: 文本描述路径
        :return: 包含 text + image_path 的 Document
        """
        img_fp = Path(image_path)
        txt_fp = Path(text_path)
        if not img_fp.exists():
            raise FileNotFoundError(f"图片不存在: {img_fp}")
        if not txt_fp.exists():
            raise FileNotFoundError(f"文本不存在: {txt_fp}")
        text = txt_fp.read_text(encoding="utf-8")
        return Document(
            text=text,
            image_path=str(img_fp.resolve()),
            metadata={
                "source": str(txt_fp.resolve()),
                "image_source": str(img_fp.resolve()),
                "type": "image_text_pair",
                "filename": txt_fp.name,
                "image_filename": img_fp.name,
            },
        )

    # ── 目录批量加载 ──────────────────────────────────────────

    def load_directory(
        self,
        dir_path: str | Path,
        *,
        recursive: bool = True,
        auto_pair: bool = True,
    ) -> List[Document]:
        """
        扫描目录，自动加载所有文本/图片文件，并尝试图文配对

        :param dir_path:   目标目录
        :param recursive:  是否递归子目录
        :param auto_pair:  是否自动将同名不同后缀的 txt+图片 配对为图文文档
        :return: 文档列表
        """
        root = Path(dir_path)
        if not root.is_dir():
            raise NotADirectoryError(f"目录不存在: {root}")

        text_files: Dict[str, Path] = {}  # stem → path
        image_files: Dict[str, Path] = {}  # stem → path
        paired: set = set()  # 已配对 stem 集合
        docs: List[Document] = []

        # 1. 扫描收集
        pattern = "**/*" if recursive else "*"
        for fp in root.glob(pattern):
            if not fp.is_file():
                continue
            suffix = fp.suffix.lower()
            stem = fp.stem
            if suffix in _TEXT_EXTS:
                text_files[stem] = fp
            elif suffix in _IMAGE_EXTS:
                image_files[stem] = fp

        # 2. 自动配对（同名 stem）
        if auto_pair:
            common = set(text_files.keys()) & set(image_files.keys())
            for stem in common:
                txt_fp = text_files[stem]
                img_fp = image_files[stem]
                text = txt_fp.read_text(encoding="utf-8")
                docs.append(
                    Document(
                        text=text,
                        image_path=str(img_fp.resolve()),
                        metadata={
                            "source": str(txt_fp.resolve()),
                            "image_source": str(img_fp.resolve()),
                            "type": "image_text_pair",
                            "filename": txt_fp.name,
                            "image_filename": img_fp.name,
                        },
                    )
                )
                paired.add(stem)

        # 3. 未配对的文本 → 纯文本文档
        for stem, fp in text_files.items():
            if stem not in paired:
                docs.append(self.load_text(fp))

        # 4. 未配对的图片 → 纯图片文档
        for stem, fp in image_files.items():
            if stem not in paired:
                docs.append(self.load_image(fp))

        return docs

    def load_texts(self, file_paths: List[str | Path]) -> List[Document]:
        """批量加载多个文本文件"""
        return [self.load_text(p) for p in file_paths]
