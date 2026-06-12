"""
图片预处理器 — 验证、读取、格式规范化

用于在 embedding 之前对图片做统一预处理:
  - 格式验证（仅允许常见图片格式）
  - RGB 转换
  - 尺寸信息提取
  - 损坏检测
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image, UnidentifiedImageError

# ── 允许的图片格式 ──
_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


class ImagePreprocessor:
    """
    图片预处理器

    用法:
        proc = ImagePreprocessor()
        img  = proc.load("photo.jpg")               # 单张
        imgs = proc.batch_load(["a.jpg", "b.png"])  # 批量
        ok   = proc.validate("photo.jpg")           # 有效性检查
    """

    def __init__(self, target_mode: str = "RGB"):
        """
        :param target_mode: 目标色彩模式，默认 RGB（CLIP 要求）
        """
        self.target_mode = target_mode

    # ── 验证 ──────────────────────────────────────────────

    def validate(self, image_path: str | Path) -> bool:
        """
        快速验证图片是否可读

        :return: True 表示文件存在、格式合法、可正常打开
        """
        fp = Path(image_path)
        if not fp.exists():
            return False
        if fp.suffix.lower() not in _ALLOWED_EXTS:
            return False
        try:
            with Image.open(fp) as img:
                img.verify()
            return True
        except (UnidentifiedImageError, OSError):
            return False

    def get_info(self, image_path: str | Path) -> dict:
        """
        获取图片元信息（不加载像素）

        :return: {"width": int, "height": int, "format": str, "mode": str, "size_bytes": int}
        """
        fp = Path(image_path)
        with Image.open(fp) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "size_bytes": fp.stat().st_size,
            }

    # ── 加载 ──────────────────────────────────────────────

    def load(self, image_path: str | Path) -> Image.Image:
        """
        加载单张图片并转为目标模式

        :return: PIL.Image（已 convert 为 target_mode）
        :raises FileNotFoundError, UnidentifiedImageError
        """
        fp = Path(image_path)
        if not fp.exists():
            raise FileNotFoundError(f"图片不存在: {fp}")
        if fp.suffix.lower() not in _ALLOWED_EXTS:
            raise ValueError(f"不支持的图片格式: {fp.suffix}，允许: {_ALLOWED_EXTS}")
        img = Image.open(fp)
        if img.mode != self.target_mode:
            img = img.convert(self.target_mode)
        return img

    def batch_load(self, image_paths: List[str | Path]) -> List[Image.Image]:
        """
        批量加载图片

        :return: PIL.Image 列表，失败的项以 None 占位
        """
        results: List[Image.Image] = []
        for p in image_paths:
            try:
                results.append(self.load(p))
            except (FileNotFoundError, ValueError, UnidentifiedImageError) as e:
                print(f"[ImagePreprocessor] 加载失败 {p}: {e}")
                results.append(None)
        return results

    # ── 尺寸控制 ──────────────────────────────────────────

    def resize(
        self,
        image: str | Path | Image.Image,
        max_size: int = 512,
    ) -> Image.Image:
        """
        等比缩放，长边不超过 max_size（默认 512，匹配 CLIP ViT-B/32 输入）

        :param image: 图片路径或 PIL.Image
        :param max_size: 长边最大像素
        :return: 缩放后的 PIL.Image
        """
        if isinstance(image, (str, Path)):
            img = self.load(image)
        else:
            img = image

        w, h = img.size
        if max(w, h) <= max_size:
            return img
        ratio = max_size / max(w, h)
        return img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    def compress(
        self,
        image: str | Path | Image.Image,
        quality: int = 85,
        format: str = "JPEG",
    ) -> Image.Image:
        """
        有损压缩 — 减小图片体积，保留视觉质量。

        :param image:   图片路径或 PIL.Image
        :param quality: JPEG 质量 (1-100, 默认 85)
        :param format:  输出格式 (JPEG/PNG/WebP)
        :return: 压缩后的 PIL.Image（RGB 模式）
        """
        if isinstance(image, (str, Path)):
            img = self.load(image)
        else:
            img = image

        if img.mode != "RGB":
            img = img.convert("RGB")

        import io
        buf = io.BytesIO()
        img.save(buf, format=format, quality=quality, optimize=True)
        buf.seek(0)
        return Image.open(buf)

    def preprocess(
        self,
        image_path: str | Path,
        *,
        max_size: int = 512,
        compress_quality: int | None = 85,
    ) -> Image.Image:
        """
        一站式预处理：验证 → 加载 → 缩放 → 压缩 → RGB

        :param image_path:      图片路径
        :param max_size:        长边最大像素
        :param compress_quality: 压缩质量，None 则不压缩
        :return: 处理后的 PIL.Image
        """
        img = self.load(image_path)
        img = self.resize(img, max_size=max_size)
        if compress_quality is not None:
            img = self.compress(img, quality=compress_quality)
        return img

    # ── 批量校验 ──────────────────────────────────────────

    def filter_valid(self, image_paths: List[str | Path]) -> List[Path]:
        """从路径列表中筛出所有有效图片"""
        return [Path(p) for p in image_paths if self.validate(p)]
