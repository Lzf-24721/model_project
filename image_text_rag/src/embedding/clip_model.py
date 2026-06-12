"""
CLIP 多模态嵌入模型 — 文本 & 图片 → 统一向量空间

原理:
  CLIP (Contrastive Language-Image Pre-training)
  通过 4 亿图文对做对比学习，让文本和图片映射到同一向量空间。

用法:
    from src.embedding.clip_model import CLIPEmbedder
    emb = CLIPEmbedder()
    v = emb.text_to_vector("hello")
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

# 全局配置导入 — 相对导入，包内/直接运行均可用
try:
    from ..config.loader import Config
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config.loader import Config

# ── 全局常量（统一读取配置，无硬编码） ──────────────────────────────────
MODEL_NAME = Config.MODEL_NAME
DEVICE = Config.DEVICE
MAX_TEXT_LENGTH = Config.MAX_TEXT_LENGTH


class CLIPEmbedder:
    """
    CLIP 多模态嵌入器（工程封装版）
    能力：
    1. 文本/图片映射至同一512维向量空间
    2. 支持单条/批量向量化，适配入库、检索全流程
    3. 输出向量已归一化，可直接用于余弦相似度检索
    4. 仅初始化加载一次模型，显存复用
    """

    def __init__(self):
        # 设置 HF 镜像（国内网络加速）
        import os
        os.environ.setdefault("HF_ENDPOINT", Config.HF_ENDPOINT)

        # 延迟导入torch与transformers，避免启动时加载重型依赖
        import torch
        from transformers import CLIPProcessor, CLIPModel

        self._device = DEVICE
        self._max_len = MAX_TEXT_LENGTH

        print(f"[CLIP嵌入器] 正在加载模型 {MODEL_NAME}，设备: {self._device} ...")
        # 加载模型与预处理工具
        self._model = CLIPModel.from_pretrained(MODEL_NAME).to(self._device)
        self._processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self._model.eval()  # 固定评估模式，关闭梯度

        # 自动读取向量输出维度
        self._dim = self._model.config.projection_dim
        print(f"[CLIP嵌入器] 模型加载完成 √ 向量维度={self._dim}")

    @property
    def dim(self) -> int:
        """对外只读属性：嵌入向量维度（ViT-B/32固定512）"""
        return self._dim

    # ====================== 文本编码模块 ======================
    def text_to_vector(self, text: str) -> np.ndarray:
        """
        单条文本生成归一化向量
        :param text: 输入文本字符串
        :return: 一维向量 shape=(dim,)
        """
        vec_batch = self.texts_to_vectors([text])
        return vec_batch[0]

    def texts_to_vectors(self, texts: List[str]) -> np.ndarray:
        """
        批量文本向量化 + 归一化
        :param texts: 文本列表
        :return: 二维数组 shape=(N, dim)，每行一条文本归一化向量
        """
        import torch

        if not texts:
            return np.empty((0, self.dim))

        # CLIP统一预处理
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_len,
        ).to(self._device)

        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)

        # 转numpy + L2归一化（CLIP检索必须步骤，修复原代码缺失逻辑）
        vec_np = text_features.pooler_output.cpu().numpy()
        norm = np.linalg.norm(vec_np, axis=-1, keepdims=True)
        norm[norm < 1e-8] = 1e-8  # 防止除0
        vec_normalized = vec_np / norm
        return vec_normalized

    # ====================== 图像编码模块 ======================
    def image_to_vector(self, image_path: str | Path) -> np.ndarray:
        """
        单张图片生成归一化向量
        :param image_path: 图片路径 str / Path 对象
        :return: 一维向量 shape=(dim,)
        """
        vec_batch = self.images_to_vectors([image_path])
        return vec_batch[0]

    def images_to_vectors(self, image_paths: List[str | Path]) -> np.ndarray:
        """
        批量图片向量化 + 归一化
        :param image_paths: 图片路径列表
        :return: 二维数组 shape=(N, dim)，每张图片归一化向量
        """
        import torch

        if not image_paths:
            return np.empty((0, self.dim))

        # 批量读取图片并统一RGB格式
        images = [Image.open(Path(p)).convert("RGB") for p in image_paths]
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)

        with torch.no_grad():
            img_features = self._model.get_image_features(**inputs)

        # 转numpy + L2归一化（核心修复点）
        vec_np = img_features.pooler_output.cpu().numpy()
        norm = np.linalg.norm(vec_np, axis=-1, keepdims=True)
        norm[norm < 1e-8] = 1e-8
        vec_normalized = vec_np / norm
        return vec_normalized