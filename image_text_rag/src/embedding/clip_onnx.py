"""
CLIP ONNX 量化嵌入模型 — CPU 加速推理

原理:
  将 PyTorch CLIP 的 text_encoder / vision_encoder 导出为 ONNX，
  用 onnxruntime 做 CPU 推理，比 PyTorch CPU 快 2-3x。
  可选 int8 量化进一步压缩模型体积和推理延迟。

用法:
    from src.embedding.clip_onnx import CLIPEmbedderONNX
    emb = CLIPEmbedderONNX()
    v = emb.text_to_vector("hello")
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

from ..common import load_config, get_logger
Config = load_config()
_log = get_logger(__name__)

MODEL_NAME = Config.MODEL_NAME
MAX_TEXT_LENGTH = Config.MAX_TEXT_LENGTH
ONNX_ENABLED = Config.ONNX_ENABLED
ONNX_CACHE_DIR = Config.ONNX_CACHE_DIR
ONNX_QUANTIZE = Config.ONNX_QUANTIZE
ONNX_THREADS = Config.ONNX_INTRA_OP_THREADS
BATCH_SIZE = Config.BATCH_SIZE


class CLIPEmbedderONNX:
    """
    CLIP ONNX 嵌入器 — 专为 CPU 推理优化

    首次运行自动:
      1. 从 HuggingFace 加载 PyTorch CLIP
      2. 导出 text/vision encoder 为 ONNX
      3. 可选 int8 量化
      4. 缓存到 data/onnx/，后续加载 < 1s
    """

    def __init__(self):
        import torch
        from transformers import CLIPProcessor, CLIPModel
        import onnxruntime as ort

        # 强制离线模式 — 所有文件已在 HF 缓存中
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("HF_ENDPOINT", Config.HF_ENDPOINT)

        self._max_len = MAX_TEXT_LENGTH
        self._cache_dir = Path(ONNX_CACHE_DIR)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # ONNX 会话选项
        self._sess_opts = ort.SessionOptions()
        self._sess_opts.intra_op_num_threads = ONNX_THREADS
        self._sess_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self._sess_opts.enable_mem_reuse = False

        # 加载 / 导出 ONNX 模型
        self._text_path = self._cache_dir / "clip_text_encoder.onnx"
        self._vision_path = self._cache_dir / "clip_vision_encoder.onnx"

        if self._text_path.exists() and self._vision_path.exists():
            _log.info("从缓存加载ONNX模型")
            self._processor = CLIPProcessor.from_pretrained(
                MODEL_NAME, local_files_only=True
            )
            self._text_sess = ort.InferenceSession(
                str(self._text_path), self._sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self._vision_sess = ort.InferenceSession(
                str(self._vision_path), self._sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self._dim = self._text_sess.get_outputs()[0].shape[1]
        else:
            _log.info("首次运行 — 导出 ONNX 模型")
            pt_model = CLIPModel.from_pretrained(
                MODEL_NAME, local_files_only=True
            )
            pt_model.eval()
            self._processor = CLIPProcessor.from_pretrained(
                MODEL_NAME, local_files_only=True
            )
            self._dim = pt_model.config.projection_dim

            _log.info("导出 text_encoder...")
            self._export_text_encoder(pt_model)
            _log.info("导出 vision_encoder...")
            self._export_vision_encoder(pt_model)

            # 量化
            if ONNX_QUANTIZE:
                self._quantize_model(self._text_path)
                self._quantize_model(self._vision_path)

            # 加载 ONNX session
            self._text_sess = ort.InferenceSession(
                str(self._text_path), self._sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self._vision_sess = ort.InferenceSession(
                str(self._vision_path), self._sess_opts,
                providers=["CPUExecutionProvider"],
            )
            # 释放 PyTorch 模型
            del pt_model

        _log.info("模型就绪 dim=%d threads=%d", self._dim, ONNX_THREADS)

    # ── ONNX 导出 ──────────────────────────────────────────

    def _export_text_encoder(self, pt_model):
        import torch

        class TextWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.text_model = model.text_model
                self.text_projection = model.text_projection
            def forward(self, input_ids, attention_mask):
                out = self.text_model(input_ids=input_ids,
                                      attention_mask=attention_mask)
                pooled = out[1]  # pooler_output
                return self.text_projection(pooled)

        wrapper = TextWrapper(pt_model)
        dummy_ids = torch.zeros((1, self._max_len), dtype=torch.long)
        dummy_mask = torch.ones((1, self._max_len), dtype=torch.long)

        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_mask),
            str(self._text_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["text_features"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "text_features": {0: "batch"},
            },
            opset_version=18,
        )
        _log.info("text_encoder导出 → %s", self._text_path)

    def _export_vision_encoder(self, pt_model):
        import torch

        class VisionWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.vision_model = model.vision_model
                self.visual_projection = model.visual_projection
            def forward(self, pixel_values):
                out = self.vision_model(pixel_values=pixel_values)
                pooled = out[1]
                return self.visual_projection(pooled)

        wrapper = VisionWrapper(pt_model)
        dummy_pixels = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

        torch.onnx.export(
            wrapper,
            dummy_pixels,
            str(self._vision_path),
            input_names=["pixel_values"],
            output_names=["image_features"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "image_features": {0: "batch"},
            },
            opset_version=18,
        )
        _log.info("vision_encoder导出 → %s", self._vision_path)

    def _quantize_model(self, model_path: Path) -> None:
        """int8 动态量化"""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            q_path = model_path.with_suffix(".q.onnx")
            _log.info("量化 %s...", model_path.name)
            quantize_dynamic(
                str(model_path), str(q_path),
                weight_type=QuantType.QUInt8,
            )
            q_path.rename(model_path)
            _log.info("量化完成: %s", model_path.name)
        except ImportError:
            _log.warning("onnxruntime.quantization 未安装，跳过量化")
        except Exception as e:
            _log.warning("量化失败: %s，使用浮点模型", e)

    @property
    def dim(self) -> int:
        return self._dim

    # ── 文本编码 ──────────────────────────────────────────

    def text_to_vector(self, text: str) -> np.ndarray:
        return self.texts_to_vectors([text])[0]

    def texts_to_vectors(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        all_features = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            inputs = self._processor(
                text=batch, return_tensors="np",
                padding="max_length", truncation=True,
                max_length=self._max_len,
            )
            onnx_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
            features = self._text_sess.run(None, onnx_inputs)[0]
            all_features.append(features)

        if len(all_features) == 1:
            return self._normalize(all_features[0])
        return self._normalize(np.concatenate(all_features, axis=0))

    # ── 图像编码 ──────────────────────────────────────────

    def image_to_vector(self, image_path: str | Path) -> np.ndarray:
        return self.images_to_vectors([image_path])[0]

    def images_to_vectors(self, image_paths: List[str | Path]) -> np.ndarray:
        if not image_paths:
            return np.empty((0, self.dim), dtype=np.float32)

        images = [Image.open(Path(p)).convert("RGB") for p in image_paths]
        inputs = self._processor(images=images, return_tensors="np")
        onnx_inputs = {
            "pixel_values": inputs["pixel_values"].astype(np.float32),
        }
        features = self._vision_sess.run(None, onnx_inputs)[0]
        return self._normalize(features)

    # ── 归一化 ────────────────────────────────────────────

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        norm[norm < 1e-8] = 1e-8
        return vec / norm
