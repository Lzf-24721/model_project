"""
CLIP 嵌入模块测试 — F5 可调试

按 F5（或 python test_clip.py）直接运行本文件即可测试 CLIP 模型。
"""
import sys
from pathlib import Path

# ── F5 调试：将项目根目录加入 sys.path ──
# 保证 from src.embedding.clip_model 在任何运行方式下都能找到包
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.embedding.clip_model import CLIPEmbedder

# ── 当前目录（方便拼接测试图片路径） ──
_HERE = Path(__file__).resolve().parent

# 初始化嵌入器（全局仅加载一次模型）
print("=" * 50)
print("开始测试 CLIP 嵌入模块")
print("=" * 50)
embedder = CLIPEmbedder()

# 1. 测试单文本编码
text = "可爱3D潮玩盲盒手办"
text_vec = embedder.text_to_vector(text)
print(f"\n✅ 1. 单文本编码: {text_vec.shape}  期望 (512,)")
print(f"\n✅ 1*. 单文本编码: {text_vec[:10]}  期望 (512,)")
# 2. 批量文本编码
text_batch = ["猫咪插画", "机械3D模型", "可爱Q版人偶"]
batch_text_vec = embedder.texts_to_vectors(text_batch)
print(f"✅ 2. 批量文本编码: {batch_text_vec.shape}  期望 (3, 512)")

# 3. 单张图片编码
img_path = _HERE / "test.jpg"
if img_path.exists():
    img_vec = embedder.image_to_vector(img_path)
    print(f"✅ 3. 单图片编码: {img_vec.shape}  期望 (512,)")
else:
    print(f"⚠ 3. 跳过 — 测试图片不存在: {img_path}")

# 4. 批量图片编码
img_batch = [_HERE / "test.jpg"]
# 只测存在的文件
img_batch = [p for p in img_batch if p.exists()]
if img_batch:
    batch_img_vec = embedder.images_to_vectors(img_batch)
    print(f"✅ 4. 批量图片编码: {batch_img_vec.shape}  期望 (1, 512)")
else:
    print("⚠ 4. 跳过 — 无可用测试图片")

print("\n" + "=" * 50)
print("测试完成！嵌入维度:", embedder.dim)
print("=" * 50)
