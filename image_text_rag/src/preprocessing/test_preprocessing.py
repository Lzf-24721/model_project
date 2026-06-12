"""
数据预处理模块测试 — F5 可调试

按 F5（或 python test_preprocessing.py）直接运行。
"""
import sys
from pathlib import Path

# ── F5 调试路径 ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing import DocumentLoader, TextChunker, ImagePreprocessor
from src.preprocessing import Document, Chunk

_HERE = Path(__file__).resolve().parent

print("=" * 55)
print("数据预处理模块测试")
print("=" * 55)

# ── 1. 文档加载器 ──────────────────────────────────────────
print("\n── 1. DocumentLoader ──")
loader = DocumentLoader()

# 1a. 加载纯文本
txt_path = _HERE / "test_data" / "sample.txt"
if txt_path.exists():
    doc = loader.load_text(txt_path)
    print(f"  ✅ load_text: id={doc.id[:8]}..., text_len={len(doc.text)}, meta={doc.metadata}")
else:
    print(f"  ⚠ 跳过 load_text — 测试文件不存在: {txt_path}")

# 1b. 加载目录
test_dir = _HERE / "test_data"
if test_dir.is_dir():
    docs = loader.load_directory(test_dir)
    print(f"  ✅ load_directory: 共 {len(docs)} 个文档")
    for d in docs:
        print(f"     - {d.metadata.get('type'):20s} | {d.metadata.get('filename', '?'):30s} | text={d.text[:40]}...")
else:
    print(f"  ⚠ 跳过 load_directory — 目录不存在: {test_dir}")

# ── 2. 文本分块器 ──────────────────────────────────────────
print("\n── 2. TextChunker ──")
chunker = TextChunker()  # 从 Config 读取 chunk_size=300, overlap=50
print(f"  chunk_size={chunker.chunk_size}, overlap={chunker.overlap}")

long_text = (
    "CLIP（Contrastive Language-Image Pre-training）是一种多模态模型，"
    "它通过对比学习的方式将图像和文本映射到同一个向量空间中。"
    "这使得我们可以用文本去搜索图片，也可以用图片去搜索相关文本描述。"
    "在RAG（检索增强生成）系统中，CLIP常被用作多模态嵌入模型，"
    "将图文资料编码为向量后存入向量数据库，供后续检索使用。"
    "相比于传统的纯文本检索，多模态检索能够更好地理解视觉语义，"
    "在电商搜图、医疗影像分析、工业检测等场景有广泛应用。"
    "CLIP的训练使用了4亿图文对，通过InfoNCE损失函数进行优化，"
    "最终实现文本编码器和图像编码器的输出在向量空间中对齐。"
    "这种对齐使得跨模态检索成为可能，为多模态AI应用奠定了基础。"
)

chunks = chunker.chunk_text(long_text)
print(f"  原文长度: {len(long_text)} 字符")
print(f"  分块数: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"  chunk[{i}] len={len(c):3d}: {c[:60]}...")

# 2b. 文档级分块
if txt_path.exists():
    doc = loader.load_text(txt_path)
    doc_chunks = chunker.chunk_document(doc)
    print(f"\n  chunk_document: {len(doc_chunks)} 个 chunk")
    for c in doc_chunks[:3]:
        print(f"  {c.id}: text_len={len(c.text)}")

# ── 3. 图片预处理器 ────────────────────────────────────────
print("\n── 3. ImagePreprocessor ──")
img_proc = ImagePreprocessor()
img_files = list(_HERE.glob("*.jpg")) + list(_HERE.glob("*.png"))

if img_files:
    img_path = img_files[0]
    ok = img_proc.validate(img_path)
    print(f"  validate({img_path.name}): {ok}")
    if ok:
        info = img_proc.get_info(img_path)
        print(f"  info: {info['width']}x{info['height']}, format={info['format']}, mode={info['mode']}")
        img = img_proc.load(img_path)
        print(f"  load: size={img.size}, mode={img.mode}")
else:
    print("  ⚠ 跳过 — 无测试图片")

print("\n" + "=" * 55)
print("数据预处理模块测试完成！")
print("=" * 55)
