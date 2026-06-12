"""
vLLM 部署模块测试 — F5 可调试

按 F5 直接运行。
"""
import sys
from pathlib import Path

# ── F5 路径 ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.vllm_server.deploy import (
    VLLMDeployer,
    detect_gpu,
    recommend_model,
    GPUInfo,
    ServerStatus,
)

_HERE = Path(__file__).resolve().parent

print("=" * 55)
print("vLLM 部署模块测试")
print("=" * 55)

# ── 1. GPU 检测 ─────────────────────────────────────────────
print("\n── 1. GPU 硬件检测 ──")
gpu = detect_gpu()
print(f"  {gpu}")
if gpu.available:
    print(f"  显存: {gpu.vram_gb:.1f} GB ({gpu.vram_mb} MB)")
    print(f"  驱动: {gpu.driver_version}")
    print(f"  CUDA: {gpu.cuda_version}")
    print(f"  算力: {gpu.compute_capability}")
else:
    print("  ⚠ 未检测到可用 GPU")

# ── 2. 模型推荐 ─────────────────────────────────────────────
print("\n── 2. 模型推荐 ──")
model = recommend_model(gpu)
print(f"  推荐模型: {model} (显存 {gpu.vram_gb:.1f}GB)")

# ── 3. 兼容检查 ─────────────────────────────────────────────
print("\n── 3. vLLM 安装检查 ──")
try:
    import vllm
    print(f"  ✅ vLLM {vllm.__version__} 已安装")
except ImportError:
    print("  ⚠ vLLM 未安装")

# ── 4. 部署器初始化 ───────────────────────────────────────────
print("\n── 4. 部署器初始化 ──")
deployer = VLLMDeployer()
print(f"  model:      {deployer.model}")
print(f"  host:port:  {deployer.host}:{deployer.port}")
print(f"  base_url:   {deployer.base_url}")
print(f"  gpu_mem:    {deployer.gpu_mem}")
print(f"  max_len:    {deployer.max_len}")
print(f"  dtype:      {deployer.dtype}")

# ── 5. CLI 预览 ─────────────────────────────────────────────
print("\n── 5. vLLM 启动命令预览 ──")
cmd = deployer._build_cmd()
print(f"  {' '.join(cmd)}")

# ── 6. 状态检查（未启动） ───────────────────────────────────
print("\n── 6. 服务状态（未启动）─")
status = deployer.status()
print(f"  running: {status.running}, ready: {status.ready}")
assert not status.running, "不应该运行中"
print("  ✅ 状态正常")

print("\n" + "=" * 55)
print("vLLM 部署模块测试完成！")
print()
print("提示:")
if not gpu.available:
    print("  ⚠ 未检测到 GPU — vLLM 需要 NVIDIA GPU")
else:
    print(f"  ✅ GPU 可用 — {gpu.name} ({gpu.vram_gb:.1f}GB)")
    print(f"  📦 推荐模型: {model}")
    print(f"  🚀 启动命令: deployer.start(wait=True)")
print()
# 不实际启动服务（测试环境只有单个 8GB 卡，避免阻塞）
print("=" * 55)
