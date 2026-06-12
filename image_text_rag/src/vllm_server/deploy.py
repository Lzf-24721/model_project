"""
vLLM 服务部署器 — 管理 vLLM OpenAI 兼容服务器的完整生命周期

功能:
  - 自动 GPU 检测（显存 / 算力 / 驱动版本）
  - 智能模型推荐（根据可用显存选最优模型）
  - subprocess 管理（启动 / 停止 / 重启 / 状态）
  - 健康检查 & 就绪轮询
  - 模型下载 & 缓存管理

架构:
  RAG 应用 (Generator) ──HTTP──→ vLLM Server (独立进程)
                                    └── GPU 推理

用法:
    from src.vllm_server.deploy import VLLMDeployer

    deployer = VLLMDeployer()
    deployer.start()              # 启动服务
    deployer.wait_ready(timeout=120)  # 等就绪
    deployer.status()            # 运行状态
    deployer.stop()              # 停止
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ── 全局配置 ──
try:
    from ..config.loader import Config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config.loader import Config


# ═══════════════════════════════════════════════════════════════
# GPU 信息
# ═══════════════════════════════════════════════════════════════
@dataclass
class GPUInfo:
    """GPU 硬件信息"""
    name: str = "unknown"
    vram_mb: int = 0
    vram_gb: float = 0.0
    cuda_version: str = "unknown"
    driver_version: str = "unknown"
    compute_capability: str = "unknown"
    available: bool = False

    @property
    def is_usable(self) -> bool:
        return self.available and self.vram_mb > 2000  # 至少 2GB

    def __repr__(self) -> str:
        if not self.available:
            return "GPUInfo(no GPU available)"
        return (
            f"GPUInfo({self.name}, {self.vram_gb:.1f}GB, "
            f"CUDA {self.cuda_version}, driver {self.driver_version})"
        )


def detect_gpu() -> GPUInfo:
    """检测本机 GPU 硬件信息"""
    info = GPUInfo()

    # 1. nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 2:
                info.name = parts[0]
                info.vram_mb = int(parts[1])
                info.vram_gb = info.vram_mb / 1024
                info.available = True
            if len(parts) >= 3:
                info.compute_capability = parts[2]
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # 2. nvidia-smi driver / cuda version
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            import re
            driver_m = re.search(r"Driver Version:\s+(\S+)", result.stdout)
            cuda_m = re.search(r"CUDA Version:\s+(\S+)", result.stdout)
            if driver_m:
                info.driver_version = driver_m.group(1)
            if cuda_m:
                info.cuda_version = cuda_m.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return info


# ═══════════════════════════════════════════════════════════════
# 模型推荐（根据显存大小）
# ═══════════════════════════════════════════════════════════════
_MODEL_RECOMMENDATIONS = [
    # (最小显存 GB, 模型名, 描述)
    (20.0, "Qwen/Qwen2.5-7B-Instruct",        "7B 参数，强推理能力，需 14GB+ 显存"),
    (12.0, "Qwen/Qwen2.5-3B-Instruct",         "3B 参数，均衡性能，需 6GB 显存"),
    (6.0,  "Qwen/Qwen2.5-1.5B-Instruct",       "1.5B 参数，轻量快速，需 3GB 显存"),
    (3.0,  "Qwen/Qwen2.5-0.5B-Instruct",       "0.5B 参数，极轻量，需 1.5GB 显存"),
]


def recommend_model(gpu: GPUInfo | None = None) -> str:
    """根据 GPU 显存自动推荐最优模型"""
    if gpu is None:
        gpu = detect_gpu()
    vram_gb = gpu.vram_gb if gpu.available else 0

    for min_vram, model, desc in _MODEL_RECOMMENDATIONS:
        if vram_gb >= min_vram and gpu.is_usable:
            return model

    # CPU 或无 GPU — 用最小的
    return "Qwen/Qwen2.5-0.5B-Instruct"


# ═══════════════════════════════════════════════════════════════
# vLLM 部署器
# ═══════════════════════════════════════════════════════════════
@dataclass
class ServerStatus:
    """服务器运行状态"""
    running: bool = False
    pid: Optional[int] = None
    port: int = 8000
    model: str = ""
    uptime_seconds: float = 0.0
    gpu_memory_used_mb: float = 0.0
    error: str = ""
    ready: bool = False


class VLLMDeployer:
    """
    vLLM 服务部署器

    用法:
        d = VLLMDeployer()
        d.start()
        d.wait_ready()
        # ... 使用 Generator 调用 http://localhost:8000/v1 ...
        d.stop()
    """

    def __init__(
        self,
        model: str | None = None,
        port: int | None = None,
        host: str | None = None,
        gpu_memory_utilization: float | None = None,
        max_model_len: int | None = None,
        dtype: str | None = None,
        trust_remote_code: bool | None = None,
        enforce_eager: bool | None = None,
        max_num_seqs: int | None = None,
    ):
        """
        Args:
            model:                  模型名 (HF hub id)，None → Config / 自动推荐
            port:                   服务端口
            host:                   绑定地址
            gpu_memory_utilization: GPU 显存使用比例 (0~1)
            max_model_len:          最大上下文长度
            dtype:                  推理精度 (auto / float16 / bfloat16)
            trust_remote_code:      是否信任模型远程代码
            enforce_eager:          禁用 CUDA Graph (调试用)
            max_num_seqs:           最大并发请求数
        """
        self.model = model or Config.VLLM_MODEL
        self.port = port or Config.VLLM_PORT
        self.host = host or Config.VLLM_HOST
        self.gpu_mem = gpu_memory_utilization or Config.VLLM_GPU_MEMORY_UTILIZATION
        self.max_len = max_model_len or Config.VLLM_MAX_MODEL_LEN
        self.dtype = dtype or Config.VLLM_DTYPE
        self.trust_code = (
            trust_remote_code
            if trust_remote_code is not None
            else Config.VLLM_TRUST_REMOTE_CODE
        )
        self.enforce_eager = (
            enforce_eager
            if enforce_eager is not None
            else Config.VLLM_ENFORCE_EAGER
        )
        self.max_seqs = max_num_seqs or Config.VLLM_MAX_NUM_SEQS

        self._process: Optional[subprocess.Popen] = None
        self._start_time: float = 0.0
        self._gpu: GPUInfo = detect_gpu()

    # ── 属性 ────────────────────────────────────────────────

    @property
    def gpu(self) -> GPUInfo:
        return self._gpu

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process else None

    # ── CLI 构建 ────────────────────────────────────────────

    def _build_cmd(self) -> List[str]:
        """构建 vllm serve 命令"""
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_mem),
            "--max-model-len", str(self.max_len),
            "--dtype", self.dtype,
            "--max-num-seqs", str(self.max_seqs),
        ]
        if self.trust_code:
            cmd.append("--trust-remote-code")
        if self.enforce_eager:
            cmd.append("--enforce-eager")
        return cmd

    # ── 生命周期 ────────────────────────────────────────────

    def start(self, wait: bool = False, timeout: int = 300) -> ServerStatus:
        """
        启动 vLLM 服务（后台子进程）

        Args:
            wait:    是否阻塞等待就绪
            timeout: 就绪等待超时秒数

        Returns:
            当前服务状态
        """
        if self.is_running:
            return self.status()

        # 检查 vLLM 是否安装
        try:
            subprocess.run(
                [sys.executable, "-c", "import vllm"],
                capture_output=True, text=True, timeout=10,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ServerStatus(
                error="vLLM 未安装。请运行: pip install vllm"
            )

        # 检查 GPU
        if not self._gpu.available:
            return ServerStatus(
                error="未检测到可用 GPU。vLLM 需要 NVIDIA GPU。"
            )

        # 验证模型适配显存
        recommended = recommend_model(self._gpu)
        if self.model != recommended:
            print(
                f"[vLLM] ⚠ 当前显存 {self._gpu.vram_gb:.1f}GB，"
                f"推荐模型: {recommended}\n"
                f"       当前选择: {self.model}（可能显存不足）"
            )

        cmd = self._build_cmd()
        print(f"[vLLM] 🚀 启动服务: {' '.join(cmd)}")

        # 后台启动
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env.setdefault("HF_ENDPOINT", Config.HF_ENDPOINT)

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )
        self._start_time = time.time()

        if wait:
            self.wait_ready(timeout=timeout)

        return self.status()

    def stop(self, graceful: bool = True) -> ServerStatus:
        """停止 vLLM 服务"""
        if self._process is None:
            return ServerStatus(error="服务未在运行")

        print(f"[vLLM] 🛑 停止服务 (PID={self._process.pid})...")
        if graceful:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
        else:
            self._process.kill()
            self._process.wait(timeout=10)

        self._process = None
        self._start_time = 0.0
        return ServerStatus(running=False)

    def restart(self, timeout: int = 120) -> ServerStatus:
        """重启服务"""
        self.stop(graceful=True)
        time.sleep(2)
        return self.start(wait=True, timeout=timeout)

    def status(self) -> ServerStatus:
        """查询服务状态"""
        st = ServerStatus(
            running=self.is_running,
            pid=self.pid,
            port=self.port,
            model=self.model,
        )

        if self.is_running and self._start_time > 0:
            st.uptime_seconds = time.time() - self._start_time

        # 健康检查
        st.ready = self._check_health()
        if st.ready:
            st.gpu_memory_used_mb = self._get_gpu_usage()

        return st

    # ── 健康检查 ────────────────────────────────────────────

    def _check_health(self) -> bool:
        """检查 /v1/models 是否可达"""
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def wait_ready(self, timeout: int = 300, interval: float = 2.0) -> bool:
        """轮询等待服务就绪"""
        if not self.is_running:
            print("[vLLM] ❌ 进程未启动")
            return False

        print(f"[vLLM] ⏳ 等待服务就绪 (最长 {timeout}s)...")
        elapsed = 0.0

        # 先等进程启动
        time.sleep(3)

        while elapsed < timeout:
            # 检查进程是否挂掉
            if self._process and self._process.poll() is not None:
                stderr = self._process.stderr
                err_text = ""
                if stderr:
                    try:
                        err_text = stderr.read()
                    except Exception:
                        err_text = "<无法读取 stderr>"
                print(f"[vLLM] ❌ 进程异常退出 (code={self._process.returncode})")
                print(f"[vLLM] stderr: {err_text[-500:]}")
                return False

            if self._check_health():
                print(f"[vLLM] ✅ 服务就绪! ({elapsed:.0f}s)")
                return True

            # 读取 stderr 看模型加载进度
            if self._process and self._process.stderr:
                try:
                    line = self._process.stderr.readline()
                    if line and "Loading" in line:
                        print(f"  {line.strip()}")
                except Exception:
                    pass

            time.sleep(interval)
            elapsed += interval

        print(f"[vLLM] ⏰ 等待超时 ({timeout}s)，请检查日志")
        return False

    def _get_gpu_usage(self) -> float:
        """获取 GPU 显存占用 (MB)"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass
        return 0.0

    # ── 日志 ────────────────────────────────────────────────

    def tail_logs(self, lines: int = 20) -> str:
        """读取 vLLM 最近 N 行 stderr 输出"""
        if not self._process or not self._process.stderr:
            return "(服务未运行)"
        # subprocess.PIPE 不支持 seek，返回最近收集的内容
        return "(实时日志仅在启动过程中捕获)"

    def get_output(self) -> Tuple[str, str]:
        """获取 stdout / stderr 当前内容（非阻塞）"""
        if not self._process:
            return ("", "(服务未运行)")
        stdout = stderr = ""
        try:
            if self._process.stdout:
                import select
                if select.select([self._process.stdout], [], [], 0)[0]:
                    stdout = self._process.stdout.read(4096) or ""
            if self._process.stderr:
                import select
                if select.select([self._process.stderr], [], [], 0)[0]:
                    stderr = self._process.stderr.read(4096) or ""
        except Exception:
            pass
        return (stdout, stderr)
