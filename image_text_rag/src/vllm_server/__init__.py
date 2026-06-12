"""
vLLM 服务部署模块 — OpenAI 兼容 API 服务器

用法:
    from src.vllm_server import VLLMDeployer, detect_gpu, recommend_model

    # 查询 GPU
    gpu = detect_gpu()
    print(gpu)

    # 部署
    d = VLLMDeployer()
    d.start(wait=True)
    # ... 运行 RAG 应用 ...
    d.stop()
"""

from .deploy import (
    VLLMDeployer,
    detect_gpu,
    recommend_model,
    GPUInfo,
    ServerStatus,
)
