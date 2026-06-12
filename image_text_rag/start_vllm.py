#!/usr/bin/env python
"""
vLLM 一键启动脚本

用法:
    python start_vllm.py              # 自动检测 GPU，推荐模型，启动
    python start_vllm.py --status     # 查看服务状态
    python start_vllm.py --stop       # 停止服务
    python start_vllm.py --model Qwen/Qwen2.5-3B-Instruct  # 指定模型

启动后访问:
    API:   http://localhost:8000/v1
    Docs:  http://localhost:8000/docs
"""
import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.vllm_server.deploy import VLLMDeployer, detect_gpu, recommend_model


def main():
    parser = argparse.ArgumentParser(description="vLLM 服务管理")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名 (HF hub id)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-mem", type=float, default=None,
                        help="GPU 显存使用比例 (0~1)")
    parser.add_argument("--max-len", type=int, default=None,
                        help="最大上下文长度")
    parser.add_argument("--status", action="store_true",
                        help="查询服务状态")
    parser.add_argument("--stop", action="store_true",
                        help="停止服务")
    parser.add_argument("--restart", action="store_true",
                        help="重启服务")
    parser.add_argument("--no-wait", action="store_true",
                        help="启动后不等待就绪")
    args = parser.parse_args()

    # GPU 检测
    gpu = detect_gpu()
    if not gpu.available:
        print("❌ 未检测到可用 GPU。vLLM 需要 NVIDIA GPU。")
        sys.exit(1)

    print(f"🖥  {gpu}")
    model = args.model or recommend_model(gpu)
    print(f"📦 模型: {model}")
    print(f"🔗 地址: http://localhost:{args.port}/v1")

    deployer = VLLMDeployer(
        model=model,
        port=args.port,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_len,
    )

    if args.stop:
        deployer.stop()
        print("✅ 已停止")
    elif args.restart:
        status = deployer.restart()
        print(f"✅ 已重启: PID={status.pid}, ready={status.ready}")
    elif args.status:
        s = deployer.status()
        print(f"  running: {s.running}")
        print(f"  ready:   {s.ready}")
        print(f"  pid:     {s.pid}")
        print(f"  port:    {s.port}")
        print(f"  model:   {s.model}")
        if s.running:
            print(f"  uptime:  {s.uptime_seconds:.0f}s")
            print(f"  gpu_mem: {s.gpu_memory_used_mb:.0f}MB")
    else:
        status = deployer.start(wait=not args.no_wait, timeout=600)
        if status.error:
            print(f"❌ {status.error}")
            sys.exit(1)
        print(f"✅ vLLM 服务已启动: PID={status.pid}")
        print(f"🔗 API: http://localhost:{args.port}/v1/chat/completions")


if __name__ == "__main__":
    main()
