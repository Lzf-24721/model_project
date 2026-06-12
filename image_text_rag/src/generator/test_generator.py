"""
问答生成模块测试 — F5 可调试

按 F5 (或 python test_generator.py) 直接运行。
"""
import sys
from pathlib import Path

# ── F5 调试路径 ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.generator.llm import Generator, ROLE_DESCRIPTION

print("=" * 55)
print("问答生成模块测试")
print("=" * 55)

# ── 1. 初始化 Generator ─────────────────────────────────────
print("\n── 1. 初始化 Generator ──")
gen = Generator()
print(f"  base_url:    {gen.base_url}")
print(f"  model:       {gen.model}")
print(f"  system_role: {ROLE_DESCRIPTION[:60]}...")

# ── 2. 健康检查 ─────────────────────────────────────────────
print("\n── 2. LLM 服务健康检查 ──")
is_healthy = gen.health_check()
print(f"  状态: {'✅ 可连接' if is_healthy else '⚠ 无法连接'}")

if is_healthy:
    # ── 3. 获取模型列表 ──────────────────────────────────
    print("\n── 3. 模型列表 ──")
    models = gen.list_models()
    if models:
        for m in models[:5]:
            print(f"  - {m}")
        if len(models) > 5:
            print(f"  ... 共 {len(models)} 个模型")
    else:
        print("  (未能获取模型列表)")

    # ── 4. RAG 问答 ──────────────────────────────────────
    print("\n── 4. RAG 问答测试 ──")

    # 模拟检索结果
    mock_chunks = [
        {
            "text": "CLIP（Contrastive Language-Image Pre-training）是OpenAI提出的多模态模型，通过4亿图文对的对比学习将图像和文本映射到同一向量空间。",
            "score": 0.92,
            "source": "clip_intro.txt",
        },
        {
            "text": "对比学习的核心思想是拉近正样本对、推远负样本对。CLIP使用InfoNCE损失函数进行优化，实现了图文跨模态检索。",
            "score": 0.85,
            "source": "contrastive_learning.txt",
        },
        {
            "text": "FAISS是Facebook AI Research开发的高效向量相似度搜索库，支持Flat、IVF、HNSW等多种索引。",
            "score": 0.31,
            "source": "faiss_guide.txt",
        },
    ]

    question = "CLIP模型的核心原理是什么？"
    print(f"  问题: {question}")
    print(f"  上下文: {len(mock_chunks)} 个片段")
    print("  -- LLM 回答 --")

    try:
        answer = gen.answer(question, mock_chunks)
        # 缩短过长回答的显示
        if len(answer) > 500:
            print(f"  {answer[:500]}...")
            print(f"  (总长 {len(answer)} 字符)")
        else:
            print(f"  {answer}")
    except Exception as e:
        print(f"  ❌ LLM 调用失败: {e}")

    # ── 5. 边界测试：无上下文 ────────────────────────────
    print("\n── 5. 边界测试：无上下文 ──")
    try:
        no_ctx_answer = gen.answer("今天天气怎么样？", [])
        print(f"  {no_ctx_answer[:200]}")
    except Exception as e:
        print(f"  ❌ 调用失败: {e}")

    # ── 6. 直接上下文生成 ─────────────────────────────────
    print("\n── 6. 直接上下文生成 ──")
    raw_context = "Python是一种解释型、面向对象的高级编程语言，由Guido van Rossum于1991年发布。"
    try:
        direct_answer = gen.answer_with_raw_context("谁创造了Python？", raw_context)
        print(f"  问题: 谁创造了Python？")
        print(f"  回答: {direct_answer[:300]}")
    except Exception as e:
        print(f"  ❌ 调用失败: {e}")

    # ── 7. 流式生成 ──────────────────────────────────────
    print("\n── 7. 流式生成（前 200 字符截断）─")
    messages = [
        {"role": "system", "content": "用一句话简洁回答。"},
        {"role": "user", "content": "什么是向量数据库？"},
    ]
    try:
        printed = 0
        for token in gen.chat_stream(messages, max_tokens=50):
            print(token, end="", flush=True)
            printed += len(token)
            if printed > 200:
                print("...")
                break
        if printed <= 200:
            print()  # 换行
    except Exception as e:
        print(f"\n  ❌ 流式调用失败: {e}")

else:
    print("\n  ⚠ LLM 服务不可达，跳过在线测试。")
    print("  提示: 启动 Ollama 或配置正确的 base_url")

print("\n" + "=" * 55)
print("问答生成模块测试完成！")
print("=" * 55)
