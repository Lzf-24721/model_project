"""
公共工具 — 配置加载 & 结构化日志

项目内所有模块统一用此文件加载 Config 和日志实例，
消除重复的 try/except ImportError 样板代码。

用法:
    from src.common import load_config, get_logger
    Config = load_config()
    log = get_logger(__name__)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.loader import _ConfigNamespace as ConfigType

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> "ConfigType":
    """统一配置加载器 — 消除重复的 try/except ImportError"""
    try:
        from src.config.loader import Config  # 包内导入
    except ImportError:
        # 直接运行模块 (F5 调试)
        if str(_PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(_PROJECT_ROOT))
        from config.loader import Config  # type: ignore[no-redef]
    return Config


def get_logger(name: str) -> logging.Logger:
    """获取结构化日志实例

    >>> log = get_logger(__name__)
    >>> log.info("模型加载完成 dim=%d", 512)
    [2026-06-12 10:00:00] [embedding.clip_onnx] INFO  模型加载完成 dim=512
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(name)s] %(levelname)-5s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
