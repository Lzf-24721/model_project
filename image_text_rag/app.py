"""
多模态 RAG 图文检索系统 — 入口

启动方式:
    streamlit run app.py
    streamlit run src/ui/visualizer.py
    python app.py
"""
import sys
from pathlib import Path

# ── 确保项目根目录在 sys.path ──
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if __name__ == "__main__":
    import streamlit.web.cli as st_cli

    app_path = str(_PROJECT_ROOT / "src" / "ui" / "visualizer.py")
    sys.argv = ["streamlit", "run", app_path,
                "--server.port", "8501",
                "--server.headless", "true"]
    st_cli.main()
