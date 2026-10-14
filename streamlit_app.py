"""
Streamlit Community Cloud entrypoint for Memento-Agri.
Directs execution to app/streamlit_app.py.
"""
from pathlib import Path
import runpy
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

app_path = ROOT_DIR / "app" / "streamlit_app.py"
runpy.run_path(str(app_path), run_name="__main__")
