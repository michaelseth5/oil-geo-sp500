"""
Repo-root Streamlit entry — same behavior as ``streamlit run src/app.py``.

Use this as the Main file on Streamlit Community Cloud (``app.py``), or keep using
``src/app.py`` if you prefer.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
_src_s = str(_SRC)
if _src_s not in sys.path:
    sys.path.insert(0, _src_s)

_entry = _SRC / "app.py"
_spec = importlib.util.spec_from_file_location("_oil_geo_suite_app", _entry)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load Streamlit app from {_entry}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
