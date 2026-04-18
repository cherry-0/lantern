"""
Verify — Streamlit entry point

Defines the multi-page navigation with human-readable page names.
Run with: streamlit run verify/frontend/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent
if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Verify",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

_PAGES_DIR = Path(__file__).parent / "pages"

pg = st.navigation([
    st.Page(str(_PAGES_DIR / "0_Initialization.py"),          title="Initialization",                      icon="⚙️"),
    st.Page(str(_PAGES_DIR / "1_Perturb_Input.py"),           title="Perturb Input",                       icon="🔍"),
    st.Page(str(_PAGES_DIR / "1_View_Results.py"),            title="View I-PI Comparison Results",        icon="📊"),
    st.Page(str(_PAGES_DIR / "2_Input_Output_Comparison.py"), title="Input / Output Comparison",           icon="🔄"),
    st.Page(str(_PAGES_DIR / "3_View_IOC_Results.py"),        title="View Input-Output Comparison Results", icon="🔬"),
    st.Page(str(_PAGES_DIR / "4_Batch_Runner.py"),            title="Batch Runner",                        icon="⚡"),
    st.Page(str(_PAGES_DIR / "5_Experiment_Progress.py"),     title="Experiment Progress",                 icon="📋"),
    st.Page(str(_PAGES_DIR / "6_Reeval.py"),                  title="Re-evaluate",                         icon="🔄"),
    st.Page(str(_PAGES_DIR / "7_Eval_Validation.py"),         title="Evaluation Validation",               icon="✅"),
])
pg.run()
