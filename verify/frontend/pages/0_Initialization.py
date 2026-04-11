"""
Verify — Initialization page

Lists all registered target-app adapters, shows their conda environment status,
and lets the user trigger one-time environment setup per app.
"""

from __future__ import annotations

import concurrent.futures
import sys
import threading
from pathlib import Path
from typing import Tuple

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Verify — Initialization",
    page_icon="⚙️",
    layout="wide",
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_adapters() -> dict:
    from verify.backend.adapters import ADAPTER_REGISTRY
    return {name: cls() for name, cls in sorted(ADAPTER_REGISTRY.items())}


def _env_status(adapter) -> Tuple[str, str]:
    """
    Return (level, message) where level is one of:
      "ready"    — env exists and is fully set up
      "pending"  — env exists but setup incomplete, or not yet created
      "unavail"  — conda not found / no adapter
      "noop"     — no conda env needed (serverless adapter)
    """
    env_spec = adapter.env_spec
    if env_spec is None:
        ok, msg = adapter.check_availability()
        return ("ready" if ok else "unavail"), msg

    from verify.backend.utils.conda_runner import CondaRunner
    conda = CondaRunner.find_conda()
    if not conda:
        return "unavail", "conda/mamba not found on PATH — install Miniconda or Mambaforge."

    if CondaRunner.is_ready(env_spec.name):
        return "ready", f"conda env `{env_spec.name}` (Python {env_spec.python}) is ready."
    if CondaRunner.env_exists(env_spec.name):
        return "pending", f"conda env `{env_spec.name}` exists but setup is incomplete."
    return "pending", f"conda env `{env_spec.name}` (Python {env_spec.python}) not yet created."


# ─── Session state ────────────────────────────────────────────────────────────

def _init_session():
    if "_init_futures" not in st.session_state:
        st.session_state._init_futures = {}   # app_name -> Future | None
    if "_init_results" not in st.session_state:
        st.session_state._init_results = {}   # app_name -> (ok, msg) | None
    if "_executor" not in st.session_state:
        st.session_state._executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)


# ─── Main UI ─────────────────────────────────────────────────────────────────

def main():
    st.title("⚙️ Initialization")
    st.markdown(
        "Set up isolated conda environments for each target app. "
        "Each environment only needs to be initialized once — subsequent runs reuse it."
    )

    _init_session()
    adapters = _get_adapters()

    # Resolve any completed futures into results
    for app_name, future in list(st.session_state._init_futures.items()):
        if future is not None and future.done():
            try:
                ok, msg = future.result()
            except Exception as e:
                ok, msg = False, str(e)
            st.session_state._init_results[app_name] = (ok, msg)
            st.session_state._init_futures[app_name] = None

    any_running = any(
        f is not None and not f.done()
        for f in st.session_state._init_futures.values()
    )

    # ── "Initialize All" button ───────────────────────────────────────────────
    col_hdr, col_all = st.columns([5, 2])
    with col_all:
        if st.button("Initialize All", use_container_width=True, disabled=any_running):
            for app_name, adapter in adapters.items():
                existing = st.session_state._init_futures.get(app_name)
                if existing is not None and not existing.done():
                    continue  # already running
                future = st.session_state._executor.submit(adapter.initialize)
                st.session_state._init_futures[app_name] = future
                st.session_state._init_results.pop(app_name, None)
            st.rerun()

    st.divider()

    # ── Per-app rows ──────────────────────────────────────────────────────────
    for app_name, adapter in adapters.items():
        future = st.session_state._init_futures.get(app_name)
        is_running = future is not None and not future.done()
        result = st.session_state._init_results.get(app_name)

        col_name, col_status, col_action = st.columns([2, 5, 2])

        with col_name:
            st.markdown(f"**`{app_name}`**")
            modalities = ", ".join(adapter.supported_modalities) if adapter.supported_modalities else "—"
            python_ver = adapter.env_spec.python if adapter.env_spec else "—"
            st.caption(f"Python {python_ver} · {modalities}")

        with col_status:
            if is_running:
                st.info("Installing dependencies… this may take a few minutes.")
            elif result is not None:
                ok, msg = result
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                try:
                    level, msg = _env_status(adapter)
                except Exception as e:
                    level, msg = "unavail", str(e)

                if level == "ready":
                    st.success(msg)
                elif level == "pending":
                    st.warning(msg)
                else:
                    st.error(msg)

        with col_action:
            try:
                level, _ = _env_status(adapter)
            except Exception:
                level = "unavail"

            already_ready = (level == "ready") and result is None
            btn_label = (
                "Initializing…" if is_running
                else "Re-initialize" if already_ready
                else "Initialize"
            )
            if st.button(btn_label, key=f"init_{app_name}", disabled=is_running, use_container_width=True):
                future = st.session_state._executor.submit(adapter.initialize)
                st.session_state._init_futures[app_name] = future
                st.session_state._init_results.pop(app_name, None)
                st.rerun()

        st.divider()

    # Auto-refresh while any initialization is running
    if any_running:
        import time
        time.sleep(3)
        st.rerun()


if __name__ == "__main__":
    main()
