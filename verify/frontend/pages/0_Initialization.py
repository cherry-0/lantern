"""
Verify — Initialization page

Lists all registered target-app adapters, shows their conda environment status,
and lets the user trigger one-time environment setup per app.

Also hosts the per-app execution mode selector (Native / Serverless / Auto).
"""

from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path
from typing import Tuple

LANTERN_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(LANTERN_ROOT) not in sys.path:
    sys.path.insert(0, str(LANTERN_ROOT))

import streamlit as st


_MODE_OPTIONS = ["native", "serverless"]
_MODE_LABELS  = ["Native", "Serverless"]
_APP_DEFAULT_MODES = {
    "chat-driven-expense-tracker": "serverless",
    "oxproxion": "serverless",
    # No backend server — Firebase Cloud Functions / browser-only PWA
    "edupal": "serverless",
    "spendsense": "serverless",
    "fiscalflow": "serverless",
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _global_default_mode() -> str:
    """Return 'native' or 'serverless' based on USE_APP_SERVERS in .env."""
    from verify.backend.utils.config import get_env
    val = get_env("USE_APP_SERVERS", "false") or "false"
    return "native" if val.strip().lower() in ("1", "true", "yes") else "serverless"


def _sync_app_modes() -> None:
    """Push per-app mode choices from session state into the config module."""
    from verify.backend.utils.config import set_app_mode_override
    for app_name, mode in st.session_state.get("app_modes", {}).items():
        set_app_mode_override(app_name, mode)


def _default_mode_for_app(app_name: str, global_mode: str) -> str:
    """Return the default execution mode for an app."""
    return _APP_DEFAULT_MODES.get(app_name, global_mode)


def _get_adapters() -> dict:
    from verify.backend.adapters import ADAPTER_REGISTRY
    return {name: cls() for name, cls in sorted(ADAPTER_REGISTRY.items())}


def _env_status(app_name: str, adapter) -> Tuple[str, str]:
    """
    Return (level, message) where level is one of:
      "ready"    — env exists and is fully set up
      "pending"  — env exists but setup incomplete, or not yet created
      "unavail"  — conda not found / adapter unavailable
      "noop"     — no conda env needed (serverless-only adapter)
    """
    from verify.backend.utils.config import set_current_app_context
    set_current_app_context(app_name)

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
    if "app_modes" not in st.session_state:
        st.session_state.app_modes = {}        # app_name -> "auto" | "native" | "serverless"


# ─── Main UI ─────────────────────────────────────────────────────────────────

def main():
    st.title("⚙️ Initialization")

    _init_session()
    _sync_app_modes()
    adapters = _get_adapters()

    # ── Execution Mode section ────────────────────────────────────────────────
    st.subheader("Execution Mode")

    global_mode = _global_default_mode()
    st.caption(
        f"Global default from `.env` (USE_APP_SERVERS): **{global_mode}**. "
        "Choose Native or Serverless for each app below."
    )

    from verify.backend.utils.config import set_app_mode_override
    mode_changed = False
    for app_name in sorted(adapters):
        current_mode = st.session_state.app_modes.get(
            app_name,
            _default_mode_for_app(app_name, global_mode),
        )
        if current_mode not in _MODE_OPTIONS:
            current_mode = _default_mode_for_app(app_name, global_mode)
        current_idx = _MODE_OPTIONS.index(current_mode)

        col_name, col_radio = st.columns([2, 5])
        with col_name:
            st.markdown(f"**`{app_name}`**")
        with col_radio:
            chosen_label = st.radio(
                label=f"mode_{app_name}",
                options=_MODE_LABELS,
                index=current_idx,
                horizontal=True,
                label_visibility="collapsed",
                key=f"mode_radio_{app_name}",
            )
            chosen_mode = _MODE_OPTIONS[_MODE_LABELS.index(chosen_label)]

        if chosen_mode != current_mode:
            st.session_state.app_modes[app_name] = chosen_mode
            set_app_mode_override(app_name, chosen_mode)
            mode_changed = True
        elif app_name not in st.session_state.app_modes:
            # Persist the .env default into session state so other pages can sync it
            st.session_state.app_modes[app_name] = current_mode
            set_app_mode_override(app_name, current_mode)

    if mode_changed:
        st.rerun()

    st.divider()

    # ── Environment setup section ─────────────────────────────────────────────
    st.subheader("Environment Setup")
    st.markdown(
        "Set up isolated conda environments for each target app. "
        "Each environment only needs to be initialized once — subsequent runs reuse it."
    )

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
    _, col_all = st.columns([5, 2])
    with col_all:
        if st.button("Initialize All", width="stretch", disabled=any_running):
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
            effective_mode = st.session_state.app_modes.get(app_name, global_mode)
            from verify.backend.adapters.blackbox_base import BlackBoxAdapter
            kind = "black-box · device required" if isinstance(adapter, BlackBoxAdapter) else f"Python {python_ver}"
            st.caption(f"{kind} · {modalities} · **{effective_mode}**")

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
                    level, msg = _env_status(app_name, adapter)
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
                level, _ = _env_status(app_name, adapter)
            except Exception:
                level = "unavail"

            already_ready = (level == "ready") and result is None
            btn_label = (
                "Initializing…" if is_running
                else "Re-initialize" if already_ready
                else "Initialize"
            )
            if st.button(btn_label, key=f"init_{app_name}", disabled=is_running, width="stretch"):
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
