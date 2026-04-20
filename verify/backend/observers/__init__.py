"""
Out-of-process externalization observers for black-box (closed-source) target apps.

Parallel to verify/backend/runners/_runtime_capture.py (which monkey-patches the
target process for open-source apps). Black-box observers run *outside* the app
process and watch it from the host side — mitmproxy for network, adb for
filesystem/logs, uiautomator for UI.

Each observer exposes start() / stop() and an 'events' attribute populated on
stop. Observers are context managers.
"""

from verify.backend.observers.network_observer import NetworkObserver
from verify.backend.observers.fs_observer import FsObserver
from verify.backend.observers.log_observer import LogObserver
from verify.backend.observers.ui_observer import UiObserver
from verify.backend.observers.phase_classifier import classify_phases, DEFAULT_LLM_HOSTS

__all__ = [
    "NetworkObserver",
    "FsObserver",
    "LogObserver",
    "UiObserver",
    "classify_phases",
    "DEFAULT_LLM_HOSTS",
]
