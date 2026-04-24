"""
Host-side drivers for invoking black-box (closed-source) target apps.

Parallel to verify/backend/utils/conda_runner.py (which spawns a conda
subprocess for open-source apps). Black-box drivers manage Android emulators
and drive app UIs via uiautomator2.
"""

from verify.backend.drivers.emulator_manager import EmulatorManager

__all__ = ["EmulatorManager"]
