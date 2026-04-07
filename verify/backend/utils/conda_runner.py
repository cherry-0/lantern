"""
CondaRunner — create isolated conda environments for native app pipelines and run
inference as a subprocess, completely bypassing sys.path hacks and Django
multi-app import conflicts.

Usage pattern in an adapter
----------------------------
    from verify.backend.utils.conda_runner import CondaRunner, EnvSpec

    _SPEC = EnvSpec(
        name="snapdo",
        python="3.10",
        install_cmds=[["pip", "install", "-r", str(REQUIREMENTS_PATH)]],
    )
    _RUNNER = Path(__file__).parent.parent / "runners" / "snapdo_runner.py"

    # check_availability:
    ok, msg = CondaRunner.probe(_SPEC)   # fast, non-blocking

    # run_pipeline:
    ok, msg = CondaRunner.ensure(_SPEC)  # blocking first-time setup
    ok, result, err = CondaRunner.run(_SPEC.name, _RUNNER, input_dict, timeout=90)
"""

import json
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from verify.backend.utils.verbose_log import log_setup_start, log_setup_step, log_setup_done


@dataclass
class EnvSpec:
    name: str                           # conda env name (= app name)
    python: str                         # Python version, e.g. "3.10"
    install_cmds: List[List[str]]       # commands run inside env after creation
    cwd: Optional[Path] = None          # working directory for install commands


class CondaRunner:
    """Manages per-app conda environments and subprocess inference calls."""

    # Class-level cache so we don't shell out repeatedly
    _conda_exe: Optional[str] = None
    _env_cache: Dict[str, bool] = {}    # name → exists

    # ---------------------------------------------------------------------------
    # Finding conda
    # ---------------------------------------------------------------------------

    @classmethod
    def find_conda(cls) -> Optional[str]:
        if cls._conda_exe:
            return cls._conda_exe
        for candidate in ("conda", "mamba", "micromamba"):
            result = subprocess.run(
                ["which", candidate], capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                cls._conda_exe = result.stdout.strip()
                return cls._conda_exe
        return None

    # ---------------------------------------------------------------------------
    # Env presence
    # ---------------------------------------------------------------------------

    @classmethod
    def env_exists(cls, name: str) -> bool:
        if name in cls._env_cache:
            return cls._env_cache[name]
        conda = cls.find_conda()
        if not conda:
            return False
        result = subprocess.run(
            [conda, "env", "list", "--json"], capture_output=True, text=True
        )
        if result.returncode != 0:
            return False
        try:
            data = json.loads(result.stdout)
            exists = any(Path(p).name == name for p in data.get("envs", []))
        except json.JSONDecodeError:
            exists = False
        cls._env_cache[name] = exists
        return exists

    @classmethod
    def _sentinel(cls, name: str) -> Optional[Path]:
        """Return path to the sentinel file that marks a finished setup."""
        conda = cls.find_conda()
        if not conda:
            return None
        result = subprocess.run(
            [conda, "env", "list", "--json"], capture_output=True, text=True
        )
        if result.returncode != 0:
            return None
        try:
            data = json.loads(result.stdout)
            for p in data.get("envs", []):
                if Path(p).name == name:
                    return Path(p) / ".verify_ready"
        except json.JSONDecodeError:
            pass
        return None

    @classmethod
    def is_ready(cls, name: str) -> bool:
        """Return True only when the env exists AND setup has completed."""
        s = cls._sentinel(name)
        return s is not None and s.exists()

    # ---------------------------------------------------------------------------
    # probe — fast, non-blocking (for check_availability)
    # ---------------------------------------------------------------------------

    @classmethod
    def probe(cls, spec: EnvSpec) -> Tuple[bool, str]:
        """
        Non-blocking availability check.  Does NOT create or install anything.

        Returns:
            (True, reason)  if conda exists and env is already set up
            (True, reason)  if conda exists but env is pending (will be created on first run)
            (False, reason) if conda is missing
        """
        conda = cls.find_conda()
        if not conda:
            return False, "conda/mamba not found on PATH — install Miniconda or Mambaforge."

        if cls.is_ready(spec.name):
            return True, f"[NATIVE] conda env '{spec.name}' (python {spec.python}) ready."
        if cls.env_exists(spec.name):
            return True, (
                f"[NATIVE] conda env '{spec.name}' exists but setup is incomplete. "
                "Will finish on first run."
            )
        return True, (
            f"[NATIVE] conda env '{spec.name}' (python {spec.python}) will be created "
            "on first run (one-time setup)."
        )

    # ---------------------------------------------------------------------------
    # ensure — blocking first-time setup (called from run_pipeline)
    # ---------------------------------------------------------------------------

    @classmethod
    def ensure(cls, spec: EnvSpec) -> Tuple[bool, str]:
        """
        Create conda env and install dependencies if not already done.
        Uses a sentinel file to skip re-installation on subsequent calls.

        Returns (ok, message).
        """
        conda = cls.find_conda()
        if not conda:
            return False, "conda/mamba not found on PATH."

        # Already set up?
        if cls.is_ready(spec.name):
            return True, f"Env '{spec.name}' already set up."

        # One-time setup — print progress so the user knows what is happening
        log_setup_start(spec.name, spec.python)

        # Create env if needed
        if not cls.env_exists(spec.name):
            result = subprocess.run(
                [conda, "create", "-n", spec.name, f"python={spec.python}", "-y"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                err = f"conda create failed for '{spec.name}': {result.stderr}"
                log_setup_done(ok=False, error=err)
                return False, err
            cls._env_cache[spec.name] = True

        # Run install commands
        for cmd in spec.install_cmds:
            log_setup_step(cmd)
            result = subprocess.run(
                [conda, "run", "-n", spec.name] + cmd,
                capture_output=True, text=True,
                cwd=str(spec.cwd) if spec.cwd else None,
            )
            if result.returncode != 0:
                err = (
                    f"Install step failed in env '{spec.name}' "
                    f"({' '.join(cmd)}): {result.stderr[-1000:]}"
                )
                log_setup_done(ok=False, error=err)
                return False, err

        # Mark as ready
        sentinel = cls._sentinel(spec.name)
        if sentinel:
            sentinel.touch()

        log_setup_done(ok=True)
        return True, f"Env '{spec.name}' created and ready."

    # ---------------------------------------------------------------------------
    # run — execute a runner script and return parsed JSON result
    # ---------------------------------------------------------------------------

    @classmethod
    def run(
        cls,
        env_name: str,
        runner_script: Path,
        input_data: Dict[str, Any],
        timeout: int = 120,
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Run a runner script inside the conda env.

        The runner script must:
          - Accept a single CLI argument: path to a JSON input file
          - Print a single JSON object to stdout (the result)
          - Exit 0 on success, non-zero on failure

        Returns:
            (ok, result_dict, error_message)
        """
        conda = cls.find_conda()
        if not conda:
            return False, {}, "conda not found."

        tmp_input = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        try:
            json.dump(input_data, tmp_input)
            tmp_input.close()

            proc = subprocess.Popen(
                [conda, "run", "-n", env_name,
                 "python", str(runner_script), tmp_input.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Relay stderr line-by-line to the terminal in real time so
            # database-creation and inference-module logs are immediately visible.
            stderr_lines: List[str] = []

            def _relay_stderr() -> None:
                for line in proc.stderr:
                    print(line, end="", flush=True)
                    stderr_lines.append(line)

            stderr_thread = threading.Thread(target=_relay_stderr, daemon=True)
            stderr_thread.start()

            try:
                stdout_raw = proc.stdout.read()
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                return False, {}, f"Runner timed out after {timeout}s."
            finally:
                stderr_thread.join(timeout=5)

            stderr = "".join(stderr_lines)
            stdout = stdout_raw.strip()

            # Extract JSON from stdout.
            # Use find/rfind to grab from the first '{' to the last '}' — this
            # correctly handles multi-line JSON bodies that contain '{' characters.
            # Logging now goes to stderr so stdout contains only the result JSON.
            json_start = stdout.find("{")
            json_end   = stdout.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                try:
                    output = json.loads(stdout[json_start:json_end])
                    if proc.returncode != 0:
                        err = output.get("error") or stderr[-1500:]
                        return False, output, err
                    return True, output, ""
                except json.JSONDecodeError:
                    pass

            if proc.returncode != 0:
                # No JSON at all — surface stderr so the user can debug
                return False, {}, (
                    f"Runner exited {proc.returncode}.\n"
                    f"stderr: {stderr[-2000:]}\n"
                    f"stdout: {stdout[:500]}"
                )

            return False, {}, f"No JSON found in runner output: {stdout[:500]}"

        except Exception as exc:
            return False, {}, f"CondaRunner.run error: {exc}"
        finally:
            os.unlink(tmp_input.name)
