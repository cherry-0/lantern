"""
Runtime externalization capture — stdlib-only, safe for all conda envs.

Intercepts real network calls (urllib3 + httpx + aiohttp), Django ORM saves,
and relevant log records during native pipeline execution.

Network capture layers:
  urllib3  — covers all requests-based calls AND boto3/direct urllib3 calls
  httpx    — covers LangChain, OpenAI SDK ≥1.0, litellm (sync + async)
  aiohttp  — covers async frameworks using aiohttp (future apps)

Usage in runners:
    # 1. After sys.path.insert(0, runners_dir):
    import _runtime_capture
    _runtime_capture.install()

    # 2. After django.setup() (only for Django apps):
    _runtime_capture.connect_django_signals()

    # 3. After inference is complete:
    externalizations = _runtime_capture.finalize()
"""

import logging
import sys

# ── Internal state ─────────────────────────────────────────────────────────────
_events: dict[str, list[dict]] = {
    "NETWORK": [],
    "STORAGE": [],
    "LOGGING": [],
    "UI": [],
}
_current_phase: str = "DURING"  # "DURING" or "POST"
_installed: bool = False

# Logger names whose DEBUG/INFO output is too noisy to include
_NOISY_LOGGERS = {
    "django.db.backends",
    "django.template",
    "django.contrib",
    "django.request",
    "django.security",
    "django.server",
    "asyncio",
    "urllib3",
    "urllib3.connectionpool",
    "httpcore",
    "httpx",
    "filelock",
    "transformers",
    "huggingface_hub",
    "sentence_transformers",
}

# Keywords that make a log record worth keeping regardless of logger name
_INTERESTING_KEYWORDS = {
    "error", "exception", "traceback", "fail",
    "inference", "chain", "llm", "model", "embed",
    "generated", "subject", "verdict", "tag", "caption",
    "mail", "session", "expense", "category", "receipt",
    "tutor", "response", "chat", "vtuber",
}

# URL substrings to skip (internal health checks, static files, etc.)
_SKIP_URL_FRAGMENTS = {"/static/", "/favicon.ico", "/health", "/__debug__"}


def set_phase(phase: str) -> None:
    """Set the current execution phase: 'DURING' or 'POST'."""
    global _current_phase
    if phase in ["DURING", "POST"]:
        _current_phase = phase


def _record_event(channel: str, content: str) -> None:
    """Append an event to the specified channel with current phase metadata."""
    _events[channel].append({"phase": _current_phase, "content": content})


def record_ui_event(action: str, details: str = "") -> None:
    """Manually record a UI event (useful for headless runners)."""
    msg = f"[{action.upper()}] {details}" if details else f"[{action.upper()}]"
    _record_event("UI", msg)


def _record_network(method: str, url: str, status: int) -> None:
    """Append a network event if the URL is not filtered."""
    if not any(frag in url for frag in _SKIP_URL_FRAGMENTS):
        _record_event("NETWORK", f"[{method.upper()}] {url[:120]} → {status}")


class _CaptureHandler(logging.Handler):
    """Logging handler that records relevant log records to _events['LOGGING']."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            logger_root = record.name.split(".")[0]
            top2 = ".".join(record.name.split(".")[:2])

            # Always keep WARNING+ regardless of source
            if record.levelno >= logging.WARNING:
                msg = self.format(record)
                _record_event("LOGGING", msg[:300])
                return

            # Filter out noisy Django / framework namespaces at DEBUG/INFO
            if record.name in _NOISY_LOGGERS or top2 in _NOISY_LOGGERS or logger_root in _NOISY_LOGGERS:
                return

            # Keep records that contain interesting keywords
            msg_lower = record.getMessage().lower()
            if any(kw in msg_lower for kw in _INTERESTING_KEYWORDS):
                msg = self.format(record)
                _record_event("LOGGING", msg[:300])
        except Exception:
            pass


def install() -> None:
    """
    Patch urllib3, httpx, and aiohttp to intercept all HTTP calls, and install
    the logging capture handler.  Safe to call multiple times (no-op after first).
    """
    global _installed
    if _installed:
        return
    _installed = True

    # ── Patch urllib3 ──────────────────────────────────────────────────────────
    # urllib3 is the transport layer under requests AND boto3, so patching here
    # captures both without needing a separate requests patch.
    try:
        import urllib3.connectionpool as _pool

        _orig_urlopen = _pool.HTTPConnectionPool.urlopen

        def _patched_urlopen(self, method: str, url: str, **kwargs):
            resp = _orig_urlopen(self, method, url, **kwargs)
            # Reconstruct full URL from pool host/port/scheme
            scheme = getattr(self, "scheme", "https" if "HTTPS" in type(self).__name__ else "http")
            port = self.port
            default_port = 443 if scheme == "https" else 80
            port_str = f":{port}" if port and port != default_port else ""
            full_url = f"{scheme}://{self.host}{port_str}{url}"
            _record_network(method, full_url, resp.status)
            return resp

        _pool.HTTPConnectionPool.urlopen = _patched_urlopen
    except (ImportError, AttributeError):
        pass

    # ── Patch httpx (sync) ─────────────────────────────────────────────────────
    # httpx is used by LangChain, OpenAI SDK ≥1.0, litellm — does NOT go
    # through urllib3, so a separate patch is required.
    try:
        import httpx as _httpx

        _orig_sync_send = _httpx.Client.send

        def _patched_sync_send(self, request, **kwargs):
            resp = _orig_sync_send(self, request, **kwargs)
            _record_network(request.method, str(request.url), resp.status_code)
            return resp

        _httpx.Client.send = _patched_sync_send

        # ── Patch httpx (async) ────────────────────────────────────────────────
        _orig_async_send = _httpx.AsyncClient.send

        async def _patched_async_send(self, request, **kwargs):
            resp = await _orig_async_send(self, request, **kwargs)
            _record_network(f"ASYNC {request.method}", str(request.url), resp.status_code)
            return resp

        _httpx.AsyncClient.send = _patched_async_send
    except (ImportError, AttributeError):
        pass

    # ── Patch aiohttp ──────────────────────────────────────────────────────────
    try:
        import aiohttp as _aiohttp

        _orig_aiohttp_request = _aiohttp.ClientSession._request

        async def _patched_aiohttp_request(self, method: str, str_or_url, **kwargs):
            resp = await _orig_aiohttp_request(self, method, str_or_url, **kwargs)
            _record_network(f"ASYNC {method}", str(str_or_url), resp.status)
            return resp

        _aiohttp.ClientSession._request = _patched_aiohttp_request
    except (ImportError, AttributeError):
        pass

    # ── Patch FastAPI/Starlette WebSocket (UI Tracking) ────────────────────────
    # Captures data pushed to the frontend.
    try:
        from starlette.websockets import WebSocket as _WS

        _orig_send_text = _WS.send_text
        async def _patched_send_text(self, data: str):
            record_ui_event("PUSH", data[:500])
            return await _orig_send_text(self, data)
        _WS.send_text = _patched_send_text

        _orig_send_json = _WS.send_json
        async def _patched_send_json(self, data: any, **kwargs):
            import json
            record_ui_event("PUSH", json.dumps(data)[:500])
            return await _orig_send_json(self, data, **kwargs)
        _WS.send_json = _patched_send_json
    except (ImportError, AttributeError):
        pass

    # ── Install logging handler ────────────────────────────────────────────────
    handler = _CaptureHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    # Ensure root logger level lets DEBUG through to our handler
    root = logging.getLogger()
    if root.level == logging.NOTSET or root.level > logging.DEBUG:
        root.setLevel(logging.DEBUG)


def connect_django_signals() -> None:
    """
    Hook Django post_save signal to capture ORM writes.
    Must be called AFTER django.setup().
    """
    try:
        from django.db.models.signals import post_save

        def _on_save(sender, instance, created: bool, **kwargs):
            action = "CREATE" if created else "UPDATE"
            _record_event("STORAGE", f"[{sender.__name__}] {action}: {str(instance)[:150]}")

        post_save.connect(_on_save, weak=False)
    except Exception:
        pass


def finalize() -> dict:
    """
    Return the captured externalizations dict, organized by phase.
    Call this after inference and post-inference actions are complete.

    Returns:
        {
            "DURING": {"NETWORK": "...", "UI": "...", ...},
            "POST": {"NETWORK": "...", "UI": "...", ...}
        }
    """
    result: dict = {"DURING": {}, "POST": {}}

    for channel, events in _events.items():
        for phase in ["DURING", "POST"]:
            phase_events = [e["content"] for e in events if e["phase"] == phase]
            if not phase_events:
                continue

            # Deduplicate and cap
            seen = set()
            deduped = []
            cap = 15 if channel == "NETWORK" else (10 if channel == "STORAGE" else 8)
            for entry in phase_events:
                if entry not in seen:
                    seen.add(entry)
                    deduped.append(entry)
                if len(deduped) >= cap:
                    break

            result[phase][channel] = "\n".join(deduped)

    # If result["POST"] is empty, move all to flat dict for backward compat if needed,
    # but orchestrator will be updated to handle this nested structure.
    return result

