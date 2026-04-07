"""
Runtime externalization capture — stdlib-only, safe for all conda envs.

Intercepts real network calls (requests + httpx), Django ORM saves, and
relevant log records during native pipeline execution.

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
_network_events: list = []
_storage_events: list = []
_log_events: list = []
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


class _CaptureHandler(logging.Handler):
    """Logging handler that records relevant log records to _log_events."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            logger_root = record.name.split(".")[0]
            top2 = ".".join(record.name.split(".")[:2])

            # Always keep WARNING+ regardless of source
            if record.levelno >= logging.WARNING:
                msg = self.format(record)
                _log_events.append(msg[:300])
                return

            # Filter out noisy Django / framework namespaces at DEBUG/INFO
            if record.name in _NOISY_LOGGERS or top2 in _NOISY_LOGGERS or logger_root in _NOISY_LOGGERS:
                return

            # Keep records that contain interesting keywords
            msg_lower = record.getMessage().lower()
            if any(kw in msg_lower for kw in _INTERESTING_KEYWORDS):
                msg = self.format(record)
                _log_events.append(msg[:300])
        except Exception:
            pass


def install() -> None:
    """
    Patch requests and httpx to intercept all HTTP calls, and install
    the logging capture handler.  Safe to call multiple times (no-op after first).
    """
    global _installed
    if _installed:
        return
    _installed = True

    # ── Patch requests ─────────────────────────────────────────────────────────
    try:
        import requests as _requests

        _orig_req = _requests.Session.request

        def _patched_request(self, method: str, url: str, **kwargs):
            resp = _orig_req(self, method, url, **kwargs)
            url_str = str(url)
            if not any(frag in url_str for frag in _SKIP_URL_FRAGMENTS):
                _network_events.append(
                    f"[{method.upper()}] {url_str[:120]} → {resp.status_code}"
                )
            return resp

        _requests.Session.request = _patched_request
    except ImportError:
        pass

    # ── Patch httpx (sync) ─────────────────────────────────────────────────────
    try:
        import httpx as _httpx

        _orig_sync_send = _httpx.Client.send

        def _patched_sync_send(self, request, **kwargs):
            resp = _orig_sync_send(self, request, **kwargs)
            url_str = str(request.url)
            if not any(frag in url_str for frag in _SKIP_URL_FRAGMENTS):
                _network_events.append(
                    f"[{request.method}] {url_str[:120]} → {resp.status_code}"
                )
            return resp

        _httpx.Client.send = _patched_sync_send

        # ── Patch httpx (async) ────────────────────────────────────────────────
        _orig_async_send = _httpx.AsyncClient.send

        async def _patched_async_send(self, request, **kwargs):
            resp = await _orig_async_send(self, request, **kwargs)
            url_str = str(request.url)
            if not any(frag in url_str for frag in _SKIP_URL_FRAGMENTS):
                _network_events.append(
                    f"[ASYNC {request.method}] {url_str[:120]} → {resp.status_code}"
                )
            return resp

        _httpx.AsyncClient.send = _patched_async_send
    except ImportError:
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
            _storage_events.append(
                f"[{sender.__name__}] {action}: {str(instance)[:150]}"
            )

        post_save.connect(_on_save, weak=False)
    except Exception:
        pass


def finalize() -> dict:
    """
    Return the captured externalizations dict.
    Call this after inference is complete.

    Returns a dict with any of: NETWORK, STORAGE, LOGGING.
    Empty channels are omitted.
    """
    result: dict = {}

    if _network_events:
        result["NETWORK"] = "\n".join(_network_events[:15])

    if _storage_events:
        result["STORAGE"] = "\n".join(_storage_events[:10])

    # Filter log events — deduplicate and cap
    seen: set = set()
    filtered_logs: list = []
    for entry in _log_events:
        key = entry[:80]  # rough dedup key
        if key not in seen:
            seen.add(key)
            filtered_logs.append(entry)
        if len(filtered_logs) >= 8:
            break
    if filtered_logs:
        result["LOGGING"] = "\n".join(filtered_logs)

    return result
