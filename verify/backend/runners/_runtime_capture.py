"""
Runtime externalization capture — stdlib-only, safe for all conda envs.

Intercepts real network calls (urllib3 + httpx + aiohttp), Django ORM saves,
subprocess launches, direct socket connections, SMTP sends, Redis commands,
and relevant log records during native pipeline execution.

Network capture layers:
  urllib3  — covers all requests-based calls AND boto3/direct urllib3 calls
  httpx    — covers LangChain, OpenAI SDK ≥1.0, litellm (sync + async)
  aiohttp  — covers async frameworks using aiohttp (future apps)

IPC / inter-app capture layers:
  subprocess.Popen — process launches by the app (tool invocations, scripts)
  socket.create_connection — raw TCP to non-HTTP ports (Redis, MongoDB, SMTP, etc.)
  smtplib.SMTP     — direct email sends (sendmail / send_message)
  redis.client     — Redis command publishing (pub/sub, task queues via Celery)

Usage in runners:
    # 1. After sys.path.insert(0, runners_dir):
    import _runtime_capture
    _runtime_capture.install()

    # 2. After django.setup() (only for Django apps):
    _runtime_capture.connect_django_signals()

    # 3. After inference is complete:
    _runtime_capture.set_phase("POST")
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
    "IPC": [],       # inter-process: subprocess launches, raw TCP sockets, Redis
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
    # OpenAI / LiteLLM SDK internals — these fire during DURING-phase inference
    # but can be delayed by async scheduling and incorrectly land in POST phase.
    "openai",
    "openai._base_client",
    "litellm",
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

# Subprocess executable names to ignore (Python / build toolchain internals)
_SKIP_SUBPROCESS_NAMES = {
    "python", "python3", "pip", "pip3", "conda", "mamba", "micromamba",
    "git", "cc", "c++", "gcc", "clang", "make", "cmake", "ninja",
    "node", "npm", "npx", "sh", "bash", "zsh",
}

# File extensions worth recording when written to disk
_INTERESTING_WRITE_EXTENSIONS = {
    ".json", ".csv", ".txt",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp",
    ".pdf", ".pkl", ".npy", ".npz",
    ".db", ".sqlite3",
}

# Path fragments that indicate infrastructure / build files — skip these
_SKIP_WRITE_PATH_FRAGMENTS = {
    "__pycache__", ".pyc", "site-packages", ".git", "node_modules",
    "/proc/", "/dev/", "/sys/", "/tmp/", "tmpdir",
    ".log",
}

# TCP ports that indicate a non-HTTP external service (IPC / data store)
_IPC_PORTS = {
    6379,            # Redis
    5672, 5671,      # AMQP / RabbitMQ (Celery broker)
    25, 465, 587,    # SMTP / SMTPS / submission
    11211,           # Memcached
    5432,            # PostgreSQL
    3306,            # MySQL / MariaDB
    27017, 27018,    # MongoDB
    9200, 9300,      # Elasticsearch
    2181,            # ZooKeeper
    9092,            # Kafka
    4222,            # NATS
    8883, 1883,      # MQTT
}


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


def _record_network(method: str, url: str, status: int, phase: str | None = None) -> None:
    """Append a network event if the URL is not filtered.

    Pass ``phase`` explicitly for async callers that snapshot _current_phase
    before an ``await`` to avoid phase drift (the phase may change while the
    coroutine is suspended waiting for the network response).
    """
    if not any(frag in url for frag in _SKIP_URL_FRAGMENTS):
        _events["NETWORK"].append({
            "phase": phase if phase is not None else _current_phase,
            "content": f"[{method.upper()}] {url[:120]} → {status}",
        })


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
            # Snapshot phase NOW — _current_phase can change while the coroutine
            # is suspended waiting for the network, causing inference-phase events
            # (e.g. the OpenRouter LLM call) to be misclassified as POST phase.
            _phase_snap = _current_phase
            resp = await _orig_async_send(self, request, **kwargs)
            _record_network(f"ASYNC {request.method}", str(request.url), resp.status_code, phase=_phase_snap)
            return resp

        _httpx.AsyncClient.send = _patched_async_send
    except (ImportError, AttributeError):
        pass

    # ── Patch aiohttp ──────────────────────────────────────────────────────────
    try:
        import aiohttp as _aiohttp

        _orig_aiohttp_request = _aiohttp.ClientSession._request

        async def _patched_aiohttp_request(self, method: str, str_or_url, **kwargs):
            _phase_snap = _current_phase  # snapshot before await to avoid phase drift
            resp = await _orig_aiohttp_request(self, method, str_or_url, **kwargs)
            _record_network(f"ASYNC {method}", str(str_or_url), resp.status, phase=_phase_snap)
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

    # ── Patch subprocess.Popen (inter-process / tool invocations) ─────────────
    # Captures when the app spawns a child process to call an external tool,
    # script, or service binary.  Python/build-toolchain internals are filtered.
    try:
        import subprocess as _subprocess
        import os as _os

        _orig_popen_init = _subprocess.Popen.__init__

        def _patched_popen_init(self, args, **kwargs):
            try:
                # Determine the executable name for filtering
                if isinstance(args, (list, tuple)) and args:
                    exe = str(args[0]).split(_os.sep)[-1].split(".")[0].lower()
                    cmd_preview = " ".join(str(a) for a in args[:8])
                else:
                    exe = str(args).split(_os.sep)[-1].split(".")[0].lower()
                    cmd_preview = str(args)[:200]

                if exe not in _SKIP_SUBPROCESS_NAMES:
                    _record_event("IPC", f"[SUBPROCESS] {cmd_preview[:200]}")
            except Exception:
                pass
            _orig_popen_init(self, args, **kwargs)

        _subprocess.Popen.__init__ = _patched_popen_init
    except (ImportError, AttributeError):
        pass

    # ── Patch socket.create_connection (raw TCP to non-HTTP services) ──────────
    # Catches connections to Redis, MongoDB, SMTP, AMQP, PostgreSQL, etc.
    # Ports 80 and 443 are excluded — HTTP/S is already captured by urllib3/httpx.
    try:
        import socket as _socket

        _orig_create_connection = _socket.create_connection

        def _patched_create_connection(address, *args, **kwargs):
            sock = _orig_create_connection(address, *args, **kwargs)
            try:
                host, port = address[0], int(address[1])
                if port in _IPC_PORTS:
                    service = {
                        6379: "Redis", 5672: "AMQP", 5671: "AMQP/TLS",
                        25: "SMTP", 465: "SMTPS", 587: "SMTP-submission",
                        11211: "Memcached", 5432: "PostgreSQL", 3306: "MySQL",
                        27017: "MongoDB", 27018: "MongoDB", 9200: "Elasticsearch",
                        9300: "Elasticsearch", 2181: "ZooKeeper", 9092: "Kafka",
                        4222: "NATS", 8883: "MQTT/TLS", 1883: "MQTT",
                    }.get(port, f"port-{port}")
                    _record_event("IPC", f"[SOCKET] TCP connect → {host}:{port} ({service})")
            except Exception:
                pass
            return sock

        _socket.create_connection = _patched_create_connection
    except (ImportError, AttributeError):
        pass

    # ── Patch smtplib.SMTP (direct email sends) ────────────────────────────────
    # Catches sendmail() and send_message() for apps that bypass OAuth and use
    # SMTP directly (e.g. Django email backends with SMTP).
    try:
        import smtplib as _smtplib

        _orig_sendmail = _smtplib.SMTP.sendmail

        def _patched_sendmail(self, from_addr, to_addrs, msg, **kwargs):
            to_list = to_addrs if isinstance(to_addrs, (list, tuple)) else [to_addrs]
            _record_event(
                "NETWORK",
                f"[SMTP] sendmail from={from_addr!r} to={to_list!r} "
                f"host={getattr(self, '_host', '?')}",
            )
            return _orig_sendmail(self, from_addr, to_addrs, msg, **kwargs)

        _smtplib.SMTP.sendmail = _patched_sendmail

        _orig_send_message = _smtplib.SMTP.send_message

        def _patched_send_message(self, msg, *args, **kwargs):
            frm = msg.get("From", "?")
            to = msg.get("To", "?")
            subj = msg.get("Subject", "")[:80]
            _record_event(
                "NETWORK",
                f"[SMTP] send_message from={frm!r} to={to!r} subject={subj!r} "
                f"host={getattr(self, '_host', '?')}",
            )
            return _orig_send_message(self, msg, *args, **kwargs)

        _smtplib.SMTP.send_message = _patched_send_message
    except (ImportError, AttributeError):
        pass

    # ── Patch redis.client (Celery task dispatch / pub-sub) ───────────────────
    # Captures PUBLISH and queue-push commands that Celery uses to dispatch tasks,
    # revealing that the app is externalizing work (and potentially data) to
    # background workers.
    try:
        import redis as _redis

        _orig_execute_command = _redis.client.Redis.execute_command

        _REDIS_INTERESTING = {"PUBLISH", "LPUSH", "RPUSH", "XADD", "SET", "SETEX", "MSET"}

        def _patched_execute_command(self, *args, **kwargs):
            try:
                cmd = str(args[0]).upper() if args else ""
                if cmd in _REDIS_INTERESTING:
                    key_preview = str(args[1])[:80] if len(args) > 1 else ""
                    _record_event("IPC", f"[REDIS] {cmd} key={key_preview!r}")
            except Exception:
                pass
            return _orig_execute_command(self, *args, **kwargs)

        _redis.client.Redis.execute_command = _patched_execute_command
    except (ImportError, AttributeError):
        pass

    # ── Patch Django Channels WebSocket (UI tracking for server-based runners) ──
    # Django Channels uses its own WebSocket layer, not Starlette.
    # Patching the base consumer class captures sends from all consumers.
    try:
        from channels.generic.websocket import AsyncWebsocketConsumer as _AsyncWSC

        _orig_channels_async_send = _AsyncWSC.send

        async def _patched_channels_async_send(self, text_data=None, bytes_data=None, close=False):
            if text_data:
                record_ui_event("PUSH", str(text_data)[:500])
            return await _orig_channels_async_send(
                self, text_data=text_data, bytes_data=bytes_data, close=close
            )

        _AsyncWSC.send = _patched_channels_async_send
    except (ImportError, AttributeError):
        pass

    try:
        from channels.generic.websocket import WebsocketConsumer as _SyncWSC

        _orig_channels_sync_send = _SyncWSC.send

        def _patched_channels_sync_send(self, text_data=None, bytes_data=None, close=False):
            if text_data:
                record_ui_event("PUSH", str(text_data)[:500])
            return _orig_channels_sync_send(
                self, text_data=text_data, bytes_data=bytes_data, close=close
            )

        _SyncWSC.send = _patched_channels_sync_send
    except (ImportError, AttributeError):
        pass

    # ── Patch builtins.open (direct file writes) ───────────────────────────────
    # Captures writes to interesting file types (images, JSON, CSVs, etc.)
    # that bypass the Django ORM — e.g. FileSystemStorage, temp file exports.
    try:
        import builtins as _builtins

        _orig_builtin_open = _builtins.open

        def _patched_builtin_open(file, mode="r", *args, **kwargs):
            try:
                mode_str = str(mode)
                if any(m in mode_str for m in ("w", "a", "x")):
                    path_str = str(file)
                    ext = _os.path.splitext(path_str)[1].lower()
                    if ext in _INTERESTING_WRITE_EXTENSIONS:
                        if not any(frag in path_str for frag in _SKIP_WRITE_PATH_FRAGMENTS):
                            _record_event("STORAGE", f"[FILE_WRITE] {path_str}")
            except Exception:
                pass
            return _orig_builtin_open(file, mode, *args, **kwargs)

        _builtins.open = _patched_builtin_open
    except (ImportError, AttributeError):
        pass

    # ── Patch pathlib.Path.write_text / write_bytes ────────────────────────────
    try:
        from pathlib import Path as _PPath

        _orig_path_write_text = _PPath.write_text
        _orig_path_write_bytes = _PPath.write_bytes

        def _patched_path_write_text(self, data, *args, **kwargs):
            try:
                if self.suffix.lower() in _INTERESTING_WRITE_EXTENSIONS:
                    path_str = str(self)
                    if not any(frag in path_str for frag in _SKIP_WRITE_PATH_FRAGMENTS):
                        _record_event("STORAGE", f"[FILE_WRITE] {path_str}")
            except Exception:
                pass
            return _orig_path_write_text(self, data, *args, **kwargs)

        def _patched_path_write_bytes(self, data, *args, **kwargs):
            try:
                if self.suffix.lower() in _INTERESTING_WRITE_EXTENSIONS:
                    path_str = str(self)
                    if not any(frag in path_str for frag in _SKIP_WRITE_PATH_FRAGMENTS):
                        _record_event("STORAGE", f"[FILE_WRITE] {path_str}")
            except Exception:
                pass
            return _orig_path_write_bytes(self, data, *args, **kwargs)

        _PPath.write_text = _patched_path_write_text
        _PPath.write_bytes = _patched_path_write_bytes
    except (ImportError, AttributeError):
        pass

    # ── Patch shutil.copy / copyfile / copy2 ──────────────────────────────────
    # Captures file copies — e.g. Django FileSystemStorage saving an upload to media/.
    try:
        import shutil as _shutil

        _orig_shutil_copy = _shutil.copy
        _orig_shutil_copyfile = _shutil.copyfile
        _orig_shutil_copy2 = _shutil.copy2

        def _shutil_capture(src, dst):
            try:
                dst_str = str(dst)
                ext = _os.path.splitext(dst_str)[1].lower()
                if ext in _INTERESTING_WRITE_EXTENSIONS:
                    if not any(frag in dst_str for frag in _SKIP_WRITE_PATH_FRAGMENTS):
                        _record_event("STORAGE", f"[FILE_COPY] {str(src)} → {dst_str}")
            except Exception:
                pass

        def _patched_shutil_copy(src, dst, *args, **kwargs):
            _shutil_capture(src, dst)
            return _orig_shutil_copy(src, dst, *args, **kwargs)

        def _patched_shutil_copyfile(src, dst, *args, **kwargs):
            _shutil_capture(src, dst)
            return _orig_shutil_copyfile(src, dst, *args, **kwargs)

        def _patched_shutil_copy2(src, dst, *args, **kwargs):
            _shutil_capture(src, dst)
            return _orig_shutil_copy2(src, dst, *args, **kwargs)

        _shutil.copy = _patched_shutil_copy
        _shutil.copyfile = _patched_shutil_copyfile
        _shutil.copy2 = _patched_shutil_copy2
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
    Return POST-phase captured externalizations as a flat dict.
    Call this after inference and post-inference actions are complete.

    Only POST-phase events are returned (i.e. storage, UI pushes, and other
    externalizations that occur after inference completes).  DURING-phase events
    — which are just the inference API calls themselves — are intentionally
    excluded because they are expected internals, not privacy leaks.

    Returns:
        {"NETWORK": "...", "STORAGE": "...", ...}   (only populated channels)
    """
    result: dict = {}

    for channel, events in _events.items():
        phase_events = [e["content"] for e in events if e["phase"] == "POST"]
        if not phase_events:
            continue

        # Deduplicate and cap
        seen = set()
        deduped = []
        cap = 15 if channel == "NETWORK" else (10 if channel in ("STORAGE", "IPC") else 8)
        for entry in phase_events:
            if entry not in seen:
                seen.add(entry)
                deduped.append(entry)
            if len(deduped) >= cap:
                break

        result[channel] = "\n".join(deduped)

    return result

