"""
Minimal Django settings for xend — Verify framework use only.

Injects dummy values for required env vars and overrides the database to
SQLite so the runner can call xend's LangChain chains without a running
PostgreSQL server or Redis instance.

This file is loaded by xend_runner.py via DJANGO_SETTINGS_MODULE.
Requires xend_backend to already be on sys.path.
"""
import os
import pathlib

# ── Inject fallback values for required env vars ──────────────────────────────
# These only take effect when the real .env has not set the var.
_DUMMY_HEX_32 = "00" * 32  # 64 hex chars → 32 bytes, satisfies bytes.fromhex()

_VERIFY_DEFAULTS = {
    "DEBUG":                "True",
    "SECRET_KEY":           "verify-only-dummy-secret-key-xend-not-for-production",
    "CHANNEL_URL":          "redis://localhost:6379/0",
    "CELERY_BROKER_URL":    "redis://localhost:6379/0",
    "DATABASE_NAME":        "verify",
    "DATABASE_HOST":        "localhost",
    "DATABASE_USER":        "postgres",
    "DATABASE_PASSWORD":    "postgres",
    "DATABASE_PORT":        "5432",
    "GOOGLE_CLIENT_ID":     "verify-dummy",
    "GOOGLE_CLIENT_SECRET": "verify-dummy",
    "ENCRYPTION_KEY":       "verify-dummy",
    "SERVER_BASEURL":       "http://localhost:8000",
    "GPU_SERVER_BASEURL":   "http://localhost:8001",
    "PII_MASKING_SECRET":   _DUMMY_HEX_32,
}
for _k, _v in _VERIFY_DEFAULTS.items():
    if not os.environ.get(_k):   # covers both missing and empty-string cases
        os.environ[_k] = _v

# Import everything from xend's own local settings.
# xend_backend must be on sys.path before this module is imported.
from config.settings.local import *  # noqa: F401, F403, E402

# ── Override database to SQLite ───────────────────────────────────────────────
# SQLite file lives next to this settings file (inside verify/backend/runners/).
_DB_PATH = pathlib.Path(__file__).resolve().parent / "xend_verify.sqlite3"
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(_DB_PATH),
    }
}
