"""
Minimal Django settings for momentag — Verify framework use only.

Injects dummy values for required env vars and overrides the database to
SQLite so the runner can call CLIP/BLIP inference without MySQL or Qdrant.

Requires momentag_backend (target-apps/momentag/backend) on sys.path.
"""
import os
import pathlib

_VERIFY_DEFAULTS = {
    "SECRET_KEY":           "verify-only-dummy-secret-key-momentag",
    "DEBUG":                "True",
    "QDRANT_CLUSTER_URL":   "http://localhost:6333",
    "QDRANT_API_KEY":       "verify-dummy",
    "DB_NAME":              "verify",
    "DB_USER":              "root",
    "DB_PASSWORD":          "password",
    "DB_HOST":              "localhost",
    "DB_PORT":              "3306",
}
for _k, _v in _VERIFY_DEFAULTS.items():
    if not os.environ.get(_k):
        os.environ[_k] = _v

# Trick momentag's settings into using the SQLite test path
# (it switches to SQLite when "test" appears in sys.argv)
import sys as _sys
if "test" not in _sys.argv:
    _sys.argv.append("test")

from config.settings import *  # noqa: F401, F403, E402

# Override with a real file-based SQLite (not :memory:) so data persists
_DB_PATH = pathlib.Path(__file__).resolve().parent / "momentag_verify.sqlite3"
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(_DB_PATH),
    }
}
