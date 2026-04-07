"""
Verify-only Django settings shim for the clone server.

Injects dummy env-var defaults so config.settings.local imports without errors,
then overrides DATABASES to a local SQLite file so no MySQL server is needed.
"""
import os
from pathlib import Path

_VERIFY_DEFAULTS = {
    "SECRET_KEY": "verify-only-dummy-secret-key-clone-not-for-production",
    "VECTORDB_CHAT_HOST": "http://localhost:6333",
    "VECTORDB_SCREEN_HOST": "http://localhost:6334",
    "SENDGRID_API_KEY": "SG.dummy",
    "SENDGRID_FROM_EMAIL": "noreply@verify.local",
    # DB vars (needed if base.py reads them at import time)
    "DB_NAME": "clone_verify",
    "DB_USERNAME": "verify",
    "DB_PASSWORD": "verify",
    "DB_HOSTNAME": "localhost",
    "DB_PORT": "3306",
}
for _k, _v in _VERIFY_DEFAULTS.items():
    if not os.environ.get(_k):
        os.environ[_k] = _v

_DB_PATH = Path(__file__).parent / "clone_verify.sqlite3"

from config.settings.local import *  # noqa: F401, E402

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(_DB_PATH),
    }
}
