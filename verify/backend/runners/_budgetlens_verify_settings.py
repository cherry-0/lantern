"""
Minimal Django settings for budget-lens — Verify framework use only.

Overrides the database to SQLite so the runner can call process_receipt()
without a running PostgreSQL server.

Requires budgetlens_root (target-apps/budget-lens/budgetlens) on sys.path.
"""
import os
import pathlib

# budget-lens uses os.getenv() with no strict required vars, so setdefault works.
_VERIFY_DEFAULTS = {
    "SECRET_KEY":   "verify-only-dummy-secret-key-budgetlens",
    "DEBUG":        "True",
    "DB_NAME":      "verify",
    "DB_USER":      "postgres",
    "DB_PASSWORD":  "postgres",
    "DB_HOST":      "localhost",
    "DB_PORT":      "5432",
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "dummy"),
}
for _k, _v in _VERIFY_DEFAULTS.items():
    if not os.environ.get(_k):
        os.environ[_k] = _v

from budgetlens.settings import *  # noqa: F401, F403, E402

_DB_PATH = pathlib.Path(__file__).resolve().parent / "budgetlens_verify.sqlite3"
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(_DB_PATH),
    }
}
