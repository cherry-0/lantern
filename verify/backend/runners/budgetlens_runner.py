"""
Budget-lens runner — executed inside the 'budget-lens' conda env.

Input JSON keys:
  image_base64       str   base64-encoded JPEG
  openrouter_api_key str

Output JSON:
  success  bool
  category str
  date     str
  amount   float | null
  currency str
  error    str | null
"""
import base64
import io
import json
import os
import sys
import tempfile
import traceback


def main():
    input_path = sys.argv[1]
    with open(input_path) as f:
        data = json.load(f)

    image_b64: str = data["image_base64"]
    api_key: str = data.get("openrouter_api_key", "")

    # ── Inject env vars ────────────────────────────────────────────────────────
    base_url = "https://openrouter.ai/api/v1"
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("OPENAI_BASE_URL", base_url)

    # ── Bootstrap Django ───────────────────────────────────────────────────────
    runners_dir = os.path.dirname(os.path.abspath(__file__))
    budgetlens_root = os.path.normpath(os.path.join(
        runners_dir, "..", "..", "..", "target-apps", "budget-lens", "budgetlens"
    ))
    sys.path.insert(0, budgetlens_root)
    sys.path.insert(0, runners_dir)
    import _runtime_capture
    _runtime_capture.install()

    # Use verify-specific settings: SQLite DB + dummy env var fallbacks
    os.environ["DJANGO_SETTINGS_MODULE"] = "_budgetlens_verify_settings"
    import django
    django.setup()
    _runtime_capture.connect_django_signals()

    # Create the SQLite schema on first run
    from django.conf import settings as _django_settings
    from django.core.management import call_command
    db_path = _django_settings.DATABASES["default"]["NAME"]
    if not os.path.exists(db_path):
        print("[budget-lens] Creating SQLite database (first run) ...", file=sys.stderr, flush=True)
        call_command("migrate", "--run-syncdb", verbosity=0)
        print("[budget-lens] Database ready.", file=sys.stderr, flush=True)

    from _runner_log import log_input
    log_input("budget-lens", "image", data.get("path", "<base64>"))

    # Write image to temp file (process_receipt expects a path)
    img_bytes = base64.b64decode(image_b64)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.write(img_bytes)
    tmp.close()

    try:
        print("[budget-lens] Loading process_receipt ...", file=sys.stderr, flush=True)
        from core.views import process_receipt
        print("[budget-lens] Running receipt extraction ...", file=sys.stderr, flush=True)
        category, expense_date, amount, currency = process_receipt(tmp.name)
        print("[budget-lens] Inference complete.", file=sys.stderr, flush=True)
        _runtime_capture.set_phase("POST")
    finally:
        os.unlink(tmp.name)

    externalizations = _runtime_capture.finalize()

    print(json.dumps({
        "success": True,
        "category": category,
        "date": str(expense_date),
        "amount": float(amount) if amount is not None else None,
        "currency": currency,
        "externalizations": externalizations,
        "error": None,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"success": False, "error": traceback.format_exc()}))
        sys.exit(1)
