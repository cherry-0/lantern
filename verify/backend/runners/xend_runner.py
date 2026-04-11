"""
Xend runner — executed inside the 'xend' conda env.

Input JSON keys:
  text_content       str   raw email content / scenario
  openrouter_api_key str
  model              str   (optional)

Output JSON:
  success  bool
  subject  str
  body     str
  error    str | null
"""
import json
import os
import sys
import traceback


def main():
    input_path = sys.argv[1]
    with open(input_path) as f:
        data = json.load(f)

    text_content: str = data.get("text_content", "")
    api_key: str = data.get("openrouter_api_key", "")
    model: str = data.get("model", "google/gemini-2.5-pro")

    # ── Inject env vars before any LangChain import ───────────────────────────
    base_url = "https://openrouter.ai/api/v1"
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("OPENAI_BASE_URL", base_url)
    os.environ.setdefault("OPENAI_MODEL", model)

    # ── Bootstrap Django ───────────────────────────────────────────────────────
    runners_dir = os.path.dirname(os.path.abspath(__file__))
    xend_backend = os.path.normpath(os.path.join(
        runners_dir, "..", "..", "..", "target-apps", "xend", "backend"
    ))
    # runners_dir must come first so _xend_verify_settings is importable
    sys.path.insert(0, xend_backend)
    sys.path.insert(0, runners_dir)
    import _runtime_capture
    _runtime_capture.install()

    # Load xend .env first (real values take precedence over dummy defaults)
    env_file = os.path.join(xend_backend, ".env")
    env_example = os.path.join(xend_backend, ".env_example")
    if not os.path.exists(env_file) and os.path.exists(env_example):
        import shutil
        shutil.copyfile(env_example, env_file)
    if os.path.exists(env_file):
        for line in open(env_file).read().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

    # Use verify-specific settings: SQLite DB + dummy env var fallbacks
    os.environ["DJANGO_SETTINGS_MODULE"] = "_xend_verify_settings"
    import django
    django.setup()
    _runtime_capture.connect_django_signals()

    # Create the SQLite schema on first run (fast no-op on subsequent runs)
    from django.conf import settings as _django_settings
    from django.core.management import call_command
    db_path = _django_settings.DATABASES["default"]["NAME"]
    if not os.path.exists(db_path):
        print("[xend] Creating SQLite database (first run) ...", file=sys.stderr, flush=True)
        call_command("migrate", "--run-syncdb", verbosity=0)
        print("[xend] Database ready.", file=sys.stderr, flush=True)

    from _runner_log import log_input
    log_input("xend", "text", text_content)
    print("[xend] Loading LangChain chains ...", file=sys.stderr, flush=True)
    from apps.ai.services.chains import body_chain, subject_chain

    inputs = {
        "body": text_content,
        "subject": "",
        "language": "en",
        "recipients": "",
        "group_name": "",
        "group_description": "",
        "prompt_text": "",
        "sender_role": "",
        "recipient_role": "",
        "plan_text": "",
        "analysis": None,
        "fewshots": None,
        "profile": "",
        "attachments": [],
        "locked_subject": "",
    }

    print("[xend] Running subject_chain ...", file=sys.stderr, flush=True)
    subject = (subject_chain.invoke(inputs) or "").strip()
    inputs["locked_subject"] = subject
    print("[xend] Running body_chain ...", file=sys.stderr, flush=True)
    body = (body_chain.invoke(inputs) or "").strip()
    print("[xend] Inference complete.", file=sys.stderr, flush=True)

    # ── Post-Inference Externalization ────────────────────────────────────────
    _runtime_capture.set_phase("POST")
    print("[xend] Starting post-inference actions (Sending email)...", file=sys.stderr, flush=True)

    try:
        from apps.mail.services import send_email_logic
        # Use a dummy token; the service will still attempt the call, 
        # which our urllib3 patch will capture as [POST] googleapis.com → 401
        send_email_logic(
            access_token="dummy-verify-token",
            to=["recipient@example.com"],
            subject=subject,
            body=body,
            is_html=False
        )
    except Exception as e:
        # We expect a 401/Invalid Token error, which is fine—the network call 
        # is what we want to capture.
        print(f"[xend] Post-inference action (send_email) triggered: {type(e).__name__}", file=sys.stderr)

    _runtime_capture.record_ui_event("NOTIFICATION", f"Email sent: {subject[:50]}...")
    print("[xend] Post-inference complete.", file=sys.stderr, flush=True)

    externalizations = _runtime_capture.finalize()

    print(json.dumps({
        "success": True,
        "subject": subject,
        "body": body,
        "externalizations": externalizations,
        "error": None,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"success": False, "subject": "", "body": "", "error": traceback.format_exc()}))
        sys.exit(1)
