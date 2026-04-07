"""
Clone runner — executed inside the 'clone' conda env.

Input JSON keys:
  frames_base64      list[str]  base64-encoded JPEG frames (>=1); or
  image_base64       str        single frame (used when frames_base64 is absent)
  openrouter_api_key str
  model              str        (optional, default: google/gemini-2.5-pro)
  filename           str        (optional) label for the chat session
  path               str        (optional) source path for log display

Output JSON:
  success     bool
  description str
  activity    str
  details     str
  summary     str
  session_id  int | null
  externalizations dict
  error       str | null
"""
import json
import os
import sys
import time
import traceback


_FRAME_PROMPT = (
    "You are an AI personal assistant analyzing screen activity recordings "
    "(like the clone app). You receive sampled frames from a video or screenshot. "
    "Describe:\n"
    "1. What activity/scene is shown.\n"
    "2. Any visible text, applications, or identifiable elements.\n"
    "3. A concise summary suitable for a personal knowledge base.\n\n"
    "Format:\n"
    "Activity: <description>\n"
    "Details: <visible elements>\n"
    "Summary: <1-2 sentence summary>"
)


def _parse_description(text: str):
    activity, details, summary = "", "", ""
    for line in text.splitlines():
        lower = line.lower()
        if lower.startswith("activity:"):
            activity = line.split(":", 1)[1].strip()
        elif lower.startswith("details:"):
            details = line.split(":", 1)[1].strip()
        elif lower.startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
    return activity, details, summary


def main():
    input_path = sys.argv[1]
    with open(input_path) as f:
        data = json.load(f)

    frames_b64: list = data.get("frames_base64", [])
    if not frames_b64 and data.get("image_base64"):
        frames_b64 = [data["image_base64"]]
    api_key: str = data.get("openrouter_api_key", "")
    model: str = data.get("model", "google/gemini-2.5-pro")
    filename: str = data.get("filename", "verify-input")

    # ── Bootstrap paths ────────────────────────────────────────────────────────
    runners_dir = os.path.dirname(os.path.abspath(__file__))
    clone_server = os.path.normpath(os.path.join(
        runners_dir, "..", "..", "..", "target-apps", "clone", "server"
    ))
    sys.path.insert(0, clone_server)
    sys.path.insert(0, runners_dir)
    import _runtime_capture
    _runtime_capture.install()

    from _runner_log import log_input
    log_input("clone", "image", data.get("path", f"<{len(frames_b64)} frame(s)>"))

    # ── Load clone .env ────────────────────────────────────────────────────────
    env_file = os.path.join(clone_server, ".env")
    env_example = os.path.join(clone_server, ".env_example")
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

    # ── Bootstrap Django ───────────────────────────────────────────────────────
    os.environ["DJANGO_SETTINGS_MODULE"] = "_clone_verify_settings"
    import django
    django.setup()
    _runtime_capture.connect_django_signals()

    from django.conf import settings as _django_settings
    from django.core.management import call_command
    db_path = _django_settings.DATABASES["default"]["NAME"]
    if not os.path.exists(db_path):
        print("[clone] Creating SQLite database (first run) ...", file=sys.stderr, flush=True)
        call_command("migrate", "--run-syncdb", verbosity=0)
        print("[clone] Database ready.", file=sys.stderr, flush=True)

    # ── Get or create verify user ──────────────────────────────────────────────
    print("[clone] Setting up verify user ...", file=sys.stderr, flush=True)
    from user.models import User
    from chat.models import ChatMessage, ChatSession

    _email = "verify@lantern.local"
    try:
        user = User.objects.get(email=_email)
    except User.DoesNotExist:
        user = User.objects.create_user(
            email=_email, username="verify_lantern", password="Verify_lantern_123!"
        )

    # ── Create chat session ────────────────────────────────────────────────────
    session = ChatSession.objects.create(
        user=user,
        title=f"Verify: {filename[:80]}",
        last_message_timestamp=int(time.time() * 1000),
    )
    session_id = session.id

    # ── OpenRouter vision call ─────────────────────────────────────────────────
    print("[clone] Running vision inference via OpenRouter ...", file=sys.stderr, flush=True)
    import requests as _requests

    content = [{"type": "text", "text": _FRAME_PROMPT}]
    for b64 in frames_b64:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )

    resp = _requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Verify",
            "X-Title": "Verify",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 512,
        },
        timeout=60,
    )
    resp.raise_for_status()
    description = resp.json()["choices"][0]["message"]["content"]
    print("[clone] Inference complete.", file=sys.stderr, flush=True)

    # ── Persist message ────────────────────────────────────────────────────────
    ts = int(time.time() * 1000)
    ChatMessage.objects.create(session=session, role="user", content=description, timestamp=ts)
    session.last_message_timestamp = ts
    session.save(update_fields=["last_message_timestamp"])

    activity, details, summary = _parse_description(description)

    externalizations = _runtime_capture.finalize()

    print(json.dumps({
        "success": True,
        "description": description,
        "activity": activity,
        "details": details,
        "summary": summary,
        "session_id": session_id,
        "externalizations": externalizations,
        "error": None,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({
            "success": False,
            "description": "",
            "activity": "",
            "details": "",
            "summary": "",
            "session_id": None,
            "error": traceback.format_exc(),
        }))
        sys.exit(1)
