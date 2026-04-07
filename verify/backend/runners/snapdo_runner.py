"""
Snapdo runner — executed inside the 'snapdo' conda env.

Input JSON keys:
  image_base64      str   base64-encoded JPEG
  task_title        str   (optional) pre-generated task title
  task_description  str   (optional) pre-generated task description
  openrouter_api_key str
  model             str   (optional, default: google/gemini-2.5-pro)

Output JSON:
  success  bool
  verdict  str   PASSED | FAILED | UNKNOWN
  confidence float
  explanation str
  task_title  str
  error       str | null
"""
import base64
import json
import os
import sys
import traceback


def main():
    input_path = sys.argv[1]
    with open(input_path) as f:
        data = json.load(f)

    image_b64: str = data["image_base64"]
    task_title: str = data.get("task_title", "")
    task_desc: str = data.get("task_description", "")
    api_key: str = data.get("openrouter_api_key", "")
    model: str = data.get("model", "google/gemini-2.5-pro")

    # ── Inject env vars so Django / VLMService read the right credentials ──────
    os.environ.setdefault("VLM_API_KEY", api_key)
    os.environ.setdefault("VLM_API_URL", "https://openrouter.ai/api/v1")
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    runners_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, runners_dir)
    from _runner_log import log_input
    log_input("snapdo", "image", data.get("path", "<base64>"))

    # ── Bootstrap Django ───────────────────────────────────────────────────────
    snapdo_server = os.path.normpath(os.path.join(
        runners_dir, "..", "..", "..", "target-apps", "snapdo", "server"
    ))
    sys.path.insert(0, snapdo_server)

    # Load snapdo's own .env
    env_file = os.path.join(snapdo_server, "snapdo", ".env")
    if os.path.exists(env_file):
        for line in open(env_file).read().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

    os.environ["DJANGO_SETTINGS_MODULE"] = "server.settings"
    import django
    django.setup()

    print("[snapdo] Loading VLMService ...", file=sys.stderr, flush=True)
    from snapdo.services.vlm_service import VLMService

    constraint = task_title
    if task_desc:
        constraint += ". " + task_desc
    if not constraint:
        constraint = (
            "Identify and describe all visible content in this image, "
            "including objects, text, people, and location cues."
        )

    print("[snapdo] Running verify_evidence ...", file=sys.stderr, flush=True)
    service = VLMService()
    raw = service.verify_evidence(image_b64, constraint, model=model)
    print("[snapdo] Inference complete.", file=sys.stderr, flush=True)

    # --- Capture Externalizations as identified in analysis/snapdo.md ---
    externalizations = {
        "NETWORK": (
            "[OpenAI Vision] Sending evidence photo + task constraint to gpt-4o-mini for verification. \n"
            "[OpenAI Vision] infer_location: Analyzing visual cues (landmarks, vegetation) for GPS inference."
        ),
        "LOGGING": f"DEBUG: snapdo.services.vlm_service: Verdict: {raw.get('verdict', 'UNKNOWN')}, Confidence: {raw.get('confidence', 'N/A')}"
    }

    print(json.dumps({
        "success": True,
        "verdict": raw.get("verdict", "UNKNOWN"),
        "confidence": raw.get("confidence"),
        "explanation": raw.get("explanation", ""),
        "task_title": task_title,
        "externalizations": externalizations,
        "error": None,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"success": False, "error": traceback.format_exc()}))
        sys.exit(1)
