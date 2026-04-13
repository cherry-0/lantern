"""
Momentag runner — executed inside the 'momentag' conda env.

Input JSON keys:
  image_base64  str   base64-encoded JPEG

Output JSON:
  success   bool
  captions  list[str]
  tags      list[str]
  error     str | null
"""
import base64
import io
import json
import os
import sys
import traceback


def main():
    input_path = sys.argv[1]
    with open(input_path) as f:
        data = json.load(f)

    image_b64: str = data["image_base64"]

    # Decode image
    img_bytes = base64.b64decode(image_b64)
    from PIL import Image as PILImage
    pil_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

    # ── Bootstrap Django ───────────────────────────────────────────────────────
    runners_dir = os.path.dirname(os.path.abspath(__file__))
    momentag_backend = os.path.normpath(os.path.join(
        runners_dir, "..", "..", "..", "target-apps", "momentag", "backend"
    ))
    sys.path.insert(0, momentag_backend)
    sys.path.insert(0, runners_dir)
    import _runtime_capture
    _runtime_capture.install()

    os.environ["DJANGO_SETTINGS_MODULE"] = "_momentag_verify_settings"
    import django
    django.setup()
    _runtime_capture.connect_django_signals()

    from django.conf import settings as _django_settings
    from django.core.management import call_command
    db_path = _django_settings.DATABASES["default"]["NAME"]
    if not os.path.exists(db_path):
        print("[momentag] Creating SQLite database (first run) ...", file=sys.stderr, flush=True)
        call_command("migrate", "--run-syncdb", verbosity=0)
        print("[momentag] Database ready.", file=sys.stderr, flush=True)

    from _runner_log import log_input
    log_input("momentag", "image", data.get("path", "<base64>"))
    print("[momentag] Loading CLIP + BLIP models ...", file=sys.stderr, flush=True)
    from gallery.gpu_tasks import get_image_captions

    print("[momentag] Running get_image_captions ...", file=sys.stderr, flush=True)
    captions_data = get_image_captions(pil_image)
    print("[momentag] Inference complete.", file=sys.stderr, flush=True)
    _runtime_capture.set_phase("POST")

    captions, tags = [], []
    for item in captions_data:
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            captions.append(str(item[0]))
            if len(item) >= 2 and item[1]:
                tags.extend(item[1])
        elif isinstance(item, str):
            captions.append(item)

    tags = list(dict.fromkeys(tags))  # deduplicate

    externalizations = _runtime_capture.finalize()

    print(json.dumps({
        "success": True,
        "captions": captions,
        "tags": tags,
        "externalizations": externalizations,
        "error": None,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"success": False, "captions": [], "tags": [], "error": traceback.format_exc()}))
        sys.exit(1)
