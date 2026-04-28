import json
from pathlib import Path

root = Path("verify/outputs")
failed = []

for f in root.glob("*/**/*.json"):
    if f.name in {"run_config.json", "dir_summary.json", "report.json"}:
        continue
    try:
        item = json.loads(f.read_text())
    except Exception:
        continue

    if item.get("status") == "success" and item.get("ext_eval_ok") is False:
        failed.append((f, item.get("filename"), item.get("eval_model"), item.get("eval_prompt"), item.get("ext_eval_error")))

print(f"eval failures: {len(failed)}")
for row in failed[:100]:
    path, filename, model, prompt, err = row
    print(f"\n{path}")
    print(f"  filename: {filename}")
    print(f"  model: {model}")
    print(f"  prompt: {prompt}")
    print(f"  error: {err}")
