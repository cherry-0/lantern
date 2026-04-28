#!/usr/bin/env python3
"""
Download a GGUF model for ToolNeuron and optionally wire it into .env.

ToolNeuron's README says the app can load any local GGUF file and recommends
downloading one from Hugging Face, typically from a `*-GGUF` repo with a
quantization like `Q4_K_M`.

Examples:
  python scripts/download_toolneuron_gguf.py \
    --repo bartowski/Phi-3.5-mini-instruct-GGUF

  python scripts/download_toolneuron_gguf.py \
    --repo bartowski/Phi-3.5-mini-instruct-GGUF \
    --file Phi-3.5-mini-instruct-Q4_K_M.gguf \
    --set-env

  python scripts/download_toolneuron_gguf.py \
    --url https://huggingface.co/OWNER/REPO/resolve/main/model.gguf \
    --set-env
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "models" / "tool-neuron"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
HF_BASE = "https://huggingface.co"
HF_API_BASE = "https://huggingface.co/api"


def _headers() -> dict[str, str]:
    headers = {"User-Agent": "Lantern-ToolNeuron-GGUF-Downloader/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_json(url: str) -> object:
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return json.loads(resp.read().decode(charset))


def _list_repo_files(repo: str) -> list[str]:
    encoded_repo = urllib.parse.quote(repo, safe="/")
    url = f"{HF_API_BASE}/models/{encoded_repo}/tree/main?recursive=true&expand=false"
    data = _request_json(url)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected Hugging Face API response for repo {repo!r}")

    files: list[str] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        typ = entry.get("type")
        if isinstance(path, str) and typ == "file":
            files.append(path)
    return files


def _score_gguf_name(name: str) -> tuple[int, int, int, str]:
    upper = name.upper()
    preferred_quant_order = [
        "Q4_K_M",
        "Q4_K_S",
        "Q5_K_M",
        "Q5_K_S",
        "Q6_K",
        "Q8_0",
        "F16",
    ]
    quant_score = 0
    for idx, quant in enumerate(preferred_quant_order):
        if quant in upper:
            quant_score = len(preferred_quant_order) - idx
            break

    instruct_score = 1 if "INSTRUCT" in upper or "CHAT" in upper else 0
    size_hint = 0
    m = re.search(r"(\d+(?:\.\d+)?)B", upper)
    if m:
        try:
            size_hint = int(float(m.group(1)) * 10)
        except ValueError:
            size_hint = 0

    return (quant_score, instruct_score, size_hint, name)


def _pick_gguf_file(files: Iterable[str], pattern: str = "") -> str:
    candidates = [f for f in files if f.lower().endswith(".gguf")]
    if pattern:
        pattern_lower = pattern.lower()
        candidates = [f for f in candidates if pattern_lower in f.lower()]
    if not candidates:
        raise RuntimeError("No matching .gguf files found in the repo.")
    return sorted(candidates, key=_score_gguf_name, reverse=True)[0]


def _build_resolve_url(repo: str, filename: str) -> str:
    encoded_repo = urllib.parse.quote(repo, safe="/")
    encoded_file = "/".join(urllib.parse.quote(part, safe="") for part in filename.split("/"))
    return f"{HF_BASE}/{encoded_repo}/resolve/main/{encoded_file}?download=true"


def _infer_filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    if not name:
        raise RuntimeError(f"Could not infer filename from URL: {url}")
    return name


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=_headers())

    with urllib.request.urlopen(req) as resp, tmp_dest.open("wb") as out:
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        downloaded = 0
        chunk_size = 1024 * 1024

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                pct = downloaded / total_bytes * 100
                sys.stderr.write(
                    f"\rDownloading {dest.name}: {downloaded / (1024**2):.1f} MiB / "
                    f"{total_bytes / (1024**2):.1f} MiB ({pct:.1f}%)"
                )
            else:
                sys.stderr.write(
                    f"\rDownloading {dest.name}: {downloaded / (1024**2):.1f} MiB"
                )
            sys.stderr.flush()

    tmp_dest.replace(dest)
    sys.stderr.write("\n")
    sys.stderr.flush()


def _upsert_env_line(env_path: Path, key: str, value: str) -> None:
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()

    new_line = f"{key}={value}"
    updated = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        existing_key = stripped.split("=", 1)[0].strip()
        if existing_key == key:
            lines[idx] = new_line
            updated = True
            break

    if not updated:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(new_line)

    env_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a GGUF model for ToolNeuron and optionally update .env."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="Hugging Face repo id, e.g. bartowski/Phi-3.5-mini-instruct-GGUF")
    src.add_argument("--url", help="Direct URL to a .gguf file")
    parser.add_argument("--file", help="Exact .gguf filename within the Hugging Face repo")
    parser.add_argument(
        "--match",
        default="",
        help="Substring filter when auto-selecting a GGUF from the repo, e.g. Q4_K_M",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Directory to store the downloaded GGUF (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--set-env",
        action="store_true",
        help="Write TOOL_NEURON_GGUF_MODEL_PATH to the repo .env file",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help=f"Path to .env file to update (default: {DEFAULT_ENV_FILE})",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print the TOOL_NEURON_GGUF_MODEL_PATH line after download",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List available GGUF files in the repo and exit",
    )

    args = parser.parse_args()

    try:
        if args.repo:
            files = _list_repo_files(args.repo)
            gguf_files = [f for f in files if f.lower().endswith(".gguf")]
            if not gguf_files:
                raise RuntimeError(f"No .gguf files found in repo {args.repo!r}.")

            if args.list_only:
                for name in gguf_files:
                    print(name)
                return 0

            selected_file = args.file or _pick_gguf_file(gguf_files, args.match)
            url = _build_resolve_url(args.repo, selected_file)
            filename = Path(selected_file).name
        else:
            url = args.url
            filename = _infer_filename_from_url(url)

        if not filename.lower().endswith(".gguf"):
            raise RuntimeError(f"Target file is not a .gguf: {filename}")

        out_dir = Path(args.out_dir).expanduser().resolve()
        dest = out_dir / filename

        print(f"Downloading to: {dest}")
        if args.repo:
            print(f"Source repo: {args.repo}")
            print(f"Selected file: {selected_file}")
        else:
            print(f"Source URL: {url}")

        _download(url, dest)

        env_line = f"TOOL_NEURON_GGUF_MODEL_PATH={dest}"
        if args.set_env:
            env_path = Path(args.env_file).expanduser().resolve()
            _upsert_env_line(env_path, "TOOL_NEURON_GGUF_MODEL_PATH", str(dest))
            print(f"Updated env file: {env_path}")

        if args.print_env or args.set_env:
            print(env_line)

        print("Done.")
        return 0

    except urllib.error.HTTPError as exc:
        print(f"HTTP error: {exc.code} {exc.reason}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Network error: {exc.reason}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
