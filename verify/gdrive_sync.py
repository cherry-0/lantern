"""
Google Drive sync for verify/outputs/ via rclone.

Usage:
  python gdrive_sync.py upload    # push local outputs/ → Drive (skips cache_*)
  python gdrive_sync.py download  # pull Drive → local (only if Drive file is newer)

Requirements:
  rclone configured with a remote named "lantern" pointing to your Google Drive.
  Check with: rclone listremotes

Environment variables:
  GDRIVE_REMOTE       rclone remote name (default: lantern)
  GDRIVE_REMOTE_PATH  path inside the remote (default: verify/outputs)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent / "outputs"

DEFAULT_REMOTE = "lantern"
DEFAULT_REMOTE_PATH = "verify/outputs"


def rclone(*args: str) -> int:
    cmd = ["rclone", *args]
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def get_remote() -> str:
    remote = os.getenv("GDRIVE_REMOTE", DEFAULT_REMOTE)
    path = os.getenv("GDRIVE_REMOTE_PATH", DEFAULT_REMOTE_PATH)
    return f"{remote}:{path}"


def cmd_upload():
    remote = get_remote()
    rc = rclone(
        "copy", str(OUTPUTS_DIR), remote,
        "--update",              # skip if destination is newer
        "--progress",
    )
    if rc != 0:
        print(f"[ERR] rclone exited with code {rc}")
        sys.exit(rc)
    print("Upload complete.")


def cmd_download():
    remote = get_remote()
    rc = rclone(
        "copy", remote, str(OUTPUTS_DIR),
        "--update",              # skip if local file is newer
        "--progress",
    )
    if rc != 0:
        print(f"[ERR] rclone exited with code {rc}")
        sys.exit(rc)
    print("Download complete.")


def main():
    parser = argparse.ArgumentParser(description="Sync verify/outputs/ with Google Drive via rclone")
    parser.add_argument("command", choices=["upload", "download"])
    args = parser.parse_args()

    if args.command == "upload":
        cmd_upload()
    else:
        cmd_download()


if __name__ == "__main__":
    main()
