"""
sync_delete.py — Propagate local deletions to Google Drive.

Compares local verify/outputs/ against the remote and deletes any remote
directories that no longer exist locally (i.e., were deleted locally since
the last upload).

Safe by default: runs in dry-run mode and only prints what would be deleted.
Pass --execute to actually delete.

Usage:
    python sync_delete.py              # dry-run: show what would be deleted
    python sync_delete.py --execute    # actually delete remote-only dirs
    python sync_delete.py --verbose    # also list local-only (not yet uploaded) dirs

Environment variables (can be set in verify/.env):
    GDRIVE_REMOTE       rclone remote name       (default: lantern)
    GDRIVE_REMOTE_PATH  path inside the remote   (default: verify/outputs)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_OUTPUTS_DIR    = Path(__file__).resolve().parent.parent / "outputs"
_DEFAULT_REMOTE = "lantern"
_DEFAULT_PATH   = "verify/outputs"


def _get_remote() -> str:
    remote = os.getenv("GDRIVE_REMOTE", _DEFAULT_REMOTE)
    path   = os.getenv("GDRIVE_REMOTE_PATH", _DEFAULT_PATH)
    return f"{remote}:{path}"


def _rclone_list_dirs(target: str) -> set[str]:
    """Return the set of top-level directory names inside *target*."""
    result = subprocess.run(
        ["rclone", "lsf", target, "--dirs-only"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[ERR] rclone lsf failed:\n{result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return {line.rstrip("/") for line in result.stdout.splitlines() if line.strip()}


def _rclone_purge(remote_dir: str, *, dry_run: bool) -> int:
    """Delete an entire remote directory tree.  Prints what it does."""
    if dry_run:
        print(f"  [would delete]  {remote_dir}")
        return 0
    print(f"  Deleting  {remote_dir} …", flush=True)
    result = subprocess.run(["rclone", "purge", remote_dir])
    if result.returncode != 0:
        print(f"  [ERR] purge failed for {remote_dir}", file=sys.stderr)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually delete remote-only dirs; default is dry-run (no writes)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Also list local-only dirs (uploaded locally but not yet on remote)",
    )
    args = parser.parse_args()

    dry_run = not args.execute
    remote  = _get_remote()

    print(f"\n[sync_delete]  mode={'DRY RUN' if dry_run else 'EXECUTE'}  remote={remote}\n")

    # ── Local dirs ────────────────────────────────────────────────────────────
    if not _OUTPUTS_DIR.exists():
        print(f"[ERR] Local outputs directory not found: {_OUTPUTS_DIR}", file=sys.stderr)
        sys.exit(1)
    local_dirs = {d.name for d in _OUTPUTS_DIR.iterdir() if d.is_dir()}

    # ── Remote dirs ───────────────────────────────────────────────────────────
    print("Fetching remote directory list…", flush=True)
    remote_dirs = _rclone_list_dirs(remote)

    # ── Diff ──────────────────────────────────────────────────────────────────
    remote_only = sorted(remote_dirs - local_dirs)   # deleted locally → delete remote
    local_only  = sorted(local_dirs  - remote_dirs)  # not yet uploaded
    in_sync     = remote_dirs & local_dirs

    print(f"  Local dirs   : {len(local_dirs)}")
    print(f"  Remote dirs  : {len(remote_dirs)}")
    print(f"  In sync      : {len(in_sync)}")
    print(f"  Local only   : {len(local_only)}  (not yet uploaded — ignored)")
    print(f"  Remote only  : {len(remote_only)}  (deleted locally → will be removed from Drive)")
    print()

    if args.verbose and local_only:
        print("Local-only dirs (not yet uploaded):")
        for d in local_only:
            print(f"  {d}")
        print()

    if not remote_only:
        print("Nothing to delete — remote is already consistent with local.")
        return

    # ── Delete ────────────────────────────────────────────────────────────────
    print(f"{'Would delete' if dry_run else 'Deleting'} {len(remote_only)} remote dir(s):\n")
    errors = 0
    for d in remote_only:
        rc = _rclone_purge(f"{remote}/{d}", dry_run=dry_run)
        if rc != 0:
            errors += 1

    print()
    if dry_run:
        print(f"Dry run complete — {len(remote_only)} dir(s) would be deleted.")
        print("Re-run with --execute to apply.")
    else:
        deleted = len(remote_only) - errors
        print(
            f"Done — {deleted} dir(s) deleted"
            + (f", {errors} error(s)." if errors else ".")
        )


if __name__ == "__main__":
    main()
