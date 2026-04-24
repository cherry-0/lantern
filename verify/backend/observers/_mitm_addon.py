"""
mitmproxy addon — writes one JSON record per completed flow to the path given
in MITM_FLOW_LOG. Runs inside mitmdump's process (not the verify process), so
it must only depend on stdlib + mitmproxy.

Started by NetworkObserver as:
    mitmdump -p <port> -s _mitm_addon.py --set flow_detail=0
with env MITM_FLOW_LOG=<path>.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from mitmproxy import http


_LOG_PATH = os.environ.get("MITM_FLOW_LOG", "/tmp/verify_mitm_flows.jsonl")
_KEEP_JSON_ONLY = True      # only retain response body for application/json content-types


def _decode_body(b: bytes) -> str:
    if not b:
        return ""
    try:
        return b.decode("utf-8", errors="replace")
    except Exception:
        return f"<{len(b)} bytes binary>"


def response(flow: http.HTTPFlow) -> None:
    req = flow.request
    res = flow.response
    if res is None:
        return

    content_type = res.headers.get("content-type", "")
    keep_body = (not _KEEP_JSON_ONLY) or ("json" in content_type.lower())

    record: Dict[str, Any] = {
        "ts": time.time(),
        "method": req.method,
        "scheme": req.scheme,
        "host": req.pretty_host,
        "port": req.port,
        "path": req.path,
        "url": f"{req.scheme}://{req.pretty_host}:{req.port}{req.path}",
        "status": res.status_code,
        "req_content_type": req.headers.get("content-type", ""),
        "req_bytes": len(req.raw_content or b""),
        "res_content_type": content_type,
        "res_bytes": len(res.raw_content or b""),
        "req_body": _decode_body(req.raw_content or b""),
        "res_body": _decode_body(res.raw_content or b"") if keep_body else "",
    }

    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # Never let logging failure break the proxy.
        pass
