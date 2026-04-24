"""
UiObserver — samples the Android UI hierarchy before and after the driven
session. The diff (text nodes that appeared, image/resource nodes that became
visible) is the black-box analogue of the UI externalization channel captured
by _runtime_capture.py's Django/Channels patches for open-source adapters.

Sampling strategy (one baseline, one post-response):
  - `start()` captures the baseline once the observer context is entered.
    BlackBoxAdapter.run_pipeline enters the UI observer *before* launching the
    app, so "baseline" is typically the launcher/home screen — everything
    user-visible from the app itself counts as new.
  - `capture_post()` can be called by the adapter right after it has read its
    response, to record the exact moment when private information first
    appears on screen. Falls back to a capture on `stop()` if never called.

See verify_report_blackbox.md §6.4.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Set


_UI_MAX_EVENTS = 100
_UI_MIN_TEXT_LEN = 2


class UiObserver:
    """
    Context manager. Uses uiautomator2 (imported lazily) to dump the view
    hierarchy. Subclasses the black-box events contract:
        {"ts": float, "kind": "text"|"image"|"desc", "value": str, "resource_id": str}
    """

    def __init__(self, serial: str):
        self.serial = serial
        self.events: List[Dict[str, Any]] = []
        self._d = None
        self._baseline: Set[str] = set()
        self._post_dump_nodes: Optional[List[Dict[str, str]]] = None

    def __enter__(self) -> "UiObserver":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        try:
            import uiautomator2 as u2
        except ImportError:
            # Soft-fail: UI observer is best-effort. Other channels still work.
            self._d = None
            return
        try:
            self._d = u2.connect(self.serial)
            self._baseline = self._collect_keys(self._dump_nodes())
        except Exception:
            self._d = None

    def capture_post(self) -> None:
        """Call after the response is rendered to snapshot the final UI state."""
        if self._d is None:
            return
        try:
            self._post_dump_nodes = self._dump_nodes()
        except Exception:
            self._post_dump_nodes = []

    def stop(self) -> None:
        if self._d is None:
            self.events = []
            return
        if self._post_dump_nodes is None:
            try:
                self._post_dump_nodes = self._dump_nodes()
            except Exception:
                self._post_dump_nodes = []

        ts = time.time()
        seen: Set[str] = set()
        out: List[Dict[str, Any]] = []
        for node in self._post_dump_nodes or []:
            key = self._node_key(node)
            if not key or key in self._baseline or key in seen:
                continue
            seen.add(key)
            evt = self._node_to_event(node, ts)
            if evt:
                out.append(evt)
        self.events = out[:_UI_MAX_EVENTS]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _dump_nodes(self) -> List[Dict[str, str]]:
        """Return a flat list of {text, content-desc, resource-id, class} dicts."""
        xml = self._d.dump_hierarchy()
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return []
        nodes: List[Dict[str, str]] = []
        for el in root.iter("node"):
            nodes.append({
                "text": (el.get("text") or "").strip(),
                "desc": (el.get("content-desc") or "").strip(),
                "resource_id": el.get("resource-id") or "",
                "cls": el.get("class") or "",
                "visible": el.get("visible-to-user") or el.get("visible") or "true",
            })
        return nodes

    @staticmethod
    def _collect_keys(nodes: List[Dict[str, str]]) -> Set[str]:
        return {UiObserver._node_key(n) for n in nodes if UiObserver._node_key(n)}

    @staticmethod
    def _node_key(node: Dict[str, str]) -> str:
        """Stable identity so we can diff baseline vs post-response dumps."""
        text = node.get("text", "")
        desc = node.get("desc", "")
        rid = node.get("resource_id", "")
        if not (text or desc or rid):
            return ""
        return f"{rid}|{text}|{desc}"

    @staticmethod
    def _node_to_event(node: Dict[str, str], ts: float) -> Optional[Dict[str, Any]]:
        text = node.get("text", "")
        desc = node.get("desc", "")
        rid = node.get("resource_id", "")
        cls = node.get("cls", "")

        if text and len(text) >= _UI_MIN_TEXT_LEN:
            return {"ts": ts, "kind": "text", "value": text, "resource_id": rid}
        if desc and len(desc) >= _UI_MIN_TEXT_LEN:
            return {"ts": ts, "kind": "desc", "value": desc, "resource_id": rid}
        # Image/resource-only nodes — useful when an app silently renders e.g.
        # a cropped face thumbnail with no text label.
        if "Image" in cls and rid:
            return {"ts": ts, "kind": "image", "value": rid, "resource_id": rid}
        return None
