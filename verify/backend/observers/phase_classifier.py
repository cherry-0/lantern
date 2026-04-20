"""
Phase classifier — replaces _runtime_capture.set_phase("POST") for black-box
adapters. Without source access we cannot call a function at the inference
boundary, so we reconstruct the boundary from network timestamps.

Heuristic:
  1. Scan NETWORK events for requests whose host matches a known inference-
     provider pattern (DEFAULT_LLM_HOSTS plus per-adapter additions).
  2. The *last response* from any matched host defines t* (inference boundary).
  3. Events with ts < t*  → DURING  (discarded)
     Events with ts >= t* → POST    (returned as externalizations)

If no LLM host is seen (e.g. the app proxies inference through its own
backend), fall back to the first response from `primary_backend_host`.
"""
from __future__ import annotations

import fnmatch
from typing import Any, Dict, List, Optional, Tuple


# Host patterns treated as LLM/AI inference providers. fnmatch-style globs.
DEFAULT_LLM_HOSTS: Tuple[str, ...] = (
    "*.openai.com",
    "*.anthropic.com",
    "generativelanguage.googleapis.com",
    "*.googleapis.com",               # broad; trimmed below via explicit excludes
    "openrouter.ai",
    "api.together.xyz",
    "api.mistral.ai",
    "api.groq.com",
    "*.azure.com",
    "*.cohere.ai",
    "api.replicate.com",
    "api.x.ai",
    "api.perplexity.ai",
    "api.deepseek.com",
)

# Subtract known-non-inference Google endpoints from the broad *.googleapis.com glob.
_GOOGLE_NON_INFERENCE = (
    "firebaseinstallations.googleapis.com",
    "firebaseremoteconfig.googleapis.com",
    "play.googleapis.com",
    "android.googleapis.com",
    "mobilecrashreporting.googleapis.com",
    "people-pa.googleapis.com",
)


def _host_matches_llm(host: str, patterns: Tuple[str, ...]) -> bool:
    if host in _GOOGLE_NON_INFERENCE:
        return False
    return any(fnmatch.fnmatch(host, p) for p in patterns)


def classify_phases(
    events: Dict[str, List[Dict[str, Any]]],
    llm_hosts: Tuple[str, ...] = DEFAULT_LLM_HOSTS,
    primary_backend_host: Optional[str] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """
    Split captured events into (during, post) using a network-timestamp boundary.

    Args:
        events: {"NETWORK": [...], "STORAGE": [...], "LOGGING": [...], "UI": [...]}
                Each channel is a list of dicts; network events must have a 'ts'
                field and a 'host' field.
        llm_hosts: fnmatch patterns for inference-provider hostnames.
        primary_backend_host: fallback host used when no LLM host is seen.

    Returns:
        (during, post) — same channel dict shape, partitioned by timestamp.
    """
    network = events.get("NETWORK", [])
    t_boundary = _find_inference_boundary(
        network, llm_hosts=llm_hosts, primary_backend_host=primary_backend_host
    )

    if t_boundary is None:
        # No recognizable inference call — treat everything as POST to avoid
        # dropping potentially useful externalizations silently.
        return ({k: [] for k in events}, dict(events))

    during: Dict[str, List[Dict[str, Any]]] = {}
    post: Dict[str, List[Dict[str, Any]]] = {}
    for channel, items in events.items():
        during[channel] = [e for e in items if _event_ts(e) < t_boundary]
        post[channel] = [e for e in items if _event_ts(e) >= t_boundary]
    return during, post


def _event_ts(event: Dict[str, Any]) -> float:
    """Extract a timestamp from a heterogeneous event dict; 0 if missing."""
    for key in ("ts", "timestamp", "time"):
        v = event.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def _find_inference_boundary(
    network_events: List[Dict[str, Any]],
    llm_hosts: Tuple[str, ...],
    primary_backend_host: Optional[str],
) -> Optional[float]:
    """
    Return the timestamp of the last LLM-host response, or the first backend-host
    response if no LLM host is matched. None if neither is found.
    """
    # Ordered by ts already (NetworkObserver sorts), but be defensive.
    ordered = sorted(network_events, key=_event_ts)

    last_llm: Optional[float] = None
    for e in ordered:
        host = e.get("host", "")
        if _host_matches_llm(host, llm_hosts):
            last_llm = _event_ts(e)
    if last_llm is not None:
        return last_llm

    if primary_backend_host:
        for e in ordered:
            host = e.get("host", "")
            if fnmatch.fnmatch(host, primary_backend_host):
                return _event_ts(e)

    return None
