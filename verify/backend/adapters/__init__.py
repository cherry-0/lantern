# Adapters package — one module per target app
from typing import Dict, Optional, Type

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.adapters.momentag import MomentagAdapter
from verify.backend.adapters.clone import CloneAdapter
from verify.backend.adapters.snapdo import SnapdoAdapter
from verify.backend.adapters.xend import XendAdapter

# Registry: name → adapter class
ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {
    "momentag": MomentagAdapter,
    "clone": CloneAdapter,
    "snapdo": SnapdoAdapter,
    "xend": XendAdapter,
}


def get_adapter(app_name: str) -> Optional[BaseAdapter]:
    """Return an instantiated adapter for the given app name, or None."""
    cls = ADAPTER_REGISTRY.get(app_name)
    if cls is None:
        return None
    return cls()
