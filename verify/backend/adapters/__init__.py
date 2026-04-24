# Adapters package — one module per target app
from typing import Dict, Optional, Type

from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.adapters.momentag import MomentagAdapter
from verify.backend.adapters.clone import CloneAdapter
from verify.backend.adapters.snapdo import SnapdoAdapter
from verify.backend.adapters.xend import XendAdapter
from verify.backend.adapters.budgetlens import BudgetLensAdapter
from verify.backend.adapters.deeptutor import DeepTutorAdapter
from verify.backend.adapters.llmvtuber import LLMVTuberAdapter
from verify.backend.adapters.skindisease import SkinDiseaseAdapter
from verify.backend.adapters.googleaiedge import GoogleAIEdgeAdapter
from verify.backend.adapters.toolneuron import ToolNeuronAdapter
from verify.backend.adapters.chatexpensetracker import ChatExpenseTrackerAdapter
from verify.backend.adapters.photomath_blackbox import PhotomathAdapter
from verify.backend.adapters.replika_blackbox import ReplikaAdapter
from verify.backend.adapters.expensify_blackbox import ExpensifyAdapter

# Registry: name → adapter class
ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {
    "momentag": MomentagAdapter,
    "clone": CloneAdapter,
    "snapdo": SnapdoAdapter,
    "xend": XendAdapter,
    "budget-lens": BudgetLensAdapter,
    "deeptutor": DeepTutorAdapter,
    "llm-vtuber": LLMVTuberAdapter,
    "skin-disease-detection": SkinDiseaseAdapter,
    "google-ai-edge-gallery": GoogleAIEdgeAdapter,
    "tool-neuron": ToolNeuronAdapter,
    "chat-driven-expense-tracker": ChatExpenseTrackerAdapter,
    "photomath": PhotomathAdapter,
    "replika": ReplikaAdapter,
    "expensify": ExpensifyAdapter,
}


def get_adapter(app_name: str) -> Optional[BaseAdapter]:
    """Return an instantiated adapter for the given app name, or None."""
    cls = ADAPTER_REGISTRY.get(app_name)
    if cls is None:
        return None
    return cls()
