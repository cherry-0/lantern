"""
Base adapter interface for Verify.

All target-app adapters must subclass BaseAdapter and implement the three required methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AdapterResult:
    """Normalized result from a target-app pipeline run."""

    success: bool
    # Human-readable output (text summary of what the app produced)
    output_text: str = ""
    # Raw output dict as returned by the app's pipeline
    raw_output: Dict[str, Any] = field(default_factory=dict)
    # Optional structured output fields (tags, verdict, subject/body, etc.)
    structured_output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output_text": self.output_text,
            "raw_output": self.raw_output,
            "structured_output": self.structured_output,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseAdapter(ABC):
    """
    Abstract base class for all target-app adapters.

    Subclasses wrap the specific AI pipeline of each app and expose
    a uniform interface to the Verify orchestrator.
    """

    # Human-readable app name
    name: str = "base"

    # Modalities this adapter can process: "image", "text", "video"
    supported_modalities: List[str] = []

    @abstractmethod
    def check_availability(self) -> Tuple[bool, str]:
        """
        Check whether the adapter's required dependencies are available.

        Returns:
            (available: bool, reason: str)
            If available is False, reason explains what is missing.
        """

    @abstractmethod
    def run_pipeline(self, input_item: Dict[str, Any]) -> AdapterResult:
        """
        Run the target app's main AI pipeline on the given input item.

        Args:
            input_item: dict with at least:
                - "modality": "image" | "text" | "video"
                - "path": str — absolute path to the source file
                - "filename": str — display name
                - "data": loaded data (PIL.Image, str, list[PIL.Image] for video frames)
                - optional app-specific extra keys

        Returns:
            AdapterResult with success flag and output.
        """

    def get_display_label(self) -> str:
        """Return a short display label (used in UI dropdowns)."""
        return self.name

    def supports_modality(self, modality: str) -> bool:
        return modality in self.supported_modalities
