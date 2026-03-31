from __future__ import annotations

from typing import Any, Protocol


class SupportsPredict(Protocol):
    """Structural type for SB3-like policies exposing ``predict``."""

    def predict(self, observation: Any, deterministic: bool = True):
        """Return action and optional policy state."""
