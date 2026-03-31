from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class SupportsPredict(Protocol):
    """Structural type for SB3-like policies exposing ``predict``."""

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        """Return action and optional policy state."""


SB3ObsType = np.ndarray | dict[str, np.ndarray]
