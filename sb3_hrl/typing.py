from __future__ import annotations

from typing import Generic, Protocol

import numpy as np
from gymnasium.core import ActType, ObsType


class SupportsPredict(Protocol, Generic[ObsType, ActType]):
    """Structural type for SB3-like policies exposing ``predict``."""

    def predict(
        self,
        observation: ObsType,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> ActType:
        """Return action and optional policy state."""
        ...


SB3ObsType = np.ndarray | dict[str, np.ndarray]
