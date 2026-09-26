from __future__ import annotations

from typing import Any, Generic, Protocol

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
    ) -> tuple[ActType, Any]:
        """Return action and optional policy state."""
        ...

    @classmethod
    def load(
        cls, path: str, device: str | None = None
    ) -> SupportsPredict[ObsType, ActType]:
        """Load a policy from file."""
        ...


SB3ObsType = np.ndarray | dict[str, np.ndarray]
