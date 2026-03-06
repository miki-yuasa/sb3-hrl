"""Policy-side utilities for HIRO.

This module contains small environment-agnostic helpers used by the
HIRO algorithm implementation.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from gymnasium import spaces
from gymnasium.spaces import utils as space_utils


class SubgoalProjectionWrapper:
    """Wrap a state-to-goal projection callable.

    Parameters
    ----------
    projection_fn : callable, optional
            Callable that maps a flattened environment state to a subgoal vector.
            When ``None``, the identity map is used.

    Notes
    -----
    The wrapper standardizes dtype and shape validation so projection behavior
    is explicit and re-usable across rollout and replay relabeling code.
    """

    def __init__(
        self,
        projection_fn: Optional[
            Callable[[Union[np.ndarray, dict[str, np.ndarray]]], np.ndarray]
        ] = None,
        observation_space: Optional[spaces.Space] = None,
    ) -> None:
        self._projection_fn = projection_fn
        self._observation_space = observation_space

    def __call__(
        self, observation: Union[np.ndarray, dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Project an observation into subgoal space.

        Parameters
        ----------
        observation : np.ndarray | dict[str, np.ndarray]
                Environment observation in its original format.

        Returns
        -------
        np.ndarray
                Projected subgoal vector with ``dtype=float32``.
        """
        if self._projection_fn is None:
            if self._observation_space is not None:
                flat = space_utils.flatten(self._observation_space, observation)
                return np.asarray(flat, dtype=np.float32)
            if isinstance(observation, dict):
                raise ValueError(
                    "Identity projection for dict observations requires observation_space."
                )
            return np.asarray(observation, dtype=np.float32).reshape(-1)

        projected = np.asarray(self._projection_fn(observation), dtype=np.float32)
        if projected.ndim != 1:
            raise ValueError("Projection function must return a 1D subgoal vector.")
        return projected


def flatten_observation(
    observation_space: spaces.Space,
    observation: Union[np.ndarray, dict[str, np.ndarray]],
) -> np.ndarray:
    """Flatten an observation according to a Gymnasium space.

    Parameters
    ----------
    observation_space : spaces.Space
            Original environment observation space.
        observation : np.ndarray | dict[str, np.ndarray]
            Observation to flatten.

    Returns
    -------
    np.ndarray
            Flattened observation vector.
    """
    flat = space_utils.flatten(observation_space, observation)
    return np.asarray(flat, dtype=np.float32)


def build_worker_observation_space(
    observation_space: spaces.Space, subgoal_space: spaces.Box
) -> spaces.Box:
    """Construct the low-level worker observation space.

    Parameters
    ----------
    observation_space : spaces.Space
            Environment observation space.
    subgoal_space : spaces.Box
            Subgoal/action space used by the manager policy.

    Returns
    -------
    spaces.Box
            Concatenated worker observation space ``[flatten(obs), goal]``.
    """
    flat_obs_space = space_utils.flatten_space(observation_space)
    if not isinstance(flat_obs_space, spaces.Box):
        raise TypeError("HIRO worker currently requires flattenable Box observations.")

    low = np.concatenate(
        [flat_obs_space.low.astype(np.float32), subgoal_space.low.astype(np.float32)],
        axis=0,
    )
    high = np.concatenate(
        [flat_obs_space.high.astype(np.float32), subgoal_space.high.astype(np.float32)],
        axis=0,
    )
    return spaces.Box(low=low, high=high, dtype=np.float32)


def make_worker_observation(
    flat_observation: np.ndarray, subgoal: np.ndarray
) -> np.ndarray:
    """Create worker input by concatenating state and subgoal.

    Parameters
    ----------
    flat_observation : np.ndarray
            Flattened environment observation.
    subgoal : np.ndarray
            Current manager subgoal.

    Returns
    -------
    np.ndarray
            Worker observation vector.
    """
    return np.concatenate([flat_observation, subgoal], axis=0).astype(
        np.float32, copy=False
    )
