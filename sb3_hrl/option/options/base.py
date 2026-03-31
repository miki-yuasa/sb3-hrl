"""Core abstractions for Sutton-Precup-Singh style options."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from sb3_hrl.typing import SupportsPredict


class BaseOption(ABC):
    """Abstract option interface.

    An option :math:`\\omega` consists of an initiation set, an intra-option
    policy, and a termination condition.

    Parameters
    ----------
    policy : SupportsPredict | None, default=None
        Attached SB3 model (or model-like object) used to generate primitive
        actions through :meth:`predict`.
    """

    def __init__(self, policy: Optional[SupportsPredict] = None) -> None:
        self._policy: Optional[SupportsPredict] = policy

    @property
    def policy(self) -> Optional[SupportsPredict]:
        """Attached policy model used for primitive action selection."""
        return self._policy

    @policy.setter
    def policy(self, model: Optional[SupportsPredict]) -> None:
        """Attach or detach a trained policy model."""
        self._policy = model

    @abstractmethod
    def initiation_set(self, obs: Any) -> bool:
        """Check whether option is available in the current state.

        Parameters
        ----------
        obs : Any
            Current observation.

        Returns
        -------
        bool
            ``True`` if the option can be initiated.
        """

    @abstractmethod
    def termination_condition(self, obs: Any) -> bool:
        """Check whether option execution should terminate.

        Parameters
        ----------
        obs : Any
            Current observation.

        Returns
        -------
        bool
            ``True`` when the option should stop.
        """

    @abstractmethod
    def intrinsic_reward(
        self,
        obs: Any,
        action: Any,
        next_obs: Any,
        external_reward: float,
        done: bool,
    ) -> float:
        """Compute intrinsic reward used for option policy training.

        Parameters
        ----------
        obs : Any
            Current observation.
        action : Any
            Primitive action taken by the option policy.
        next_obs : Any
            Next observation.
        external_reward : float
            Reward produced by the wrapped base environment.
        done : bool
            ``True`` when the base transition ends the episode.

        Returns
        -------
        float
            Intrinsic reward value.
        """

    def reset_execution_state(self) -> None:
        """Reset internal option state before a new option invocation.

        Notes
        -----
        Stateless options can keep the default implementation.
        """

    def predict(self, obs: Any, deterministic: bool = True) -> Any:
        """Query the attached policy for a primitive action.

        Parameters
        ----------
        obs : Any
            Observation consumed by the attached policy.
        deterministic : bool, default=True
            Forwarded to SB3 ``predict``.

        Returns
        -------
        Any
            Primitive action sampled by the option policy.

        Raises
        ------
        RuntimeError
            If no policy has been attached.
        """
        if self._policy is None:
            raise RuntimeError(
                "No policy is attached to this option. Set option.policy before predict()."
            )

        prediction = self._policy.predict(obs, deterministic=deterministic)
        if isinstance(prediction, tuple):
            return prediction[0]
        return prediction
