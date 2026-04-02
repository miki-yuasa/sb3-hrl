"""Core abstractions for Sutton-Precup-Singh style options."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional

from gymnasium.core import ActType, ObsType

from sb3_hrl.typing import SupportsPredict


class BaseIntrinsicReward(ABC, Generic[ObsType, ActType]):
    """Abstract intrinsic reward interface for subpolicy training wrappers.

    Any reward-logic object used by :class:`SubpolicyTrainingWrapper` should
    implement :meth:`intrinsic_reward` and can optionally define
    :meth:`termination_condition` and :meth:`reset_execution_state`.
    """

    @abstractmethod
    def intrinsic_reward(
        self,
        obs: ObsType,
        action: ActType,
        next_obs: ObsType,
        external_reward: float,
        done: bool,
    ) -> float:
        """Compute intrinsic reward used for option policy training."""

    def reset_execution_state(self) -> None:
        """Reset internal state at episode/option boundaries."""
        pass


class BaseOption(ABC, Generic[ObsType, ActType]):
    """Abstract option interface.

    An option :math:`\\omega` consists of an initiation set, an intra-option
    policy, and a termination condition.

    Parameters
    ----------
    policy : SupportsPredict | None, default=None
        Attached SB3 model (or model-like object) used to generate primitive
        actions through :meth:`predict`.
    """

    def __init__(
        self, policy: Optional[SupportsPredict[ObsType, ActType]] = None
    ) -> None:
        self._policy: Optional[SupportsPredict[ObsType, ActType]] = policy

    @property
    def policy(self) -> Optional[SupportsPredict[ObsType, ActType]]:
        """Attached policy model used for primitive action selection."""
        return self._policy

    @policy.setter
    def policy(self, model: Optional[SupportsPredict[ObsType, ActType]]) -> None:
        """Attach or detach a trained policy model."""
        self._policy = model

    def initiation_set(self, obs: ObsType) -> bool:
        """Check whether option is available in the current state.

        Parameters
        ----------
        obs : ObsType
            Current observation.

        Returns
        -------
        bool
            ``True`` if the option can be initiated.
        """
        return True

    def termination_condition(self, obs: ObsType) -> bool:
        """Check whether option execution should terminate.

        Parameters
        ----------
        obs : ObsType
            Current observation.

        Returns
        -------        bool
            ``True`` when the option should stop.
        """
        return False

    def reset_execution_state(self) -> None:
        """Reset internal option state before a new option invocation.

        Notes
        -----
        Stateless options can keep the default implementation.
        """

        pass

    def predict(self, obs: ObsType, deterministic: bool = True) -> ActType:
        """Query the attached policy for a primitive action.

        Parameters
        ----------
        obs : ObsType
            Observation consumed by the attached policy.
        deterministic : bool, default=True
            Forwarded to SB3 ``predict``.

        Returns
        -------
        ActType
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
        return prediction
