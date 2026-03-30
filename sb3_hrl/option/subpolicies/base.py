from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import torch
from gymnasium import Env
from gymnasium.core import ActType, ObsType

PolicyType = TypeVar("PolicyType")
PolicyArgsType = TypeVar("PolicyArgsType")


class SubPolicyBuffer(Generic[PolicyArgsType]):
    def __init__(self, policy_args: PolicyArgsType) -> None:
        self.policy_args = policy_args

    def at_reset(self) -> None:
        pass


class SubPolicy(Generic[PolicyType, ObsType, ActType], ABC):
    buffer_class: type[SubPolicyBuffer] = SubPolicyBuffer

    def __init__(
        self,
        env: Env[ObsType, ActType],
        tl_spec: str,
        max_policy_steps: int,
        policy_args: dict[str, Any],
        buffer: SubPolicyBuffer[dict[str, Any]] | None = None,
        verbose: bool = False,
    ) -> None:
        self.env: Env[ObsType, ActType] = env
        self.tl_spec: str = tl_spec
        self.max_policy_steps: int = max_policy_steps
        self.policy_step: int = 0
        self.buffer: SubPolicyBuffer[dict[str, Any]] | None = buffer
        self.policy: PolicyType = self.define_policy(policy_args)
        self.verbose: bool = verbose

    def predict(
        self,
        current_env: Env[ObsType, ActType],
        obs: ObsType,
        info: dict[str, Any],
    ) -> tuple[ActType, bool, bool]:
        """
        Predict the action using the low-level policy.

        Parameters
        ----------
        current_env: Env[ObsType, ActType]
            The current environment in which the policy is acting.
        obs: ObsType
            The observation from the high-level environment.
        info: dict[str, Any]
            Additional information from the high-level environment, such as the current state of the automaton.

        Returns
        -------
        action: ActType
            The action predicted by the low-level policy.
        terminated: bool
            Whether the low-level policy has terminated.
        truncated: bool
            Whether the low-level policy has been truncated.

        """
        self.update_env(current_env, obs, info)
        terminated: bool = False

        action = self.act(obs, info, current_env)
        self.policy_step += 1
        truncated: bool = self.policy_step >= self.max_policy_steps

        return action, terminated, truncated

    def update_env(
        self,
        current_env: Env[ObsType, ActType],
        obs: ObsType,
        info: dict[str, Any],
    ) -> None:
        """Update the environment and observation."""
        pass

    def delete_policy(self) -> None:
        del self.policy
        torch.cuda.empty_cache()

    @abstractmethod
    def define_policy(self, policy_args: dict[str, Any]) -> PolicyType:
        """
        Define the subpolicy. This method should be implemented by subclasses to initialize the specific subpolicy.

        Parameters
        ----------
        policy_args: dict[str, Any]
            A dictionary of arguments required to initialize the subpolicy.
        """
        ...

    @abstractmethod
    def act(
        self,
        obs: ObsType,
        info: dict[str, Any] | None = None,
        current_env: Env[ObsType, ActType] | None = None,
    ) -> ActType: ...
