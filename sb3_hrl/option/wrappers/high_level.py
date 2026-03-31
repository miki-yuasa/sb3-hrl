import copy
import json
from collections.abc import Callable
from typing import Any, Generic, Literal, SupportsFloat, TypedDict

import numpy as np
from gymnasium import Env, Wrapper, spaces
from gymnasium.core import ActType, ObsType
from gymnasium.utils import RecordConstructorArgs
from numpy.typing import NDArray
from sympy import Predicate

from sb3_hrl.option.subpolicies import (
    BaseSubPolicy,
    BaseSubPolicyBuffer,
    PolicyArgsType,
    PolicyType,
    SB3SubPolicy,
)
from sb3_hrl.utils.io import get_class


class OptionHighLevelWrapper(
    Wrapper[ObsType, NDArray[np.integer], ObsType, ActType],
    RecordConstructorArgs,
    Generic[PolicyType, ObsType, ActType],
):
    def __init__(
        self,
        env: Env[ObsType, ActType],
        low_level_policy_class: type[BaseSubPolicy[PolicyType, ObsType, ActType]],
        low_level_policy_args: dict[str, Any],
        max_low_level_policy_steps: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the OptionHighLevelWrapper.

        Parameters
        ----------
        env : Env[ObsType, ActType]
            The environment to wrap.
        low_level_policy : Callable[[ObsType, int, TLObservationReward[ObsType, ActType], PolicyArgsType], ActType]
            The low-level policy function that takes the observation, automaton state,
            low-level environment, and policy arguments, and returns an action.
        low_level_policy_args : PolicyArgsType, optional
            Arguments for the low-level policy function (default is an empty dictionary).
        max_low_level_policy_steps : int = 10
            The maximum number of steps the low-level policy can take before resetting.
        verbose : bool = False
            If True, prints verbose output during execution.
        """
        RecordConstructorArgs.__init__(
            self,
            low_level_policy_class=low_level_policy_class,
            # low_level_policy_args=low_level_policy_args,
            max_low_level_policy_steps=max_low_level_policy_steps,
            verbose=verbose,
        )
        Wrapper.__init__(self, env)

        self.max_low_level_policy_steps: int = max_low_level_policy_steps
        self.verbose: bool = verbose

        self.action_space = self.spec_rep.action_space
        self.observation_space = self._build_observation_space_with_step_count(
            self.env.observation_space
        )

        self.low_level_policy_class = low_level_policy_class
        self.low_level_policy_args = low_level_policy_args

        self.low_level_policy_buffer: BaseSubPolicyBuffer = (
            low_level_policy_class.buffer_class(low_level_policy_args)
        )

    def _add_step_count_to_obs(self, obs: ObsType, policy_step: int) -> ObsType:
        """Add normalized low-level policy step count to dict observations."""
        if not isinstance(obs, dict):
            return obs

        denom = max(1, self.max_low_level_policy_steps)
        step_ratio = np.float32(policy_step / denom)

        obs_with_step = dict(obs)
        obs_with_step["step_count"] = np.array([step_ratio], dtype=np.float32)
        return obs_with_step  # type: ignore[return-value]

    def _build_observation_space_with_step_count(
        self,
        observation_space: spaces.Space,
    ) -> spaces.Space:
        """Extend Dict observation spaces with a normalized step_count field."""
        step_count_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )
        if not isinstance(observation_space, spaces.Dict):
            return spaces.Dict(
                observation=observation_space,
                step_count=step_count_space,
            )
        else:
            spaces_dict = dict(observation_space.spaces)
            spaces_dict["step_count"] = step_count_space
            return spaces.Dict(spaces_dict)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Saves the last observation and returns the original environment's reset observation.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs: ObsType = obs
        self.last_info: dict[str, Any] = info
        self.last_policy_args: dict[str, Any] | None = None
        obs = self._add_step_count_to_obs(obs, policy_step=0)

        # self.low_level_policy_step: int = 0
        # self.current_tl_env: TLObservationReward[ObsType, ActType] | None = None
        self.low_level_policy: BaseSubPolicy[PolicyType, ObsType, ActType] | None = None
        self.low_level_policy_buffer.at_reset()

        return obs, info

    def step(
        self, action: NDArray[np.integer]
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Takes a high-level action, converts it to a temporal logic specification,
        and uses the low-level policy to execute it in the low-level environment.
        """

        if self.verbose:
            print(
                f"High-level action: {action},\n"
                f"Subpolicy policy step: {self.low_level_policy.policy_step if self.low_level_policy else 0}."
            )

        if not self.low_level_policy:
            # Convert the high-level action to a temporal logic specification

            policy_args_update: dict[str, Any] = self.spec_rep.action2policy_args(
                action
            )
            self.last_policy_args = policy_args_update
            policy_args = copy.deepcopy(self.low_level_policy_args)
            policy_args.update(policy_args_update)
            self.low_level_policy = self.low_level_policy_class(
                env=self.env,
                max_policy_steps=self.max_low_level_policy_steps,
                policy_args=policy_args,
                buffer=self.low_level_policy_buffer,
            )

            self.low_level_policy.update_env(
                self.env,
                self.last_obs,
                self.last_info,
            )

        else:
            pass

        added_info: dict[str, str | None] = {
            "current_tl_spec": (
                self.low_level_policy.tl_spec if self.low_level_policy else None
            )
        }

        low_level_action: ActType

        low_level_action, ll_terminated, ll_truncated = self.low_level_policy.predict(
            self.env, self.last_obs, self.last_info
        )
        if self.verbose:
            print(
                f"- Subpolicy action: {low_level_action},\n"
                f"-- Subpolicy step: {self.low_level_policy.policy_step},\n"
                f"-- Subpolicy terminated: {ll_terminated}, "
            )

        if ll_terminated or ll_truncated:
            self.low_level_policy.delete_policy()
            self.low_level_policy = None
        else:
            pass

        obs, reward, terminated, truncated, info = self.env.step(low_level_action)

        current_policy_step = (
            self.low_level_policy.policy_step if self.low_level_policy else 0
        )

        # Update the info for
        # info.update({"current_tl_spec": (current_tl_spec)})
        info.update(added_info)
        info.update(self.last_policy_args if self.last_policy_args else {})
        self.last_obs = obs
        self.last_info = info

        obs = self._add_step_count_to_obs(obs, policy_step=current_policy_step)

        # self.low_level_policy_step += 1

        return obs, reward, terminated, truncated, info
