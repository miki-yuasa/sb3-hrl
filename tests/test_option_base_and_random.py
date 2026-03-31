from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from gymnasium import spaces

from sb3_hrl.option import BaseOption, RandomOption


class _ValidOption(BaseOption):
    def initiation_set(self, obs: Any) -> bool:
        del obs
        return True

    def termination_condition(self, obs: Any) -> bool:
        del obs
        return False

    def intrinsic_reward(
        self,
        obs: Any,
        action: Any,
        next_obs: Any,
        external_reward: float,
        done: bool,
    ) -> float:
        del obs, action, next_obs, done
        return float(external_reward)


class _DummyPolicy:
    def __init__(self, action: Any) -> None:
        self.action = action

    def predict(self, observation: Any, deterministic: bool = True):
        del observation, deterministic
        return self.action, None


def test_base_option_predict_requires_attached_policy() -> None:
    option = _ValidOption()
    with pytest.raises(RuntimeError):
        option.predict(np.array([0.0], dtype=np.float32))


def test_base_option_predict_forwards_to_attached_policy() -> None:
    option = _ValidOption(policy=_DummyPolicy(action=2))
    action = option.predict(np.array([1.0], dtype=np.float32))
    assert action == 2


def test_random_option_discrete_samples_valid_actions() -> None:
    option = RandomOption(action_space=spaces.Discrete(3), termination_steps=2)
    option.reset_execution_state()

    a0 = option.predict(obs=None)
    a1 = option.predict(obs=None)

    assert a0 in {0, 1, 2}
    assert a1 in {0, 1, 2}
    assert option.termination_condition(obs=None)


def test_random_option_box_action_shape_is_preserved() -> None:
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    option = RandomOption(action_space=action_space)
    action = option.predict(obs=None)
    assert action.shape == (2,)
    assert action_space.contains(action)
