from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import gymnasium as gym
import numpy as np
from absl.testing import absltest, parameterized
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_hrl.option.callbacks import PrimitiveStepCountCallback
from sb3_hrl.option.policies import PrimitiveStepPPO


class _DummyProgressBar:
    def __init__(self) -> None:
        self.total: int = 0

    def update(self, delta: int) -> None:
        self.total += delta


class _DummyCallback(BaseCallback):
    def __init__(self, pbar: _DummyProgressBar | None = None) -> None:
        super().__init__()
        self.pbar = pbar
        self.callbacks: list[BaseCallback] = []

    def _on_step(self) -> bool:
        return True


class _DummyModel:
    def __init__(
        self,
        *,
        primitive_step_aware: bool,
        initial_timesteps: int = 0,
    ) -> None:
        self._primitive_step_aware = primitive_step_aware
        self.num_timesteps = initial_timesteps


class _PrimitiveInfoEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        primitive_step_sequence: Sequence[int],
        episode_len: int = 32,
    ) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        self._primitive_step_sequence = [
            int(v) for v in primitive_step_sequence
        ]
        self._episode_len = int(episode_len)
        self._step_idx = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_idx = 0
        return np.zeros((1,), dtype=np.float32), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        self._step_idx += 1
        idx = (self._step_idx - 1) % len(self._primitive_step_sequence)
        primitive_steps = self._primitive_step_sequence[idx]
        truncated = self._step_idx >= self._episode_len
        obs = np.array([self._step_idx % 2], dtype=np.float32)
        info = {"meta_option_steps": primitive_steps}
        return obs, 1.0, False, truncated, info


def _build_model(env: DummyVecEnv, n_steps: int) -> PrimitiveStepPPO:
    return PrimitiveStepPPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=2,
        n_epochs=1,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": [8]},
    )


class PrimitiveStepUtilsTest(parameterized.TestCase):
    @parameterized.named_parameters(
        (
            "standard_values",
            [{"meta_option_steps": 3}, {"meta_option_steps": 5}],
            2,
            [3, 5],
        ),
        (
            "missing_key_fallback",
            [{}, {"meta_option_steps": 2}],
            2,
            [1, 2],
        ),
        (
            "malformed_value_fallback",
            [{"meta_option_steps": "invalid"}],
            1,
            [1],
        ),
        (
            "all_zero_fallback_to_ones",
            [{"meta_option_steps": 0}, {"meta_option_steps": 0}],
            2,
            [1, 1],
        ),
        (
            "non_dict_fallback",
            [None],
            1,
            [1],
        ),
    )
    def test_extract_primitive_steps_per_env(
        self,
        infos: list[Any],
        n_envs: int,
        expected: list[int],
    ) -> None:
        extracted = PrimitiveStepPPO._extract_primitive_steps_per_env(
            infos, n_envs
        )
        np.testing.assert_array_equal(
            extracted, np.array(expected, dtype=np.int64)
        )

    def test_adjust_progress_bar_updates_nested_callbacks(self) -> None:
        pbar = _DummyProgressBar()
        inner_callback = _DummyCallback(pbar=pbar)
        parent_callback = _DummyCallback()
        parent_callback.callbacks.append(inner_callback)

        PrimitiveStepPPO._adjust_progress_bar(parent_callback, delta_steps=6)

        self.assertEqual(pbar.total, 6)

    def test_primitive_step_count_callback_skips_aware_model(self) -> None:
        callback = PrimitiveStepCountCallback()
        dummy_model = _DummyModel(
            primitive_step_aware=True, initial_timesteps=10
        )
        callback.model = cast(BaseAlgorithm, dummy_model)
        callback.locals = {"infos": [{"meta_option_steps": 5}]}

        callback._on_step()

        self.assertEqual(dummy_model.num_timesteps, 10)

    def test_primitive_step_count_callback_adjusts_standard_model(self) -> None:
        callback = PrimitiveStepCountCallback()
        dummy_model = _DummyModel(
            primitive_step_aware=False, initial_timesteps=10
        )
        callback.model = cast(BaseAlgorithm, dummy_model)
        callback.locals = {
            "infos": [{"meta_option_steps": 4}, {"meta_option_steps": 6}]
        }

        callback._on_step()

        # Added delta is (4 - 1) + (6 - 1) = 8.
        self.assertEqual(dummy_model.num_timesteps, 18)


class PrimitiveStepPPOIntegrationTest(absltest.TestCase):
    def test_learn_tracks_primitive_steps_and_macro_budget(self) -> None:
        env = DummyVecEnv(
            [
                lambda: _PrimitiveInfoEnv([2], episode_len=64),
                lambda: _PrimitiveInfoEnv([5], episode_len=64),
            ]
        )
        model = _build_model(env, n_steps=2)

        model.learn(total_timesteps=10)

        # 2 macro steps collected across 2 envs equals (2 + 5) * 2 = 14 primitive steps.
        self.assertEqual(model.num_timesteps, 14)
        self.assertEqual(model._n_updates, 1)

    def test_monitor_records_primitive_episode_length(self) -> None:
        env = DummyVecEnv(
            [lambda: Monitor(_PrimitiveInfoEnv([3], episode_len=2))]
        )
        model = _build_model(env, n_steps=2)

        model.learn(total_timesteps=6)

        self.assertIsNotNone(model.ep_info_buffer)
        assert model.ep_info_buffer is not None
        self.assertEqual(model.ep_info_buffer[-1]["l"], 6)


if __name__ == "__main__":
    absltest.main()
