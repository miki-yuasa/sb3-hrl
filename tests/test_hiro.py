"""Unit tests for the HIRO implementation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
from absl.testing import absltest, parameterized
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_hrl.hiro import HIRO
from sb3_hrl.hiro.hiro import _MacroTransitionAccumulator, HIROReplayBuffer
from sb3_hrl.hiro.policies import (
    SubgoalProjectionWrapper,
    flatten_observation,
)


class _ContinuousEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int = 4, action_dim: int = 2) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        self._step_count = 0
        shape = self.observation_space.shape
        assert shape is not None
        return np.zeros(shape, dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        self._step_count += 1
        shape = self.observation_space.shape
        assert shape is not None
        obs = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        done = self._step_count >= 15
        return obs, 1.0, done, False, {}


class _DiscreteEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int = 4, n_actions: int = 3) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        self._step_count = 0
        shape = self.observation_space.shape
        assert shape is not None
        return np.zeros(shape, dtype=np.float32), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        self._step_count += 1
        shape = self.observation_space.shape
        assert shape is not None
        obs = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        done = self._step_count >= 15
        return obs, 1.0, done, False, {}


class HIROTest(parameterized.TestCase):
    def test_macro_accumulator_preallocation(self) -> None:
        """Verify pre-allocated accumulator storage and cursor indexing."""
        acc = _MacroTransitionAccumulator(
            subgoal_freq=5, obs_dim=4, action_dim=2, goal_dim=4
        )
        self.assertEqual(acc.length, 0)
        self.assertEqual(acc.micro_obs.shape, (0, 4))

        for step in range(3):
            acc.append_micro(
                obs=np.full((4,), step, dtype=np.float32),
                next_obs=np.full((4,), step + 1, dtype=np.float32),
                action=np.full((2,), step * 0.1, dtype=np.float32),
                proj_obs=np.full((4,), step, dtype=np.float32),
                proj_next=np.full((4,), step + 1, dtype=np.float32),
            )
        self.assertEqual(acc.length, 3)
        self.assertEqual(acc.micro_obs.shape, (3, 4))
        self.assertEqual(acc.micro_actions.shape, (3, 2))
        np.testing.assert_array_equal(acc.micro_obs[0], np.zeros(4))
        np.testing.assert_array_equal(acc.micro_obs[2], np.full(4, 2.0))

        acc.reset()
        self.assertEqual(acc.length, 0)
        self.assertEqual(acc.micro_obs.shape, (0, 4))

    def test_flatten_observation_and_projection(self) -> None:
        """Verify Box fast-path and SubgoalProjectionWrapper."""
        box_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        obs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        flat = flatten_observation(box_space, obs)
        np.testing.assert_array_equal(flat, obs)

        proj = SubgoalProjectionWrapper(None, observation_space=box_space)
        self.assertTrue(proj.is_identity)
        projected = proj(obs)
        np.testing.assert_array_equal(projected, obs)

    def test_vectorized_relabel_goals_continuous(self) -> None:
        """Verify vectorized goal relabeling for continuous worker."""
        subgoal_space = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)
        obs_space = spaces.Box(low=-2.0, high=2.0, shape=(4,), dtype=np.float32)
        buffer = HIROReplayBuffer(
            buffer_size=10,
            observation_space=obs_space,
            action_space=subgoal_space,
            subgoal_freq=5,
            state_to_goal_proj_fn=lambda s: s[:2],
            worker_action_dim=2,
            discrete_worker=False,
            n_envs=1,
        )
        buffer.set_low_level_action_fn(
            lambda obs: np.zeros((obs.shape[0], 2), dtype=np.float32)
        )

        # Add dummy micro trajectories.
        buffer.micro_lengths[0, 0] = 5
        buffer.micro_lengths[1, 0] = 3
        current_actions = np.zeros((2, 2), dtype=np.float32)
        batch_inds = np.array([0, 1])
        env_indices = np.array([0, 0])

        relabeled = buffer._relabel_goals(
            batch_inds=batch_inds,
            env_indices=env_indices,
            current_actions=current_actions,
        )
        self.assertEqual(relabeled.shape, (2, 2))
        self.assertTrue(np.all(relabeled >= -1.0) and np.all(relabeled <= 1.0))

    def test_hiro_learn_and_predict_continuous_single_env(self) -> None:
        """Verify HIRO training loop and prediction on single continuous env."""
        env = _ContinuousEnv(obs_dim=4, action_dim=2)
        model = HIRO(
            "MlpPolicy",
            env,
            learning_starts=20,
            buffer_size=1000,
            batch_size=16,
            subgoal_freq=5,
            train_freq=2,
            gradient_steps=1,
            seed=42,
        )
        model.learn(total_timesteps=40)
        self.assertGreaterEqual(model.num_timesteps, 40)

        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        self.assertEqual(action.shape, (2,))

    def test_hiro_learn_and_predict_continuous_multi_env(self) -> None:
        """Verify HIRO training loop and prediction on vectorized multi-env."""

        def make_env() -> _ContinuousEnv:
            return _ContinuousEnv(obs_dim=4, action_dim=2)

        vec_env = DummyVecEnv([make_env, make_env])
        model = HIRO(
            "MlpPolicy",
            vec_env,
            learning_starts=20,
            buffer_size=1000,
            batch_size=16,
            subgoal_freq=5,
            train_freq=2,
            gradient_steps=1,
            seed=42,
        )
        assert model.worker.replay_buffer is not None
        self.assertEqual(model.worker.replay_buffer.n_envs, 2)
        model.learn(total_timesteps=40)
        self.assertGreaterEqual(model.num_timesteps, 40)

        vec_obs = vec_env.reset()
        assert isinstance(vec_obs, np.ndarray)
        actions, _ = model.predict(vec_obs, deterministic=True)
        self.assertEqual(actions.shape, (2, 2))

    def test_hiro_learn_and_predict_discrete_worker(self) -> None:
        """Verify HIRO training loop and prediction with discrete worker."""
        env = _DiscreteEnv(obs_dim=4, n_actions=3)
        model = HIRO(
            "MlpPolicy",
            env,
            learning_starts=20,
            buffer_size=1000,
            batch_size=16,
            subgoal_freq=5,
            train_freq=2,
            gradient_steps=1,
            seed=42,
        )
        model.learn(total_timesteps=40)
        self.assertGreaterEqual(model.num_timesteps, 40)

        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        self.assertIsInstance(int(action), int)

    def test_hiro_save_and_load(self) -> None:
        """Verify save and load functionality."""
        env = _ContinuousEnv(obs_dim=4, action_dim=2)
        model = HIRO(
            "MlpPolicy",
            env,
            learning_starts=10,
            buffer_size=1000,
            batch_size=16,
            subgoal_freq=5,
            seed=42,
        )
        model.learn(total_timesteps=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "hiro_test.zip"
            model.save(str(save_path))
            self.assertTrue(save_path.exists())

            loaded_model = HIRO.load(str(save_path), env=env)
            self.assertEqual(loaded_model.subgoal_freq, model.subgoal_freq)
            obs, _ = env.reset()
            action, _ = loaded_model.predict(obs, deterministic=True)
            self.assertEqual(action.shape, (2,))


if __name__ == "__main__":
    absltest.main()
