from __future__ import annotations

import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
from absl.testing import absltest, parameterized
from gymnasium import spaces

from sb3_hrl.allo import (
    ALLO,
    ALLOCheckpointCallback,
    LossEvalCallback,
)


class _DummyBoxEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int = 4) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        shape = self.observation_space.shape
        assert shape is not None
        return np.zeros(shape, dtype=np.float32), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        shape = self.observation_space.shape
        assert shape is not None
        obs = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        return obs, 0.0, False, False, {}


class _DummyDictEnv(gym.Env[dict[str, np.ndarray], int]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "vel": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(2)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        super().reset(seed=seed)
        obs = {
            "pos": np.zeros(2, dtype=np.float32),
            "vel": np.zeros(3, dtype=np.float32),
        }
        return obs, {}

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        del action
        obs = {
            "pos": np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32),
            "vel": np.random.uniform(-1.0, 1.0, size=(3,)).astype(np.float32),
        }
        return obs, 0.0, False, False, {}


class ALLOTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        super().tearDown()

    def test_flatten_dict_observations_vectorized(self) -> None:
        env = _DummyDictEnv()
        allo = ALLO(
            env=env,
            representation_dim=2,
            buffer_size=100,
            batch_size=16,
            device="cpu",
        )
        batched_dict = {
            "pos": np.ones((allo.n_envs, 2), dtype=np.float32),
            "vel": np.full((allo.n_envs, 3), 2.0, dtype=np.float32),
        }
        flat = allo._flatten_vec_observations(batched_dict)
        self.assertEqual(flat.shape, (allo.n_envs, 5))
        np.testing.assert_allclose(flat[:, :2], 1.0)
        np.testing.assert_allclose(flat[:, 2:], 2.0)

    def test_train_step_batched_forward(self) -> None:
        env = _DummyBoxEnv(obs_dim=6)
        allo = ALLO(
            env=env,
            representation_dim=4,
            buffer_size=200,
            batch_size=32,
            device="cpu",
        )

        dummy_obs = np.random.randn(allo.buffer_size, allo.n_envs, 6).astype(np.float32)
        allo.replay_buffer.observations[:] = dummy_obs
        allo.replay_buffer.next_observations[:] = dummy_obs
        allo.replay_buffer.full = True

        stats_with_metrics = allo.train_step(record_metrics=True)
        self.assertIn("loss/total", stats_with_metrics)
        self.assertIn("loss/graph", stats_with_metrics)
        self.assertIn("loss/dual", stats_with_metrics)
        self.assertIn("loss/barrier", stats_with_metrics)
        self.assertFalse(np.isnan(stats_with_metrics["loss/total"]))

        stats_no_metrics = allo.train_step(record_metrics=False)
        self.assertEmpty(stats_no_metrics)

    def test_loss_eval_callback_saves_best_model(self) -> None:
        env = _DummyBoxEnv(obs_dim=4)
        allo = ALLO(
            env=env,
            representation_dim=2,
            buffer_size=100,
            batch_size=16,
            device="cpu",
        )

        dummy_obs = np.random.randn(allo.buffer_size, allo.n_envs, 4).astype(np.float32)
        allo.replay_buffer.observations[:] = dummy_obs
        allo.replay_buffer.next_observations[:] = dummy_obs
        allo.replay_buffer.full = True

        best_save_dir = self.output_path / "best_model_dir"
        eval_log_dir = self.output_path / "eval_log"
        eval_cb = LossEvalCallback(
            best_model_save_path=best_save_dir,
            log_path=eval_log_dir,
            eval_freq=5,
            metric="loss/total",
            verbose=0,
        )

        allo.learn(total_timesteps=15, callback=eval_cb, log_interval=0)

        saved_best_zip = best_save_dir / "best_model.zip"
        self.assertTrue(saved_best_zip.exists())
        self.assertTrue((eval_log_dir / "evaluations.npz").exists())

    def test_allo_checkpoint_callback_saves_and_training_end(self) -> None:
        env = _DummyBoxEnv(obs_dim=4)
        allo = ALLO(
            env=env,
            representation_dim=2,
            buffer_size=100,
            batch_size=16,
            device="cpu",
        )

        dummy_obs = np.random.randn(allo.buffer_size, allo.n_envs, 4).astype(np.float32)
        allo.replay_buffer.observations[:] = dummy_obs
        allo.replay_buffer.next_observations[:] = dummy_obs
        allo.replay_buffer.full = True

        ckpt_dir = self.output_path / "ckpts"
        ckpt_cb = ALLOCheckpointCallback(
            save_freq=10,
            save_path=ckpt_dir,
            name_prefix="test_ckpt",
            save_on_training_end=True,
            verbose=0,
        )

        allo.learn(total_timesteps=15, callback=ckpt_cb, log_interval=0)

        # save_freq=10 produces test_ckpt_10_steps.zip
        self.assertTrue((ckpt_dir / "test_ckpt_10_steps.zip").exists())
        # save_on_training_end produces test_ckpt_15_steps.zip
        self.assertTrue((ckpt_dir / "test_ckpt_15_steps.zip").exists())


if __name__ == "__main__":
    absltest.main()
