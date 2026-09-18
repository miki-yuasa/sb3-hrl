from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import gymnasium as gym
import numpy as np
from absl.testing import absltest
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from typing_extensions import override

from sb3_hrl.option.options import BaseOption
from sb3_hrl.option.vis import record_option_replay
from sb3_hrl.option.wrappers import MetaControllerEnvWrapper, OptionEnvWrapper


class _MockRenderEnv(gym.Env[np.ndarray, int]):
    """A mock gymnasium environment with an rgb_array render mode."""

    metadata = {"render_modes": ["rgb_array"]}  # noqa: RUF012

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.step_count = 0
        self.render_mode = "rgb_array"

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0
        return np.zeros(2, dtype=np.float32), {}

    @override
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        obs = np.full(2, float(self.step_count), dtype=np.float32)
        done = self.step_count >= 10
        return obs, 1.0, done, False, {}

    @override
    def render(self) -> np.ndarray:
        # Return a deterministic 2x2 dummy RGB array based on step_count.
        return np.full((2, 2, 3), fill_value=self.step_count, dtype=np.uint8)


class _FixedDurationOption(BaseOption):
    """An option that terminates after a fixed number of primitive steps."""

    def __init__(self, duration: int) -> None:
        super().__init__()
        self.duration = duration
        self._current_step = 0

    @override
    def initiation_set(self, obs: Any) -> bool:
        return True

    @override
    def termination_condition(self, obs: Any) -> bool:
        return self._current_step >= self.duration

    @override
    def predict(self, obs: Any, deterministic: bool = True) -> int:
        del deterministic
        self._current_step += 1
        return 0

    @override
    def reset_execution_state(self) -> None:
        self._current_step = 0


class OptionVisTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.base_env = _MockRenderEnv()

    def test_option_env_wrapper_default_no_recording(self) -> None:
        wrapped = OptionEnvWrapper(self.base_env)
        wrapped.reset()
        self.assertFalse(wrapped.record_render_frames)
        frame = wrapped.record_primitive_frame()
        self.assertIsNone(frame)
        self.assertEmpty(wrapped.pop_render_frames())

    def test_option_env_wrapper_records_and_pops_frames(self) -> None:
        wrapped = OptionEnvWrapper(self.base_env)
        wrapped.reset()
        wrapped.set_record_render_frames(True)
        self.assertTrue(wrapped.record_render_frames)

        frame1 = wrapped.record_primitive_frame()
        frame2 = wrapped.record_primitive_frame()
        self.assertIsNotNone(frame1)
        self.assertIsNotNone(frame2)

        popped = wrapped.pop_render_frames()
        self.assertLen(popped, 2)
        self.assertEmpty(wrapped.pop_render_frames())

    def test_meta_controller_wrapper_populates_render_frames(self) -> None:
        option = _FixedDurationOption(duration=4)
        wrapped = MetaControllerEnvWrapper(
            env=self.base_env,
            options=[option],
            include_random_option=False,
        )
        wrapped.reset()
        wrapped.set_record_render_frames(True)

        _, _, _, _, info = wrapped.step(0)
        self.assertIn("render_frames", info)
        self.assertLen(info["render_frames"], 4)
        # Verify step_count values inside the rendered dummy frames.
        frame_values = [f[0, 0, 0] for f in info["render_frames"]]
        self.assertEqual(frame_values, [1, 2, 3, 4])

    def test_record_option_replay_creates_animation(self) -> None:
        option = _FixedDurationOption(duration=3)
        wrapped = MetaControllerEnvWrapper(
            env=self.base_env,
            options=[option],
            include_random_option=False,
        )
        mock_model = mock.create_autospec(BaseAlgorithm, instance=True, spec_set=True)
        mock_model.predict.return_value = (0, None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "sub_dir" / "replay.gif"
            record_option_replay(
                demo_env=wrapped,
                model=mock_model,
                animation_save_path=str(save_path),
                verbose=False,
                close_env=False,
                fps=10,
            )
            self.assertTrue(save_path.exists())
            self.assertGreater(save_path.stat().st_size, 0)
            # Wrapper should have been reset to record_render_frames=False.
            self.assertFalse(wrapped.record_render_frames)


if __name__ == "__main__":
    absltest.main()
