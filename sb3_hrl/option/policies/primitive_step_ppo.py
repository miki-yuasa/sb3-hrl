"""Primitive-step aware PPO for meta-controller training.

This module provides a PPO variant where rollout collection uses macro
meta-controller timesteps while training progress (``num_timesteps`` and
``total_timesteps`` in ``learn``) is tracked in primitive low-level steps.
"""

from __future__ import annotations

from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo.ppo import PPO

SelfPrimitiveStepPPO = TypeVar("SelfPrimitiveStepPPO", bound="PrimitiveStepPPO")


class PrimitiveStepPPO(PPO):
    """PPO variant that treats timesteps as primitive low-level steps.

    In this class:
    - ``n_steps`` is the macro meta-controller-step budget collected per update.
    - ``total_timesteps`` in :meth:`learn` is interpreted as primitive steps.

    Primitive step counts are read from ``info["meta_option_steps"]`` for each
    environment. If missing or invalid, the implementation falls back to one
    primitive step per environment for robustness.
    """

    _primitive_step_aware: bool = True

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    @staticmethod
    def _extract_primitive_steps(infos: list[dict[str, Any]]) -> int:
        primitive_steps = 0
        for info in infos:
            if not isinstance(info, dict):
                primitive_steps += 1
                continue

            raw_steps = info.get("meta_option_steps", 1)
            try:
                step_count = int(raw_steps)
            except (TypeError, ValueError):
                step_count = 1
            primitive_steps += max(0, step_count)

        if primitive_steps <= 0:
            raise ValueError(
                f"Invalid primitive step count extracted from infos: {primitive_steps}. "
                "Check that 'meta_option_steps' is correctly set in the environment's info dict."
            )

        return primitive_steps

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        primitive_steps = 0
        macro_steps = 0
        new_obs = self._last_obs
        dones = np.zeros(env.num_envs, dtype=bool)
        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while macro_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and macro_steps % self.sde_sample_freq == 0
            ):
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                actions, values, log_probs = self.policy(obs_tensor)
            actions_np = actions.cpu().numpy()

            clipped_actions = actions_np
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(
                        actions_np, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            primitive_increment = self._extract_primitive_steps(infos)
            self.num_timesteps += primitive_increment
            primitive_steps += primitive_increment

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            macro_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions_np = actions_np.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions_np,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()

        self.logger.record("rollout/primitive_steps_collected", primitive_steps)
        self.logger.record("rollout/macro_steps_collected", macro_steps * env.num_envs)

        return True


__all__ = ["PrimitiveStepPPO"]
