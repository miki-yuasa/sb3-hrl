"""Callback for adapting SB3's timestep counting to primitive steps when using MetaControllerEnvWrapper."""

from stable_baselines3.common.callbacks import BaseCallback


class PrimitiveStepCountCallback(BaseCallback):
    """Callback that adjusts PPO's num_timesteps to count primitive steps instead of macro steps.

    When using MetaControllerEnvWrapper, each call to env.step() executes one option (multiple
    primitive actions). By default, SB3 counts this as 1 timestep. This callback reads the
    'meta_option_steps' from the info dict and adjusts num_timesteps to reflect primitive step count.

    This makes training logs and learning rate schedules work with primitive step counts,
    providing clearer alignment with the low-level policy interactions.

    Example
    -------
    >>> from sb3_hrl.option.callbacks import PrimitiveStepCountCallback
    >>> from sb3_hrl.option.wrappers import MetaControllerEnvWrapper, PrimitiveStepTimeLimit
    >>> import gymnasium as gym
    >>> from stable_baselines3 import PPO
    >>>
    >>> env = gym.make("CartPole-v1")
    >>> env = MetaControllerEnvWrapper(env, options=[...])
    >>> env = PrimitiveStepTimeLimit(env, max_episode_steps=500)  # Limit to 500 primitive steps
    >>>
    >>> callback = PrimitiveStepCountCallback()
    >>> model = PPO("MlpPolicy", env, ...)
    >>> model.learn(total_timesteps=100000, callback=callback)

    Notes
    -----
    When used with PrimitiveStepTimeLimit, the max_episode_steps in the TimeLimit wrapper
    should match the desired primitive step limit, not macro steps.
    """

    def _on_step(self) -> bool:
        """Read meta_option_steps from info and adjust num_timesteps accordingly.

        Returns
        -------
        bool
            Always returns True to continue training.
        """
        # Get the info dicts from the step (one per parallel environment)
        infos = self.locals.get("infos", [])

        if not infos:
            return True

        # For each environment, adjust timesteps by the delta between macro (1) and primitive steps
        for info in infos:
            if isinstance(info, dict):
                # Get number of primitive steps taken in this macro step
                # Default to 1 if not present (backward compatibility)
                meta_option_steps = info.get("meta_option_steps", 1)

                # SB3 already incremented num_timesteps by 1 for the macro step
                # We need to add (meta_option_steps - 1) to make it primitive-aware
                delta = int(meta_option_steps) - 1
                if delta > 0:
                    self.model.num_timesteps += delta

        return True
