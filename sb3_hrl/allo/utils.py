from stable_baselines3.common.base_class import BaseCallback
from tqdm.rich import tqdm


class ALLOProgressBarCallback(BaseCallback):
    """
    Display a progress bar when training SB3 agent
    using tqdm and rich packages.
    """

    pbar: tqdm

    def __init__(self) -> None:
        super().__init__()
        if tqdm is None:
            raise ImportError(
                "You must install tqdm and rich in order to use the progress bar callback. "
                "It is included if you install stable-baselines with the extra packages: "
                "`pip install stable-baselines3[extra]`"
            )

    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(
            total=self.locals["total_timesteps"] - self.model.num_timesteps
        )

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        # Flush and close progress bar
        self.pbar.refresh()
        self.pbar.close()
