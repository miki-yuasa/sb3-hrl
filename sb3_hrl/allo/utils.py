"""Utility callbacks and helpers for ALLO training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from tqdm.rich import tqdm


class ALLOProgressBarCallback(BaseCallback):
    """Progress bar callback for ALLO offline representation training.

    Displays a rich tqdm progress bar tracking offline optimization epochs.
    """

    pbar: tqdm

    def __init__(self) -> None:
        super().__init__()
        if tqdm is None:
            raise ImportError(
                "You must install tqdm and rich to use ALLOProgressBarCallback: "
                "`pip install stable-baselines3[extra]`"
            )

    def _on_training_start(self) -> None:
        total_steps = int(self.locals["total_timesteps"])
        current_steps = int(self.model.num_timesteps)
        self.pbar = tqdm(total=max(total_steps - current_steps, 0))

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        self.pbar.refresh()
        self.pbar.close()


class LossEvalCallback(BaseCallback):
    """Evaluation callback that tracks representation loss and saves the best model.

    Evaluates the offline representation loss rather than external environment
    rewards, smoothing metrics across a recent step window to mitigate batch
    variance before saving `best_model.zip`.

    Parameters
    ----------
    eval_env : Any, optional
        Unused environment handle, accepted for API compatibility with SB3
        `EvalCallback`.
    best_model_save_path : str | Path | None, default=None
        Directory where `best_model.zip` will be saved on improvement.
    log_path : str | Path | None, default=None
        Directory where `evaluations.npz` containing evaluation history will be
        persisted.
    eval_freq : int, default=10_000
        Evaluation frequency in gradient steps.
    metric : str, default="loss/total"
        Metric key to minimize (e.g. `"loss/total"`, `"loss/graph"`, or
        `"diagnostics/mean_constraint"`).
    verbose : int, default=1
        Verbosity level (0: silent, 1: console notices on new best loss).
    """

    def __init__(
        self,
        eval_env: Any = None,
        best_model_save_path: str | Path | None = None,
        log_path: str | Path | None = None,
        eval_freq: int = 10_000,
        metric: str = "loss/total",
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(verbose=verbose)
        del eval_env, kwargs
        self.best_model_save_path = (
            Path(best_model_save_path) if best_model_save_path is not None else None
        )
        self.log_path = Path(log_path) if log_path is not None else None
        self.eval_freq = int(eval_freq)
        self.metric = str(metric)
        self.best_loss = float("inf")
        self.last_eval_timesteps = 0
        self.eval_idx = 0
        self._recent_losses: list[float] = []
        self._evaluations_timesteps: list[int] = []
        self._evaluations_losses: list[float] = []

    def _extract_metric(self) -> float | None:
        """Extract current loss metric from locals or model logger.

        Returns:
            Extracted floating-point loss scalar, or None if unavailable.
        """
        stats = self.locals.get("stats")
        if isinstance(stats, dict) and self.metric in stats:
            try:
                return float(stats[self.metric])
            except (TypeError, ValueError):
                pass

        if hasattr(self.model, "logger") and self.model.logger is not None:
            name_to_val = getattr(self.model.logger, "name_to_value", {})
            if self.metric in name_to_val:
                try:
                    return float(name_to_val[self.metric])
                except (TypeError, ValueError):
                    pass

        return None

    def _should_eval(self) -> bool:
        """Determine whether evaluation should execute on this step.

        Returns:
            True if evaluation condition is satisfied.
        """
        if self.eval_freq <= 0:
            return False
        timestep_triggered = (
            self.num_timesteps > 0
            and (self.num_timesteps - self.last_eval_timesteps) >= self.eval_freq
        )
        call_triggered = self.n_calls > 0 and self.n_calls % self.eval_freq == 0
        return timestep_triggered or call_triggered

    def _evaluate_and_save(self, loss_val: float) -> None:
        """Record evaluation loss, update best score, and save checkpoint.

        Args:
            loss_val: Mean loss over the evaluation window.
        """
        self.last_eval_timesteps = self.num_timesteps
        self.eval_idx += 1
        self._evaluations_timesteps.append(self.num_timesteps)
        self._evaluations_losses.append(loss_val)

        if hasattr(self.model, "logger") and self.model.logger is not None:
            self.logger.record("eval/loss", loss_val)
            self.logger.record("eval/best_loss", min(self.best_loss, loss_val))

        if loss_val < self.best_loss:
            if self.verbose >= 1:
                print(
                    f"[Eval Step {self.num_timesteps}] New best {self.metric}: "
                    f"{loss_val:.6f} (previous: {self.best_loss:.6f})"
                )
            self.best_loss = loss_val

            if self.best_model_save_path is not None:
                self.best_model_save_path.mkdir(parents=True, exist_ok=True)
                save_dest = self.best_model_save_path / "best_model"
                self.model.save(str(save_dest))
                if self.verbose >= 1:
                    print(f"Saved new best model to {save_dest}.zip")

        if self.log_path is not None:
            self.log_path.mkdir(parents=True, exist_ok=True)
            np.savez(
                self.log_path / "evaluations.npz",
                timesteps=np.asarray(self._evaluations_timesteps, dtype=np.int64),
                losses=np.asarray(self._evaluations_losses, dtype=np.float32),
            )

    def _on_step(self) -> bool:
        loss_val = self._extract_metric()
        if loss_val is not None and not np.isnan(loss_val):
            self._recent_losses.append(loss_val)

        if self._should_eval():
            if self._recent_losses:
                current_loss = float(np.mean(self._recent_losses))
                self._recent_losses.clear()
            else:
                current_loss = loss_val if loss_val is not None else float("inf")

            self._evaluate_and_save(current_loss)

        return True

    def _on_training_end(self) -> None:
        if self._recent_losses:
            final_loss = float(np.mean(self._recent_losses))
            self._recent_losses.clear()
            self._evaluate_and_save(final_loss)


class ALLOCheckpointCallback(CheckpointCallback):
    """Checkpoint callback supporting interval saves and training-end preservation.

    Parameters
    ----------
    save_freq : int
        Frequency of checkpointing in timesteps.
    save_path : str | Path
        Directory where model checkpoints will be stored.
    name_prefix : str, default="ckpt"
        Prefix for saved checkpoint files.
    save_replay_buffer : bool, default=False
        Whether to serialize the replay buffer.
    save_vecnormalize : bool, default=False
        Whether to serialize VecNormalize statistics.
    save_on_training_end : bool, default=True
        Whether to guarantee a checkpoint save upon completion of training.
    verbose : int, default=0
        Verbosity level.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str | Path,
        name_prefix: str = "ckpt",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        save_on_training_end: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            save_freq=save_freq,
            save_path=str(save_path),
            name_prefix=name_prefix,
            save_replay_buffer=save_replay_buffer,
            save_vecnormalize=save_vecnormalize,
            verbose=verbose,
        )
        self.save_on_training_end = bool(save_on_training_end)

    def _on_training_end(self) -> None:
        if self.save_on_training_end and self.save_path is not None:
            checkpoint_file = self._checkpoint_path(extension="zip")
            if not os.path.exists(checkpoint_path := checkpoint_file):
                self.model.save(checkpoint_path)
                if self.verbose >= 1:
                    print(f"Saved end-of-training checkpoint to {checkpoint_path}")


__all__ = ["ALLOProgressBarCallback", "LossEvalCallback", "ALLOCheckpointCallback"]
