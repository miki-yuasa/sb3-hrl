"""Visualization utilities for option-based hierarchical RL rollouts."""

from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import Any

import imageio
import numpy as np
from gymnasium import Env
from stable_baselines3.common.base_class import BaseAlgorithm


def record_option_replay(
    demo_env: Env[Any, Any],
    model: BaseAlgorithm,
    animation_save_path: str,
    verbose: bool = True,
    close_env: bool = True,
    fps: int = 15,
) -> None:
    """Records an option evaluation episode replay with all primitive step frames.

    Args:
        demo_env: Option evaluation environment to rollout.
        model: Meta-controller algorithm used to sample high-level option actions.
        animation_save_path: Filepath where the animation should be saved.
        verbose: Whether to print rollout transition details.
        close_env: Whether to close the environment upon recording completion.
        fps: Frames per second for the saved animation.
    """
    try:
        demo_env.set_wrapper_attr("record_render_frames", True)
    except (AttributeError, ValueError):
        pass

    try:
        obs, _ = demo_env.reset()
        terminated: bool = False
        truncated: bool = False
        first_frame = demo_env.render()
        frames: list[Any] = []
        if isinstance(first_frame, list):
            frames.extend(first_frame)
        elif first_frame is not None:
            frames.append(first_frame)

        rewards: list[float] = []
        macro_steps = 0
        total_primitive_steps = 0

        while not (terminated or truncated):
            macro_steps += 1
            action, _ = model.predict(obs)  # type: ignore[assignment]
            obs, reward, terminated, truncated, info = demo_env.step(action)
            rewards.append(float(reward))

            step_frames: list[Any] | None = info.get("render_frames")
            if step_frames is None:
                pop_fn = getattr(demo_env, "pop_render_frames", None)
                if callable(pop_fn):
                    step_frames = pop_fn()

            if step_frames:
                frames.extend(step_frames)
                step_primitive_steps = len(step_frames)
            else:
                frame = demo_env.render()
                if isinstance(frame, list):
                    frames.extend(frame)
                    step_primitive_steps = len(frame)
                elif frame is not None:
                    frames.append(frame)
                    step_primitive_steps = 1
                else:
                    step_primitive_steps = int(info.get("meta_option_steps", 1))

            total_primitive_steps += step_primitive_steps

            if verbose:
                print(
                    f"Macro Step {macro_steps} "
                    f"(primitive steps: {step_primitive_steps}, "
                    f"cumulative: {total_primitive_steps}):"
                )
                print(f" - Action taken: {action}")
                if "current_tl_spec" in info:
                    print(f" - Specification: {info['current_tl_spec']}")
                print(
                    f" - Reward: {float(reward):.2f}, Terminated: {terminated}, "
                    f"Truncated: {truncated}, "
                    f"Success: {info.get('is_success', 'N/A')}"
                )
                info_to_print = {
                    k: f"<{len(v)} frames>" if k == "render_frames" else v
                    for k, v in info.items()
                }
                print(" - Info: ")
                pprint(info_to_print)

        if verbose:
            print(
                f" - Total reward: {sum(rewards):.2f} "
                f"across {macro_steps} macro steps "
                f"({total_primitive_steps} primitive steps)"
            )

        save_path = Path(animation_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = 1000.0 / max(1, fps)
        imageio.mimsave(save_path, frames, duration=duration_ms, loop=0)  # type: ignore[call-overload]
        if verbose:
            print(
                f" - Replay saved to {animation_save_path} "
                f"({len(frames)} frames @ {fps} fps)"
            )
    finally:
        try:
            demo_env.set_wrapper_attr("record_render_frames", False)
        except (AttributeError, ValueError):
            pass
        if close_env:
            demo_env.close()
