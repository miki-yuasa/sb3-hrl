import os
from typing import Any

import numpy as np
from gymnasium.core import ActType, Env, ObsType
from numpy.typing import NDArray
from stable_baselines3.common.base_class import BaseAlgorithm

from sb3_hrl.utils.io import get_class

from .base import BaseSubPolicy

SB3ObsType = np.ndarray | dict[str, np.ndarray]


class SB3SubPolicy(BaseSubPolicy[BaseAlgorithm, SB3ObsType, NDArray]):
    """
    This class defines SB3-based subpolicies used in option-based HRL.
    """

    def define_policy(self, policy_args: dict[str, Any]) -> BaseAlgorithm:
        """
        Define the SB3-based subpolicy. This method initializes the specific SB3 algorithm based on the provided arguments.

        Parameters
        ----------
        policy_args: dict[str, Any]
            A dictionary of arguments required to initialize the SB3 subpolicy, including:
            - model_cls: str, the class name of the SB3 algorithm (e.g., "PPO", "DQN").
            - model_save_path: str, the path to load the pre-trained model.
            - device: str, the device to run the model on (e.g., "cpu", "cuda").
        """

        self.policy_args = policy_args

        if os.path.exists(self.policy_args["model_save_path"]):
            if self.verbose:
                print(f"Loading model from {self.policy_args['model_save_path']}")
            policy_cls: type[BaseAlgorithm] = get_class(self.policy_args["model_cls"])
            policy: BaseAlgorithm = policy_cls.load(
                self.policy_args["model_save_path"], device=self.policy_args["device"]
            )
        else:
            raise FileNotFoundError(
                f"Model not found at {self.policy_args['model_save_path']}, train a model first."
            )
        return policy

    def act(
        self,
        obs: SB3ObsType,
        info: dict[str, Any] | None = None,
        current_env: Env[ObsType, ActType] | None = None,
    ) -> NDArray:
        """
        Predict the action using the low-level policy.
        """

        action, _ = self.policy.predict(obs)
        # Ensure action is a numpy int64 scalar
        return action
