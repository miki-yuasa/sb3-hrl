from typing import Callable, Generic

import numpy as np
from gymnasium.core import ActType, ObsType

from .subpolicies import PolicyArgsType, PolicyType


class Option(Generic[PolicyType, ObsType, ActType]):
    def __init__(
        self,
        policy: PolicyType,
    ):
        self.policy = policy
        self.obs_space = obs_space
        self.act_space = act_space
