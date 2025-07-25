from abc import abstractmethod
from typing import Tuple

import flax.struct
from jax import Array
import jax.numpy as jnp

class Filter:

    filter_history_length: int

    filtered_obs_dim: int
    filtered_action_dim: int

    @abstractmethod
    def __init__(self, filter_history_length: int, obs_dim: int, action_dim: int):
        super().__init__()
        self.filter_history_length = filter_history_length

    def _noop(self, a_history: Array) -> Array:
        last_action = a_history[-1]
        return jnp.astype(last_action, a_history.dtype)

    @abstractmethod
    def __call__(self, obs_history: Array, action_history: Array) -> Tuple[Array, Array]:
        """
        Filter the history. 
        """