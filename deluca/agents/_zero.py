# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""deluca.agents._zero"""
from typing import Sequence

import jax.numpy as jnp

from deluca.agents.core import Agent


class Zero(Agent):
    """
    Zero: agent that plays the zero action
    """

    def __init__(self, shape: Sequence[int]) -> None:
        """
        Description: iniitalizes the Zero agent

        Args:
            shape: shape of the action the agent should take

        Returns:
            None

        Notes:
            - The Zero agent cannot, in this implementation, return a scalar
              that passes jnp.isscalar; we can emulate via Zero(shape=()), whic
              would return jnp.array(0.0)
        """
        if not isinstance(shape, Sequence) or not all(isinstance(x, int) for x in shape):
            self.throw(ValueError, "Shape must be a list or tuple of ints")

        self.shape = tuple(shape)
        self.action = jnp.zeros(shape)

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Description: provide an action given a state

        Args:
            state (jnp.ndarray): current state

        Returns:
            action (jnp.ndarray): action to take
        """
        return self.action
