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
"""deluca.agents._pid"""
from numbers import Real
from typing import Sequence

import jax.numpy as jnp

from deluca.agents.core import Agent

DEFAULT_K = [3.0, 4.0, 0.0]


class PID(Agent):
    """
    PID: agent that plays a PID policy

    Todo:
      - Allow for variable time deltas
    """

    def __init__(self, K: Sequence[int] = None, RC: Real = 0.5, dt: Real = 0.03, **kwargs) -> None:
        """
        Description: initializes the PID agent

        Args:
            K (Sequence[int]): sequence of PID parameters
            RC (Real): decay parameter
            dt (Real): time increment

        Returns:
            None
        """
        if not isinstance(K, Sequence) or len(K) != 3 or not all(isinstance(x, Real) for x in K):
            self.throw(ValueError, "K must be a list or tuple of 3 real numbers, P, I, and D")

        # controller coefficients
        self.K_P, self.K_I, self.K_D = K or DEFAULT_K

        # controller states
        self.P, self.I, self.D = 0.0, 0.0, 0.0

        self.RC = RC
        self.dt = dt

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Description: provide an action given a state

        Args:
            state (jnp.ndarray): the error PID must compensate for

        Returns:
            jnp.ndarray: action to take
        """
        decay = self.dt / (self.dt + self.RC)

        self.I += decay * (state - self.I)
        self.D += decay * (state - self.P - self.D)
        self.P = state

        return self.K_P * self.P + self.K_I * self.I + self.K_D * self.D
