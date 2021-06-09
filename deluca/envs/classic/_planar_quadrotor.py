# Copyright 2020 Google LLC
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

from typing import Callable

import attr
import deluca.core
import jax.numpy as jnp

dissipative = lambda x, y, wind: [wind * x, wind * y]
constant = lambda x, y, wind: [wind * jnp.cos(jnp.pi / 4.0), wind * jnp.sin(jnp.pi / 4.0)]


@deluca.env
class PlanarQuadrotor:
    m: float = attr.ib(default=0.1, metadata={"state": False})
    l: float = attr.ib(default=0.2, metadata={"state": False})
    g: float = attr.ib(default=9.81, metadata={"state": False})
    dt: float = attr.ib(default=0.05, metadata={"state": False})
    H: float = attr.ib(default=100.0, metadata={"state": False})
    wind: float = attr.ib(default=0.0, metadata={"state": False})
    action_dim: int = attr.ib(default=2, metadata={"state": False})
    state_dim: int = attr.ib(default=6, metadata={"state": False})
    wind_func: Callable = attr.ib(default=dissipative)
    initial_state: jnp.ndarray = attr.ib(default=jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    goal_state: jnp.ndarray = attr.ib(default=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    goal_action: jnp.ndarray = attr.ib()

    @goal_action.default
    def __goal_action_default(self):
        return jnp.array([self.m * self.g / 2.0, self.m * self.g / 2.0])

    def dynamics(self, action):
        return self

    def cost(self, action):
        return 1
