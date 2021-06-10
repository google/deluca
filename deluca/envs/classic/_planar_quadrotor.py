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
    m: float = attr.ib(default=0.1)
    l: float = attr.ib(default=0.2)
    g: float = attr.ib(default=9.81)
    dt: float = attr.ib(default=0.05)
    H: float = attr.ib(default=100.0)
    wind: float = attr.ib(default=0.0)
    action_dim: int = attr.ib(default=2)
    state_dim: int = attr.ib(default=6)
    wind_func: Callable = attr.ib(default=dissipative)
    initial_state: jnp.ndarray = attr.ib(default=jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    goal_state: jnp.ndarray = attr.ib(default=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    goal_action: jnp.ndarray = attr.ib()
    last_action: jnp.ndarray = attr.ib(default=jnp.array([0.0, 0.0]))
    h: int = attr.ib(default=0)
    state: jnp.ndarray = attr.ib(metadata={"state": True})

    @goal_action.default
    def __goal_action_default(self):
        return jnp.array([self.m * self.g / 2.0, self.m * self.g / 2.0])

    @state.default
    def __state_default(self):
        return self.initial_state

    def wind_field(self, x, y):
        return self.wind_func(x, y, self.wind)

    def dynamics(self, action):
        x, y, th, xdot, ydot, thdot = self.state
        u1, u2 = action
        m, g, l, dt = self.m, self.g, self.l, self.dt
        wind = self.wind_field(x, y)
        xddot = -(u1 + u2) * jnp.sin(th) / m + wind[0] / m
        yddot = (u1 + u2) * jnp.cos(th) / m - g + wind[1] / m
        thddot = l * (u2 - u1) / (m * l ** 2)
        state_dot = jnp.array([xdot, ydot, thdot, xddot, yddot, thddot])
        new_state = self.state + state_dot * dt
        return self.evolve(state=new_state, last_action=action, h=self.h + 1)

    def cost(self, action):
        return 0.1 * (action - self.goal_action) @ (action - self.goal_action) + (
            self.state - self.goal_state
        ) @ (self.state - self.goal_state)
