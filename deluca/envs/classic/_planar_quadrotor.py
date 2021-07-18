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

from typing import Callable

import jax.numpy as jnp

from deluca.core import Env
from deluca.core import field
from deluca.core import Obj


def dissipative(x, y, wind):
    return [wind * x, wind * y]


def constant(x, y, wind):
    return [wind * jnp.cos(jnp.pi / 4.0), wind * jnp.sin(jnp.pi / 4.0)]


class PlanarQuadrotorState(Obj):
    arr: jnp.ndarray = field(trainable=True)
    h: int = field(0, trainable=True)


class PlanarQuadrotor(Env):
    m: float = field(0.1, trainable=False)
    l: float = field(0.2, trainable=False)
    g: float = field(9.81, trainable=False)
    dt: float = field(0.05, trainable=False)
    H: int = field(100, trainable=False)
    wind: float = field(0.0, trainable=False)
    wind_func: Callable = field(dissipative, trainable=False)
    goal_state: jnp.ndarray = field(jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), trainable=False)
    goal_action: jnp.ndarray = field(trainable=False)
    last_action: jnp.ndarray = field(jnp.array([0.0, 0.0]), trainable=False)

    def setup(self):
        self.goal_action = jnp.array([self.m * self.g / 2.0, self.m * self.g / 2.0])

    def _wind_field(self, x, y):
        return self.wind_func(x, y, self.wind)

    def init(self):
        return PlanarQuadrotorState(arr=jnp.array[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], h=0)

    def __call__(self, state, action):
        x, y, th, xdot, ydot, thdot = state.arr
        u1, u2 = action
        m, g, l, dt = self.m, self.g, self.l, self.dt
        wind = self.wind_field(x, y)
        xddot = -(u1 + u2) * jnp.sin(th) / m + wind[0] / m
        yddot = (u1 + u2) * jnp.cos(th) / m - g + wind[1] / m
        thddot = l * (u2 - u1) / (m * l ** 2)
        state_dot = jnp.array([xdot, ydot, thdot, xddot, yddot, thddot])
        arr = state.arr + state_dot * dt
        return state.replace(arr=arr, last_action=action, h=state.h + 1), arr
