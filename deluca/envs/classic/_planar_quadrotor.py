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

import jax.numpy as jnp

from deluca.core import env
from deluca.core import evolve
from deluca.core import state


def dissipative(x, y, wind):
    return [wind * x, wind * y]


def constant(x, y, wind):
    return [wind * jnp.cos(jnp.pi / 4.0), wind * jnp.sin(jnp.pi / 4.0)]


state_init = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
m_init, g_init = 0.1, 9.81


@env
class PlanarQuadrotor:
    m = m_init
    l = 0.2
    g = g_init
    dt = 0.05
    H = 100
    wind = 0.0
    wind_func = dissipative
    initial_state = state_init
    goal_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_action = jnp.array([m_init * g_init / 2.0, m_init * g_init / 2.0])
    last_action = jnp.array([0.0, 0.0])
    h = 0
    state = state(state_init)

    def _wind_field(self, x, y):
        return self.wind_func(x, y, self.wind)

    @staticmethod
    def reset(env):
        return evolve(env, state=env.initial_state)

    @staticmethod
    def dynamics(env, action):
        x, y, th, xdot, ydot, thdot = env.state
        u1, u2 = action
        m, g, l, dt = env.m, env.g, env.l, env.dt
        wind = env.wind_field(x, y)
        xddot = -(u1 + u2) * jnp.sin(th) / m + wind[0] / m
        yddot = (u1 + u2) * jnp.cos(th) / m - g + wind[1] / m
        thddot = l * (u2 - u1) / (m * l ** 2)
        state_dot = jnp.array([xdot, ydot, thdot, xddot, yddot, thddot])
        new_state = env.state + state_dot * dt
        return evolve(env, state=new_state, last_action=action, h=env.h + 1)

    @staticmethod
    def cost(env, action):
        return 0.1 * (action - env.goal_action) @ (action - env.goal_action) + (
            env.state - env.goal_state
        ) @ (env.state - env.goal_state)
