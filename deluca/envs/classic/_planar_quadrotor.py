# Copyright 2022 The Deluca Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PlanarQuadrotor."""

from typing import Callable
from typing import List

from deluca.core import Env
from deluca.core import field
from deluca.core import Obj
import jax.numpy as jnp


def dissipative(x, y, wind):
  """Dissipative wind function."""
  return [wind * x, wind * y]


def constant(x, y, wind):
  """Constant wind function."""
  return [wind * jnp.cos(jnp.pi / 4.0), wind * jnp.sin(jnp.pi / 4.0)]


class PlanarQuadrotorState(Obj):
  """PlanarQuadrotorState."""
  arr: jnp.ndarray = field(jaxed=True)
  h: float = field(0.0, jaxed=True)
  last_action: jnp.ndarray = field(jaxed=True)

  def flatten(self):
    return self.arr

  def unflatten(self, arr):
    return PlanarQuadrotorState(arr=arr, h=self.h, last_action=self.last_action)


class PlanarQuadrotor(Env):
  """PlanarQuadrotor."""
  m: float = field(0.1, jaxed=False)
  l: float = field(0.2, jaxed=False)
  g: float = field(9.81, jaxed=False)
  dt: float = field(0.05, jaxed=False)
  H: int = field(100, jaxed=False)
  wind: float = field(0.0, jaxed=False)
  wind_func: Callable[[float, float, float], List[float]] = field(
      dissipative, jaxed=False)
  goal_state: jnp.ndarray = field(jaxed=False)
  goal_action: jnp.ndarray = field(jaxed=False)
  state_dim: float = field(6, jaxed=False)
  action_dim: float = field(2, jaxed=False)

  def setup(self):
    """setup."""
    self.goal_action = jnp.array([self.m * self.g / 2.0, self.m * self.g / 2.0])
    if self.goal_state is None:
      self.goal_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

  def _wind_field(self, x, y):
    """_wind_field.

    Args:
      x:
      y:

    Returns:

    """
    return self.wind_func(x, y, self.wind)

  def init(self):
    """init."""
    new_state = PlanarQuadrotorState(arr=jnp.array([1., 1., 0., 0., 0., 0.]), last_action=jnp.array([0., 0.]))
    return new_state, new_state.arr

  def __call__(self, state, action):
    """__call__."""
    x, y, th, xdot, ydot, thdot = state.arr
    u1, u2 = action
    m, g, l, dt = self.m, self.g, self.l, self.dt
    wind = self._wind_field(x, y)
    xddot = -(u1 + u2) * jnp.sin(th) / m + wind[0] / m
    yddot = (u1 + u2) * jnp.cos(th) / m - g + wind[1] / m
    thddot = l * (u2 - u1) / (m * l**2)
    state_dot = jnp.array([xdot, ydot, thdot, xddot, yddot, thddot])
    arr = state.arr + state_dot * dt
    return state.replace(arr=arr, last_action=action, h=state.h + 1), arr
