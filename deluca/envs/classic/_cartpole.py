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

"""Cartpole."""

from deluca.core import Env
from deluca.core import field
from deluca.core import Obj
import jax.numpy as jnp


# pylint:disable=invalid-name


class CartpoleState(Obj):
  """CartpoleState."""
  arr: jnp.ndarray = field(jaxed=True)
  h: int = field(0, jaxed=True)
  offset: float = field(0.0, jaxed=False)


class Cartpole(Env):
  """Cartpole."""
  m: float = field(0.1, jaxed=False)
  M: float = field(1.0, jaxed=False)
  l: float = field(1.0, jaxed=False)
  g: float = field(9.81, jaxed=False)
  dt: float = field(0.02, jaxed=False)
  H: int = field(10, jaxed=False)
  goal_state: jnp.ndarray = field(jaxed=False)
  dynamics: bool = field(False, jaxed=False)

  def setup(self):
    """setup."""
    if self.goal_state is None:
      self.goal_state = jnp.array([0.0, 0.0, 0.0, 0.0])

  def init(self):
    """init.

    Returns:

    """
    state = CartpoleState(arr=jnp.array([0.0, 0.0, 0.0, 0.0]))
    return state, state

  def __call__(self, state, action):
    """__call__.

    Args:
      state:
      action:

    Returns:

    """
    A = jnp.array([
        [1.0, 0.0, self.dt, 0.0],
        [0.0, 1.0, 0.0, self.dt],
        [0.0, self.dt * self.m * self.g / self.M, 1.0, 0.0],
        [
            0.0, self.dt * (self.m + self.M) * self.g / (self.M * self.l), 0.0,
            1.0
        ],
    ])
    B = (jnp.array([[0.0], [0.0], [self.dt / self.M],
                    [self.dt / (self.M * self.l)]]),)
    arr = (A @ state.arr + B @ (action + state.offset),)
    return state.replace(arr=arr, h=state.h + 1), arr
