# Copyright 2024 The Deluca Authors.
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

"""Planar quadrotor."""
from typing import Protocol

from deluca._src import base
import jax.numpy as jnp

STATE_DIM = 6
ACTION_DIM = 2


class WindFunction(Protocol):
  """A callable type for describing wind."""

  def __call__(self, x: float, y: float):
    """Wind function.

    Args:
      x: Position of planar quadrotor in the x axis.
      y: Position of planar quadrotor in the y axis.

    Returns:
      tuple[float, float]: Wind force in the x axis and y axis, respectively.
    """


def make_dissipative(wind_x: float, wind_y: float):
  """Make dissipative wind function.

  Args:
    wind_x: Wind force in the x axis.
    wind_y: Wind force in the y axis.

  Returns:
    Dissipative wind function.
  """

  def dissipative(x: float, y: float):
    """Dissipative wind function."""
    return wind_x * x, wind_y * y

  return dissipative


def make_constant(
    wind_x: float = jnp.cos(jnp.pi / 4.0), wind_y: float = jnp.sin(jnp.pi / 4.0)
):
  """Make constant wind function.

  Args:
    wind_x: Wind force in the x axis.
    wind_y: Wind force in the y axis.

  Returns:
    Constant wind function.
  """

  def constant(
      x: float,
      y: float,
  ):
    """Constant wind function."""
    del x, y
    return wind_x, wind_y

  return constant


def planar_quadrotor(
    mass: float = 0.1,
    length: float = 0.2,
    gravity: float = 9.81,
    dt: float = 0.05,
    wind_fn: WindFunction = make_dissipative(wind_x=0.0, wind_y=0.0),
) -> base.Environment:
  """Planar quadrotor.
  
  Args:
    mass: Mass of planar quadrotor.
    length: Overall length of planar quadrotor.
    gravity: Gravity applied to quadrotor in the y axis.
    dt: Time delta.
    wind_fn: Function describing wind force at location (x, y).
  
  Returns:
    A planar quadrotor environment.
  """
  def init_fn(
      x: float = 0.0,
      y: float = 0.0,
      theta: float = 0.0,
      x_dot: float = 0.0,
      y_dot: float = 0.0,
      theta_dot: float = 0.0,
  ):
    """Create initial state for planar quadrotor.

    Args:
      x: Position of planar quadrotor in the x axis.
      y: Position of planar quadrotor in the y axis.
      theta: Angle of rotation of planar quadrotor.
      x_dot: Velocity of planar quadrotor in the x axis.
      y_dot: Velocrity of planar quadrotor in the y axis.
      theta_dot: Angular velocity of planar quadrotor

    Returns:
      Initial state for planar quadrotor as a 6-dimensional array.
    """
    return jnp.array([x, y, theta, x_dot, y_dot, theta_dot])

  def update_fn(state: jnp.ndarray, action: jnp.ndarray):
    x, y, theta, x_dot, y_dot, theta_dot = state
    u1, u2 = action
    wind_x, wind_y = wind_fn(x, y)

    x_ddot = -(u1 + u2) * jnp.sin(theta) / mass + wind_x / mass
    y_ddot = (u1 + u2) * jnp.cos(theta) / mass - gravity + wind_y / mass
    theta_ddot = length * (u2 - u1) / (mass * length**2)

    state_dot = jnp.array([x_dot, y_dot, theta_dot, x_ddot, y_ddot, theta_ddot])

    return state + state_dot * dt

  return base.Environment(init_fn, update_fn)
