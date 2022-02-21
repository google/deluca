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

"""@author: Olivier Sigaud A merge between two sources:
* Adaptation of the MountainCar Environment from the "FAReinforcement" library
* of Jose Antonio Martin H.

(version 1.0), adapted by  'Tom Schaul, tom@idsia.ch' and then modified by
Arnaud de Broissia * the OpenAI/gym MountainCar environment itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp permalink:
https://perma.cc/6Z2N-PFWC
"""

# pylint:disable=g-long-lambda

from deluca.core import Env
from deluca.core import field
import jax
import jax.numpy as jnp


class MountainCar(Env):
  """MountainCar."""
  key: jnp.ndarray = field(jaxed=False)
  goal_velocity: float = field(0.0, jaxed=False)
  min_action: float = field(-1.0, jaxed=False)
  max_action: float = field(1.0, jaxed=False)
  min_position: float = field(-1.2, jaxed=False)
  max_position: float = field(0.6, jaxed=False)
  max_speed: float = field(0.07, jaxed=False)
  goal_position: float = field(0.5, jaxed=False)
  power = 0.0015

  low_state: jnp.ndarray = field(jaxed=False)
  high_state: jnp.ndarray = field(jaxed=False)

  def setup(self):
    """setup."""
    self.low_state = jnp.array([self.min_position, -self.max_speed])
    self.high_state = jnp.array([self.max_position, self.max_speed])
    if self.key is None:
      self.key = jax.random.PRNGKey(0)

  def init(self):
    """init.

    Returns:

    """
    state = jnp.array(
        [jax.random.uniform(self.key, min_val=-0.6, maxval=0.4), 0])
    return state, state

  def __call__(self, state, action):
    """__call__.

    Args:
      state:
      action:

    Returns:

    """
    position, velocity = state

    force = jnp.minimum(jnp.maximum(action, self.min_action), self.max_action)

    velocity += force * self.power - 0.0025 * jnp.cos(3 * position)
    velocity = jnp.clip(velocity, -self.max_speed, self.max_speed)

    position += velocity
    position = jnp.clip(position, self.min_position, self.max_position)
    reset_velocity = (position == self.min_position) & (velocity < 0)
    velocity = jax.lax.cond(reset_velocity[0], velocity, lambda x: jnp.zeros(
        (1,)), velocity, lambda x: x)
    new_state = jnp.reshape(jnp.array([position, velocity]), (2,))

    return new_state, new_state
