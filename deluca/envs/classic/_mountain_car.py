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
import jax
import jax.numpy as jnp


class MountainCar(Env):
    """MountainCar."""

    goal_velocity: float = 0.0
    min_action: float = -1.0
    max_action: float = 1.0
    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    goal_position: float = 0.5
    power = 0.0015

    def __init__(self, rng: jax.Array):
        super().__init__(rng)

    def reset(self, rng: jax.Array):
        """reset.

        Returns:

        """
        self.key, subkey = jax.random.split(rng)
        state = jnp.array([[jax.random.uniform(subkey, minval=-0.6, maxval=0.4)], [0]])

        return 0, state, state

    def __call__(self, t, state, action, rng):
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
        velocity = jnp.where(
            reset_velocity[0],
            jnp.zeros_like(velocity),
            velocity
        )
        new_state = jnp.reshape(jnp.array([position, velocity]), (2,))

        # Add a singleton dimension to the state
        new_state = jnp.expand_dims(new_state, -1)

        t = jax.lax.cond(position[0][0] >= self.goal_position, lambda: 0, lambda: t + 1)
        new_state = jax.lax.cond(position[0][0] >= self.goal_position, lambda: jnp.array([[0.], [0.]]), lambda: new_state)

        return t, new_state, new_state

    def loss_fn(self, action: jax.Array, next_obs: jax.Array) -> jax.Array:
        # Penalize: being further from goal, low velocity, large actions, and negative actions
        position = next_obs[0, 0]
        velocity = next_obs[1, 0]
        action = action[0, 0]
        #jax.debug.print("Position: {}, Velocity: {}, Action: {}", position, velocity, action)
        #jax.debug.print("Loss: {}", loss)

        goal_reward = jnp.where(position >= self.goal_position, 1000.0, 0.0)
        position_reward = position * 10.0
        velocity_reward = jnp.where(position > -0.5, velocity * 5.0, -jnp.abs(velocity) * 2.0)
        action_penalty = 0.1 * jnp.sum(action ** 2)

        return action_penalty - goal_reward - position_reward - velocity_reward
    

    @property
    def action_size(self) -> int:
        return 1
    
    @property
    def observation_size(self) -> int:
        return 2
