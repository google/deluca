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

"""Pendulum."""

from jax import Array
from deluca.core import Env
from deluca.core import field
from deluca.core import Obj
import jax.numpy as jnp


class PendulumState(Obj):
    """PendulumState."""

    arr: jnp.ndarray = field(jaxed=True)
    h: int = field(0, jaxed=True)


class Pendulum(Env):
    """Pendulum."""

    m: float = field(1.0, jaxed=False)
    l: float = field(1.0, jaxed=False)
    g: float = field(9.81, jaxed=False)
    max_torque: float = field(1.0, jaxed=False)
    dt: float = field(0.02, jaxed=False)
    H: int = field(300, jaxed=False)
    goal_state: jnp.ndarray | None = field(default=None, jaxed=False)

    def __init__(
        self,
        rng,
        *,
        m: float = 1.0,
        l: float = 1.0,
        g: float = 9.81,
        max_torque: float = 1.0,
        dt: float = 0.02,
        H: int = 300,
        goal_state: jnp.ndarray | None = None,
    ):
        """Create a Pendulum environment.

        Args:
            rng: JAX PRNGKey.
            m: Mass of the pendulum bob.
            l: Length of the pendulum rod.
            g: Acceleration due to gravity.
            max_torque: Maximum (absolute) control torque.
            dt: Discrete time step.
            H: Horizon length (unused here but kept for API parity).
            goal_state: Optional desired state `(sin θ, cos θ, θ̇)`.
        """

        super().__init__(rng)  # Store the rng if the base class needs it.

        # Physical parameters.
        self.m = m
        self.l = l
        self.g = g
        self.max_torque = max_torque
        self.dt = dt
        self.H = H

        # Target state; lazily initialised in `reset` if not provided.
        self.goal_state = goal_state

        # Keep a private copy of the key so the env can be purely functional
        # if ever required.
        self.key = rng

    def reset(self, rng):
        if self.goal_state is None:
            self.goal_state = jnp.array([0.0, -1.0, 0.0])

        init_arr = jnp.array([0.0, 1.0, 0.0]).reshape(3, 1)

        return 0, PendulumState(arr=init_arr, h=0), init_arr

    def __call__(self, t, state, action, rng):
        """__call__.

        Args:
          state:
          action:

        Returns:

        """
        sin, cos, _ = state.arr
        action = self.max_torque * jnp.tanh(action[0])
        newthdot = jnp.arctan2(sin, cos) + (
            -3.0 * self.g / (2.0 * self.l) * jnp.sin(jnp.arctan2(sin, cos) + jnp.pi)
            + 3.0 / (self.m * self.l**2) * action
        )
        newth = jnp.arctan2(sin, cos) + newthdot * self.dt
        newsin, newcos = jnp.sin(newth), jnp.cos(newth)
        arr = jnp.array([newsin, newcos, newthdot]).reshape(3, 1)

        return t + 1, PendulumState(arr=arr, h=state.h + 1), arr
    
    def loss_fn(self, action: Array, next_state: PendulumState, next_obs: Array) -> Array:
        return jnp.sum(((next_obs[0] - self.goal_state[0]) + (next_obs[1] - self.goal_state[1])) ** 2) + 5 * jnp.sum(action ** 2) # type: ignore

    @property
    def action_size(self) -> int:
        return 1

    @property
    def observation_size(self) -> int:
        return 3
    