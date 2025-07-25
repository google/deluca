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

"""deluca.agents._grc"""
from typing import Callable

import jax
import jax.numpy as jnp

from jax import grad, jit, Array

from deluca.core import Agent


def quad_loss(y: Array, u: Array) -> Array:
    """
    Quadratic loss.

    Args:
        y (Array):
        u (Array):

    Returns:
        Real
    """
    return jnp.sum(y.T @ y + u.T @ u)


class GRC(Agent):
    def __init__(
        self,
        A: Array,
        B: Array,
        C: Array,
        rng: Array,
        cost_fn: Callable[[Array, Array], Array] = quad_loss,
        R_M: float = 10.0,
        init_scale: float = 1.0,
        m: int = 10,
        lr: float = 0.005,
        decay: bool = True,
    ) -> None:
        """
        Description: Initialize the dynamics of the model.

        Args:
            A (Array): system dynamics
            B (Array): system dynamics
            C (Array): system dynamics
            cost_fn (Callable[[Array, Array], Array]):
            R_M (float): Diameter of the learnable parameters M
            key (jax.random.key): random key
            init_scale (float): Initial scale of the learnable parameters
            m (postive int): history of the controller
            lr (float): learning rate
            decay (bool): whether to decay the learning rate
        """

        self.A, self.B, self.C = A, B, C  # System Dynamics
        self.d, self.n, self.p = (
            self.A.shape[0],
            self.B.shape[1],
            self.C.shape[0],
        )  # State, Action, Observation Dimensions
        self.cost_fn = cost_fn  # Cost Function

        self.m = m
        self.R_M = R_M
        self.lr = lr
        self.decay = decay

        self.t = 0  # Time Counter (for decaying learning rate)
        self.M = init_scale * jax.random.normal(rng, shape=(self.m, self.n, self.p))

        self.z = jnp.zeros((self.d, 1))
        self.ynat_history = jnp.zeros((2 * self.m, self.p, 1))

        def last_m_ynats():
            return jax.lax.dynamic_slice(
                self.ynat_history, (self.m, 0, 0), (self.m, self.p, 1)
            )

        self.last_m_ynats = last_m_ynats

        def policy_loss(M, ynats):
            def action(k):
                window = jax.lax.dynamic_slice(ynats, (k, 0, 0), (self.m, self.p, 1))
                contribs = jnp.einsum("mnp,mp1->mn1", M, window)
                return jnp.sum(contribs, axis=0)

            def evolve(delta, h):
                return self.A @ delta + self.B @ action(h), None

            final_delta, _ = jax.lax.scan(
                evolve, jnp.zeros((self.d, 1)), jnp.arange(self.m)
            )
            final_y = self.C @ final_delta + ynats[-1]
            return self.cost_fn(final_y, action(self.m))

        self.policy_loss = policy_loss
        self.grad = jit(grad(policy_loss, (0)))

    def __call__(self, obs, rng) -> Array:
        """
        Description: Return the action based on current observation and internal parameters.

        Args:
            y (Array): current observation

        Returns:
           Array: action to take
        """

        window = self.last_m_ynats()
        contribs = jnp.einsum("mnp,mp1->mn1", self.M, window)
        u =  jnp.sum(contribs, axis=0)
        
        self.update(obs, u)
        return u

    def update(self, obs: Array, action: Array) -> None:
        """
        Description: update agent internal state.

        Args:
            obs (Array):
            action (Array):

        Returns:
            None
        """
        self.z = self.A @ self.z + self.B @ action
        y_nat = obs - self.C @ self.z
        self.ynat_history = self.ynat_history.at[0].set(y_nat)
        self.ynat_history = jnp.roll(self.ynat_history, -1, axis=0)

        delta_M = self.grad(self.M, self.ynat_history)
        lr = self.lr * (1 / (self.t + 1) if self.decay else 1)
        self.M -= lr * delta_M

        norm = jnp.linalg.norm(self.M)
        scale = jnp.minimum(1.0, self.R_M / (norm + 1e-8))
        self.M = scale * self.M

        self.t += 1