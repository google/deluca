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

"""deluca.agents._sfc"""
from numbers import Real
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad
from jax import jit

from deluca.core import Agent


def quad_loss(y: jnp.ndarray, u: jnp.ndarray) -> Real:
    """
    Quadratic loss.

    Args:
        x (jnp.ndarray):
        u (jnp.ndarray):

    Returns:
        Real
    """
    return jnp.sum(y.T @ y + u.T @ u)


class SFC(Agent):
    def __init__(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        C: jnp.ndarray,
        cost_fn: Callable[[jnp.ndarray, jnp.ndarray], Real] = None,
        R_M: float = 10.0,
        gamma: float = 0.2,
        key: jax.random.key = jax.random.PRNGKey(0),
        init_scale: float = 1.0,
        h: int = 10,
        m: int = 30,
        lr: Real = 0.005,
        decay: bool = True,
    ) -> None:
        """
        Description: Initialize the dynamics of the model.

        Args:
            A (jnp.ndarray): system dynamics
            B (jnp.ndarray): system dynamics
            C (jnp.ndarray): system dynamics
            cost_fn (Callable[[jnp.ndarray, jnp.ndarray], Real]):
            R_M (float): Diameter of the learnable parameters M
            gamma (float): stability of the controller
            key (jax.random.key): random key
            init_scale (float): Initial scale of the learnable parameters
            h (postive int): number of parameters
            m (postive int): history of the controller
            lr (Real): learning rate
            decay (bool): whether to decay the learning rate
        """

        self.A, self.B, self.C = A, B, C  # System Dynamics
        self.d, self.n, self.p = self.A.shape[0], self.B.shape[1], self.C.shape[0] # State, Action, Observation Dimensions
        self.cost_fn = cost_fn or quad_loss # Cost Function

        self.lr = lr
        self.decay = decay
        self.R_M = R_M

        self.m = m
        self.h = h

        self.t = 0
        subkey_1, subkey2 = jax.random.split(key)
        self.M = init_scale*jax.random.normal(subkey1, shape=(self.h, self.n, self.p))
        self.M_0 = init_scale*jax.random.normal(subkey2, shape=(self.n, self.p))

        self.z = jnp.zeros((self.d, 1))
        self.ynat_history = jnp.zeros((2*self.m+1, self.p, 1))

        # Construct Filters
        indices = np.arange(1, self.m + 1)
        I, J = np.meshgrid(indices, indices, indexing='ij')
        H_matrix = ((1 - gamma) ** (I + J - 1)) / (I + J - 1)

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(H_matrix)

        # Get top H eigenvalues and eigenvectors
        top_eigvals = eigvals[-self.h:]         # shape: (h,)
        top_eigvecs = eigvecs[:, -self.h:]      # shape: (m, h)

        # Scale each eigenvector by sigma_i^{1/4}
        scaling = top_eigvals**0.25        # shape: (h,)
        self.filters = jnp.flip((scaling[:, None] * top_eigvecs.T), axis=1)  # shape: (h, m)

        def last_m_ynats():
            return jax.lax.dynamic_slice(self.ynat_history, (self.m, 0, 0), (self.m, self.p, 1))

        self.last_m_ynats = last_m_ynats

        def last_ynat():
            return jnp.squeeze(jax.lax.dynamic_slice(self.ynat_history, (2*self.m, 0, 0), (1, self.p, 1)), axis=0)

        self.last_ynat = last_ynat

        def policy_loss(M_0, M, ynats):
            def action(k):
                window = jax.lax.dynamic_slice(ynats, (k, 0, 0), (self.m, self.p, 1))
                last_ynat = jnp.squeeze(jax.lax.dynamic_slice(ynats, (k+self.m, 0, 0), (1, self.p, 1)), axis=0)
                proj = jnp.einsum("hm,mp1->hp1", self.filters, window)  # (h, p, 1)
                spectral_term = jnp.einsum("hnp,hp1->n1", M, proj)
                return M_0 @ last_ynat + spectral_term

            def evolve(delta, h):
                return self.A @ delta + self.B @ action(h), None

            final_delta, _ = jax.lax.scan(evolve, jnp.zeros((self.d, 1)), jnp.arange(self.m))
            final_y = self.C @ final_delta + ynats[-1]
            return self.cost_fn(final_y, action(self.m))

        self.policy_loss = policy_loss
        self.grad = jit(grad(policy_loss, (0, 1)))


    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        Description: Return the action based on current observation and internal parameters.

        Args:
            y (jnp.ndarray): current observation

        Returns:
           jnp.ndarray: action to take
        """

        u = self.get_action(y)
        self.update(y, u)
        return u

    def update(self, y: jnp.ndarray, u: jnp.ndarray) -> None:
        """
        Description: update agent internal state.

        Args:
            y (jnp.ndarray):
            u (jnp.ndarray):

        Returns:
            None
        """
        self.z = self.A @ self.z + self.B @ u
        y_nat = y - self.C @ self.z
        self.ynat_history = self.ynat_history.at[0].set(y_nat)
        self.ynat_history = jnp.roll(self.ynat_history, -1, axis=0)

        delta_M_0, delta_M = self.grad(self.M_0, self.M, self.ynat_history)
        lr = self.lr * (1 / (self.t + 1) if self.decay else 1)
        self.M_0 -= lr*delta_M_0
        self.M -= lr * delta_M

        norm = jnp.sqrt(jnp.sum(self.M ** 2) + jnp.sum(self.M_0 ** 2))
        scale = jnp.minimum(1.0, self.R_M / (norm + 1e-8))
        self.M_0 = scale*self.M_0
        self.M = scale*self.M

        self.t += 1

    def get_action(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        Description: get action from observation.

        Args:
            y (jnp.ndarray):

        Returns:
            jnp.ndarray
        """
        window = self.last_m_ynats()
        last_ynat = self.last_ynat()
        proj = jnp.einsum("hm,mp1->hp1", self.filters, window)  # (h, p, 1)
        spectral_term = jnp.einsum("hnp,hp1->n1", self.M, proj)
        return self.M_0 @ last_ynat + spectral_term
