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
from numbers import Real
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_discrete_are

from deluca.core import Agent


class LQG(Agent):
    def __init__(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        C: jnp.ndarray,
        Q: jnp.ndarray = None,
        R: jnp.ndarray = None,
        W: jnp.ndarray = None,
        V: jnp.ndarray = None,
    ) -> None:
        """
        Description: Initialize the dynamics of the model.

        Args:
            A (jnp.ndarray): system dynamics
            B (jnp.ndarray): system dynamics
            C (jnp.ndarray): system dynamics
            Q (jnp.ndarray): cost function matrix for x.T@Q@x
            R (jnp.ndarray): cost function matrix for u.T@R@u
            W (jnp.ndarray): Noise covariance matrix for x
            V (jnp.ndarray): Noise covariance matrix for y
        """

        self.A, self.B, self.C = A, B, C  # System Dynamics
        self.d, self.n, self.p = (
            self.A.shape[0],
            self.B.shape[1],
            self.C.shape[0],
        )  # State, Action, Observation Dimensions

        if Q is None:
            Q = jax.numpy.identity(self.d)
        if R is None:
            R = jax.numpy.identity(self.n)
        if W is None:
            W = jax.numpy.identity(self.d)
        if V is None:
            V = jax.numpy.identity(self.p)

        self.P = jnp.array(
            solve_discrete_are(
                np.asarray(A), np.asarray(B), np.asarray(Q), np.asarray(R)
            )
        )
        self.K = jnp.linalg.inv(self.B.T @ self.P @ self.B + R) @ (
            self.B.T @ self.P @ self.A
        )
        Sigma = jnp.array(
            solve_discrete_are(
                np.asarray(A).T, np.asarray(C).T, np.asarray(W), np.asarray(V)
            )
        )
        self.L = Sigma @ C.T @ jnp.linalg.inv(C @ Sigma @ C.T + V)

        self.x_hat = jnp.zeros((self.d, 1))

    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        Description: Return the action based on current observation and internal parameters.

        Args:
            y (jnp.ndarray): current observation

        Returns:
           jnp.ndarray: action to take
        """

        u = -self.K @ self.x_hat
        residual = y - self.C @ self.x_hat
        self.x_hat = self.A @ self.x_hat + self.B @ u + self.L @ residual
        return u

    def update(self, y: jnp.ndarray, u: jnp.ndarray) -> None:
        return None
