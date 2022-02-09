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

"""deluca.agents._hinf"""
from numbers import Real
from typing import Tuple

import jax.numpy as jnp
from jax.numpy.linalg import inv

from deluca.agents.core import Agent


class Hinf(Agent):
    """
    Hinf: H-infinity controller (approximately).  Solves a lagrangian min-max
    dynamic program similiar to H-infinity control rescales disturbance to have
    norm 1.
    """

    def __init__(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        T: jnp.ndarray,
        Q: jnp.ndarray = None,
        R: jnp.ndarray = None,
    ) -> None:
        """
        Description: initializes the Hinf agent

        Args:
            A (jnp.ndarray):
            B (jnp.ndarray):
            T (jnp.ndarray):
            Q (jnp.ndarray):
            R (jnp.ndarray):

        Returns:
            None
        """
        d_x, d_u = B.shape

        if Q is None:
            self.Q = jnp.identity(d_x, dtype=jnp.float32)

        if R is None:
            self.R = jnp.identity(d_u, dtype=jnp.float32)

        # self.K, self.W = solve_hinf(A, B, Q, R, T)
        self.t = 0

    def train(self, A, B, T):
        self.K, self.W = solve_hinf(A, B, self.Q, self.R, T)

    def __call__(self, state) -> jnp.ndarray:
        """
        Description: provide an action given a state

        Args:
            state (jnp.ndarray): the error PID must compensate for

        Returns:
            action (jnp.ndarray): action to take
        """
        action = -self.K[self.t] @ state
        self.t += 1
        return action


def solve_hinf(
    A: jnp.ndarray,
    B: jnp.ndarray,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    T: jnp.ndarray,
    gamma_range: Tuple[Real, Real] = (0.1, 10.8),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Description: solves H-infinity control problem

    Args:
        A (jnp.ndarray):
        B (jnp.ndarray):
        T (jnp.ndarray):
        Q (jnp.ndarray):
        R (jnp.ndarray):

    Returns
        Tuple[jnp.ndarray, jnp.ndarray]:
    """
    gamma_low, gamma_high = gamma_range
    n, m = B.shape
    M = [jnp.zeros(shape=(n, n))] * (T + 1)
    K = [jnp.zeros(shape=(m, n))] * T
    W = [jnp.zeros(shape=(n, 1))] * T

    while gamma_high - gamma_low > 0.01:
        K = [jnp.zeros(shape=(m, n))] * T
        W = [jnp.zeros(shape=(n, 1))] * T

        gamma = 0.5 * (gamma_high + gamma_low)
        M[T] = Q
        for t in range(T - 1, -1, -1):
            Lambda = jnp.eye(n) + (B @ inv(R) @ B.T - gamma ** (-2) * jnp.eye(n)) @ M[t + 1]
            M[t] = Q + A.T @ M[t + 1] @ inv(Lambda) @ A

            K[t] = -inv(R) @ B.T @ M[t + 1] @ inv(Lambda) @ A
            W[t] = (gamma ** (-2)) * M[t + 1] @ inv(Lambda) @ A
            if not is_psd(M[t], gamma):
                gamma_low = gamma
                break
        if gamma_low != gamma:
            gamma_high = gamma
    return K, W


def is_psd(P: jnp.ndarray, gamma: Real) -> bool:
    """
    Description: check if a matrix is positive semi-definite

    Args:
        P (jnp.ndarray): 
        gamma (Real): 

    Returns:
        bool: 
    """
    return jnp.all(jnp.linalg.eigvals(P) < gamma ** 2 + 1e-5)
