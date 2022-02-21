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

"""deluca.agents._adaptive"""
from numbers import Real
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from deluca.agents.core import Agent


def quad_loss(x: jnp.ndarray, u: jnp.ndarray) -> Real:
    """
    Quadratic loss.

    Args:
        x (jnp.ndarray):
        u (jnp.ndarray):

    Returns:
        Real
    """
    return jnp.sum(x.T @ x + u.T @ u)


def lifetime(x, lower_bound):
    l = 4
    while x % 2 == 0:
        l *= 2
        x /= 2

    return max(lower_bound, l + 1)


class Adaptive(Agent):
    def __init__(
        self,
        T: int,
        base_controller,
        A: jnp.ndarray,
        B: jnp.ndarray,
        cost_fn: Callable[[jnp.ndarray, jnp.ndarray], Real] = None,
        HH: int = 10,
        eta: Real = 0.5,
        eps: Real = 1e-6,
        inf: Real = 1e6,
        life_lower_bound: int = 100,
        expert_density: int = 64,
    ) -> None:

        self.A, self.B = A, B
        self.n, self.m = B.shape

        cost_fn = cost_fn or quad_loss

        self.base_controller = base_controller

        # Start From Uniform Distribution
        self.T = T
        self.weights = np.zeros(T)
        self.weights[0] = 1.0

        # Track current timestep
        self.t, self.expert_density = 0, expert_density

        # Store Model Hyperparameters
        self.eta, self.eps, self.inf = eta, eps, inf

        # State and Action
        self.x, self.u = jnp.zeros((self.n, 1)), jnp.zeros((self.m, 1))

        # Alive set
        self.alive = jnp.zeros((T,))

        # Precompute time of death at initialization
        self.tod = np.arange(T)
        for i in range(1, T):
            self.tod[i] = i + lifetime(i, life_lower_bound)
        self.tod[0] = life_lower_bound  # lifetime not defined for 0

        # Maintain Dictionary of Active Learners
        self.learners = {}
        self.learners[0] = base_controller(A, B, cost_fn=cost_fn)

        self.w = jnp.zeros((HH, self.n, 1))

        def policy_loss(controller, A, B, x, w):
            def evolve(x, h):
                """Evolve function"""
                return A @ x + B @ controller.get_action(x) + w[h], None

            final_state, _ = jax.lax.scan(evolve, x, jnp.arange(HH))
            return cost_fn(final_state, controller.get_action(final_state))

        self.policy_loss = policy_loss

    def __call__(self, x, A, B):

        play_i = np.argmax(self.weights)
        self.u = self.learners[play_i].get_action(x)

        # Update alive models
        for i in jnp.nonzero(self.alive)[0]:
            loss_i = self.policy_loss(self.learners[i], A, B, x, self.w)
            self.weights[i] *= np.exp(-self.eta * loss_i)
            self.weights[i] = min(max(self.weights[i], self.eps), self.inf)
            self.learners[i].update(x, u=self.u)

        self.t += 1

        # One is born every expert_density steps
        if self.t % self.expert_density == 0:
            self.alive = self.alive.at[self.t].set(1)
            self.weights[self.t] = self.eps
            self.learners[self.t] = self.base_controller(A, B, cost_fn=self.cost_fn)
            self.learners[self.t].x = x

        # At most one dies
        kill_list = jnp.where(self.tod == self.t)
        if len(kill_list[0]):
            kill = int(kill_list[0][0])
            if self.alive[kill]:
                self.alive = self.alive.at[kill].set(0)
                del self.learners[kill]
                self.weights[kill] = 0

        # Rescale
        max_w = np.max(self.weights)
        if max_w < 1:
            self.weights /= max_w

        # Get new noise (will be located at w[-1])
        self.w = self.w.at[0].set(x - self.A @ self.x + self.B @ self.u)
        self.w = jnp.roll(self.w, -1, axis=0)

        # Update System
        self.x, self.A, self.B = x, A, B

        return self.u
