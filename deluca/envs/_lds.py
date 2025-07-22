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

"""Linear dynamical system."""
from deluca.core import Env
from deluca.core import Disturbance
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


class ZeroDisturbance(Disturbance):
    def init(self, d):
        self.d = d

    def __call__(self, t, key):
        del t
        del key
        return jnp.zeros((self.d, 1))


class GaussianDisturbance(Disturbance):
    def init(self, d, std=0.5):
        self.d = d
        self.std = std

    def __call__(self, t, key):
        del t
        return jax.random.normal(key, (self.d, 1)) * self.std


class SinusDisturbance(Disturbance):
    def init(self, d, amplitude=0.5):
        self.d = d
        self.amplitude = amplitude
        key = jax.random.PRNGKey(0)
        self.phases = jax.random.normal(key, (self.d, 1))

    def __call__(self, t, key):
        return (jax.numpy.sin(t * self.phases)) * self.amplitude


class LDS(Env):
    """LDS."""

    def init(
        self,
        d_in=1,
        d_hidden=25,
        d_out=1,
        A=None,
        B=None,
        C=None,
        key=jax.random.PRNGKey(0),
        x0=None,
        disturbance=None,
    ):
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        self.A = (
            jnp.array(A)
            if A is not None
            else jnp.diag(
                jax.random.uniform(subkey1, (d_hidden,), minval=0.9, maxval=0.95)
            )
        )
        self.B = (
            jnp.array(B)
            if B is not None
            else jax.random.normal(subkey2, (d_hidden, d_in))
        )
        self.C = (
            jnp.array(C)
            if C is not None
            else jax.random.normal(subkey3, (d_out, d_hidden))
        )
        self.d = self.A.shape[0]
        self.n = self.B.shape[1]
        self.p = self.C.shape[0]
        self.x = jnp.zeros((self.d, 1)) if x0 is None else jnp.array(x0)
        self.t = 0
        if disturbance is None:
            self.disturbance = GaussianDisturbance()
        else:
            self.disturbance = disturbance
        self.disturbance.init(self.d)
        self.key = key

        return self.x, self.C @ self.x

    def __call__(self, _state, u, key=None):
        if key is None:
            self.key, subkey = jax.random.split(self.key, 2)
            w_t = self.disturbance(self.t, subkey)
        else:
            w_t = self.disturbance(self.t, key)
        self.x = self.A @ self.x + self.B @ u + w_t
        y = self.C @ self.x
        self.t += 1

        return self.x, y 

    @property
    def action_size(self) -> int:
        return self.n
    
    def reset(self, x0=None):
        self.x = jnp.zeros((self.d, 1)) if x0 is None else jnp.array(x0)
        self.t = 0

        return self.x, self.C @ self.x

    def generate_random_trajectory(self, trajectory_length=1000, key=None):
        """generate_random_trajectory.
        generates a trajectory of the environment with random actions

        Args: length of trajectory to generate

        Returns: random trajectory

        """
        print("trajectory length =", trajectory_length)

        if key is None:
            key = jax.random.PRNGKey(0)

        results = jnp.zeros((trajectory_length,))
        for i in range(trajectory_length):
            key, subkey1, subkey2 = jax.random.split(key, 3)
            rand_action = jax.random.normal(subkey1, (self.n, 1))
            state, obs = self(rand_action, subkey2)
            results = results.at[i].set(
                obs[0, 0]
            )  # first coordinate of the observation

        return results

    def show_me_the_signal(self, length=1000, key=None):
        results = self.generate_random_trajectory(length, key)
        plt.plot(np.array(results))
        plt.show()

    def diagnostics(self):
        print("my parameters are")
        print("d_in (n), d_hidden (d), d_out (p) are:")
        print(self.n, self.d, self.p)
        print("A, B, C are: ")
        print(self.A, self.B, self.C)
        print("state is:")
        print(self.x)
