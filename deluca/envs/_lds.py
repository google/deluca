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


class LDS(Env):
  """LDS."""

  def init(self, A, B, C, x0=None, disturbance=None):
    self.A = jnp.array(A)
    self.B = jnp.array(B)
    self.C = jnp.array(C)
    self.d = A.shape[0]
    self.n = B.shape[1]
    self.p = C.shape[0]
    self.x = jnp.zeros((self.d, 1)) if x0 is None else jnp.array(x0)
    self.t = 0
    self.disturbance = disturbance or ZeroDisturbance(self.d)
    self.history = {"x": [self.x], "u": [], "w": [], "y": []}

  def __call__(self, u_t, key):
    w_t = self.disturbance(self.t, key)
    x_next = self.A @ self.x + self.B @ u_t + w_t
    y_t = self.C @ self.x
    self.history["x"].append(x_next)
    self.history["u"].append(u_t)
    self.history["w"].append(w_t)
    self.history["y"].append(y_t)
    self.x = x_next
    self.t += 1
    return y_t


  def generate_random_trajectory(self, trajectory_length = 1000, key=None):
    """generate_random_trajectory.
    generates a trajectory of the environment with random actions

    Args: length of trajectory to generate

    Returns: random trajectory

    """
    print("trajectory length =" , trajectory_length)

    if key is None:
      key = jax.random.PRNGKey(0)

    results = jnp.zeros((trajectory_length,))
    for i in range(trajectory_length):
      key, subkey1, subkey2 = jax.random.split(key, 3)
      rand_action = jax.random.normal(subkey1, (self.n, 1))
      obs = self(rand_action, subkey2)
      results = results.at[i].set(obs[0, 0]) # first coordinate of the observation
    
    return results

  def show_me_the_signal(self,length = 1000):
    results = self.generate_random_trajectory(length)
    plt.plot(np.array(results))
    plt.show()

  def diagnostics(self):
    print("my parameters are")
    print("d_in (n), d_hidden (d), d_out (p) are:")
    print(self.n, self.d, self.p)
    print("A, B, C are: ")
    print(self.A, self.B, self.C)
    print("state is:")
    print(self.state)
