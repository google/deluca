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
from deluca.core import field
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


class LDS(Env):
  """LDS."""

  def init(self,A: jnp.array = None, B: jnp.array = None, C: jnp.array = None, d_in: int = 1, d_hidden: int = 1, d_out: int = 1, disturbance: str = "gaussian", disturbance_std: float = 0.04, key: jax.random.key = jax.random.PRNGKey(0)):
    """init.
    initialize internal state to be random

    Returns:
      obs:
    """
    self.A = A or jnp.diag( jnp.sign( jax.random.normal(key, shape=(d_hidden))) * 0.9  + jax.random.uniform(key, shape=(d_hidden) ) * 0.04  )
    self.B = B or jax.random.normal(key, shape=(d_hidden, d_in))
    self.C = C or jax.random.normal(key, shape=(d_out, d_hidden))
    self.disturbance = disturbance
    self.disturbance_std = disturbance_std
    self.d = self.A.shape[0]
    self.n = self.B.shape[1]
    self.p = self.C.shape[0]
    self.state = jnp.zeros((self.d, 1))


  def __call__(self, action, key):
    """__call__.

    Args:
      action:

    Returns:
      observation signal:

    """
    assert action.shape == (self.n, 1), "dimension of action is wrong"
    if self.disturbance == "gaussian":
      disturbance = jax.random.normal(key, shape=(self.d, 1)) * self.disturbance_std
    self.state = self.A @ self.state + self.B @ action + disturbance
    return self.C @ self.state


  def generate_random_trajectory(self, key, trajectory_length = 1000):
    """generate_random_trajectory.
    generates a trajectory of the environment with random actions

    Args: length of trajectory to generate

    Returns: random trajectory

    """
    print("trajectory length =" , trajectory_length)
    results = np.zeros(trajectory_length)
    for i in range(trajectory_length):
      rand_action = np.random.normal(0, 1, size=(self.n, 1))
      obs = self(rand_action,key)
      results[i] = obs[0,0] # first coordinate of the observation
    return results

  def show_me_the_signal(self,length = 1000):
    key = jax.random.PRNGKey(0)
    results = self.generate_random_trajectory(key,length)
    plt.plot(results)
    plt.show()

  def diagnostics(self):
    print("my parameters are")
    print("d_in, d_hidden, d_out are:")
    print(self.n, self.d, self.p)
    print("A, B, C are: ")
    print(self.A, self.B, self.C)
    print("state is:")
    print(self.state)
