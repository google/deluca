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
  key: int = field(default_factory=lambda: jax.random.key(0), jaxed=False)
  A: jnp.array = field(default_factory=lambda: jnp.array([[1.0]]), jaxed=False)
  B: jnp.array = field(default_factory=lambda: jnp.array([[1.0]]), jaxed=False)
  C: jnp.array = field(default_factory=lambda: jnp.array([[1.0]]), jaxed=False)
  state: jnp.array = field(default_factory=lambda: jnp.array([[1.0]]), jaxed=False)
  d_hidden: int = field(1, jaxed=False)
  d_in: int = field(1, jaxed=False)
  d_out: int = field(1, jaxed=False)


  def init(self,d_in = 1,d_hidden  = 1, d_out = 1):
    """init.
    initialize internal state to be random

    Returns:
      obs:
    """
    self.d_in = d_in
    self.d_hidden = d_hidden
    self.d_out = d_out
    A = jnp.diag( jnp.sign( jax.random.normal(self.key, shape=(self.d_hidden))) * 0.9  + jax.random.uniform(self.key, self.d_hidden ) * 0.04  )
    print("the eigenvalues of our system:",jnp.diag(A))
    B = jax.random.normal(self.key, shape=(self.d_hidden, self.d_in))
    C = jax.random.normal(self.key, shape=(d_out, d_hidden)) # jax.numpy.identity(self.d_hidden) #
    self.A = A
    self.B = B
    self.C = C
    self.state = jax.random.normal(self.key, shape=(self.d_hidden, 1))
    return self.C @ self.state


  def __call__(self, action):
    """__call__.

    Args:
      action:

    Returns:
      observation signal:

    """
    assert action.shape[0] == self.d_in , "dimension of action is wrong"
    self.state = self.A @ self.state + self.B @ action
    return self.C @ self.state


  def generate_random_trajectory(self, trajectory_length = 1000):
    """generate_random_trajectory.
    generates a trajectory of the environment with random actions

    Args: length of trajectory to generate

    Returns: random trajectory

    """
    print("trajectory length =" , trajectory_length)
    results = np.zeros(trajectory_length)
    for i in range(trajectory_length):
      # rand_action = jax.random.normal(self.key, shape = (self.d_in,1))
      rand_action = np.random.normal(0,1,size=(self.d_in,1))
      obs = self( rand_action )
      results[i] = obs[0,0] # first coordinate of the observation
    return results

  def show_me_the_signal(self,length = 1000):
    results = self.generate_random_trajectory(length)
    plt.plot(results)
    plt.show()

  def diagnostics(self):
    print("my parameters are")
    print("d_in, d_hidden, d_out are:")
    print(self.d_in, self.d_hidden, self.d_out)
    print("A, B, C are: ")
    print(self.A, self.B, self.C)
    print("state is:")
    print(self.state)
