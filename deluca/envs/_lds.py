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
  A: jnp.array = field(default_factory=lambda: jnp.array([[1.0]]), jaxed=False)
  B: jnp.array = field(default_factory=lambda: jnp.array([[1.0]]), jaxed=False)
  C: jnp.array = field(default_factory=lambda: jnp.array([[1.0]]), jaxed=False)
  key: int = field(default_factory=lambda: jax.random.key(0), jaxed=False)
  # state_size: int = field(1, jaxed=False)
  action_size: int = field(1, jaxed=False)
  d_hidden: int = field(1, jaxed=False)
  d_in: int = field(1, jaxed=False)
  d_out: int = field(1, jaxed=False)

  def init(self,d_in = 1,d_hidden  = 1, d_out = 1):
    """init.

    Returns:

    """
    key1, key2, key3, key4 = jax.random.split(self.key, 4) 
    state = jax.random.normal(key4, shape=(self.d_hidden, 1))
    A = jnp.diag(jnp.sign(jax.random.normal(key1, shape=(d_hidden,))) * 0.9 + jax.random.uniform(key1, shape=(d_hidden,)) * 0.04)
    B = jax.random.normal(key2, shape=(d_hidden, d_in))
    C = jax.random.normal(key3, shape=(d_out, d_hidden))

    self.unfreeze()
    self.A = A
    self.B = B
    self.C = C
    self.freeze()

    return state, state

  def __call__(self, state, action):
    """__call__.

    Args:
      state:
      action:

    Returns:

    """
    new_state = self.A @ state + self.B @ action

    return new_state, self.C @ new_state



  def generate_random_trajectory(self, env_state, trajectory_length = 1000):
    results = np.zeros(trajectory_length)
    for i in range(trajectory_length):
      rand = np.random.normal(size = (1,))
      env_state, obs = self( env_state, rand)
      results[i] = obs.item()
    return results

  def show_me_the_signal(self,length = 1000):
    results = self.generate_random_trajectory(self.init()[0],length)
    plt.plot(results)
    plt.show()
