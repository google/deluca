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


class LDS(Env):
  """LDS."""
  A: jnp.array = field(jaxed=False)
  B: jnp.array = field(jaxed=False)
  C: jnp.array = field(jaxed=False)
  key: int = field(jax.random.PRNGKey(0), jaxed=False)
  state_size: int = field(1, jaxed=False)
  action_size: int = field(1, jaxed=False)

  def init(self):
    """init.

    Returns:

    """
    state = jax.random.normal(self.key, shape=(self.state_size, 1))
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
