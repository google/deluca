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

from typing import Callable

import jax
import jax.numpy as jnp

from deluca.core import Agent
from deluca.core import field


class Random(Agent):
  seed: int = field(0, jaxed=False)
  func: Callable = field(lambda key: 0.0, jaxed=False)

  def init(self):
    return jax.random.PRNGKey(self.seed)  # Old API

  def __call__(self, state, obs):
    key, subkey = jax.random.split(state)  # Old API
    return subkey, self.func(key)


class SimpleRandom(Agent):
  """SimpleRandom.
  This agent return a normally distributed action.
  """
#  d_action: int = field(1, jaxed=False)
#  agent_state: jnp.array = field(default_factory=lambda: jnp.array([[1.0]]), jaxed=False)
#  key: int = field(default_factory=lambda: jax.random.key(0), jaxed=False)

  def __init__(self, n: int):
    self.n = n

  def __call__(self, obs: jnp.array, key: jax.random.key):
    return jax.random.normal(key, shape = (self.n, 1))