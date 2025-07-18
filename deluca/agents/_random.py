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

import time
from typing import Callable

import jax
import jax.numpy as jnp

import numpy as np

from deluca.core import Agent
from deluca.core import field


class Random(Agent):
    seed: int = field(0, jaxed=False)
    func: Callable = field(lambda key: 0.0, jaxed=False)

    def init(self):
        return jax.random.key(self.seed)

    def __call__(self, state, obs):
        key, subkey = jax.random.split(state)
        return subkey, self.func(key)


class SimpleRandom(Agent):
    """SimpleRandom.
    This agent return a normally distributed action.
    """

    def __init__(self, n: int, key=jax.random.key(0)):
        self.n = n
        self.key = key

    def __call__(self, _obs: jnp.ndarray):
        self.key, subkey = jax.random.split(self.key)
        return jax.random.normal(subkey, shape=(self.n, 1))

    def update(self, obs: jnp.ndarray, action: jnp.ndarray) -> None:
        return None