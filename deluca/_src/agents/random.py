# Copyright 2023 The Deluca Authors.
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

"""A random agent."""

from typing import Callable

import jax
import jax.numpy as jnp
from optax._src import base


class RandomState(base.AgentState):
  key: jnp.ndarray


def random(
    key: jnp.ndarray,
    shape: tuple[int, ...],
    func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
) -> base.Agent:
  """An agent that outputs random actions.

  Args:
    key: A `jax` key.
    shape: The shape of the output actions.
    func: A function that takes a key and a shape and outputs random actions.

  Returns:
    A Bang Bang agent.
  """

  if func is None:
    func = jax.random.uniform

  def init_fn():
    return RandomState(key=key)

  def action_fn(
      state: RandomState, obs: base.EnvironmentState
  ) -> tuple[RandomState, float | jnp.array]:
    del obs
    return state, func(state.key, shape)

  def update_fn(state: RandomState) -> RandomState:
    return state.replace(key=jax.random.fold_in(key, shape))

  return base.Agent(init_fn, action_fn, update_fn)
