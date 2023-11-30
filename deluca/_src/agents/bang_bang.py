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

"""A bang-bang or on-off agent."""

import jax.numpy as jnp
from optax._src import base


BangBangState = base.EmptyAgentState


def bang_bang(
    target: float | jnp.array,
    min_action: float | jnp.array,
    max_action: float | jnp.array,
) -> base.Agent:
  """An on-off agent.

  NOTE: `target`, `min_action`, `max_action`, and `obs` must be the same shape.

  Args:
    target: The target value of the agent.
    min_action: The minimum or "off" action.
    max_action: The maximum or "on" action.

  Returns:
    A Bang Bang agent.
  """

  def init_fn():
    return BangBangState()

  def action_fn(
      state: BangBangState, obs: base.EnvironmentState
  ) -> tuple[BangBangState, float | jnp.array]:
    return state, jnp.where(obs > target, min_action, max_action)

  def update_fn(state: BangBangState) -> BangBangState:
    return state

  return base.Agent(init_fn, action_fn, update_fn)
