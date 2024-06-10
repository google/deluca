# Copyright 2024 The Deluca Authors.
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

"""A constant controller."""

import jax.numpy as jnp
from optax._src import base


def constant(
    control: float | jnp.array,
) -> base.Agent:
  """A controller that gives a constant response.

  Args:
    control: The control to give.

  Returns:
    A constant agent.
  """

  def init_fn():
    return None

  def control_fn(
      state: base.AgentState, obs: base.EnvironmentState
  ) -> tuple[base.AgentState, float | jnp.array]:
    del obs
    return state, control

  def update_fn(state: base.AgentState) -> base.AgentState:
    return state

  return base.Agent(init_fn, control_fn, update_fn)
