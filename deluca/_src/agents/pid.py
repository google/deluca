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

"""Vanilla PID agent."""

import jax.numpy as jnp
from optax._src import base


class PIDState(base.AgentState):
  P: float
  I: float
  D: float


def pid(
    kp: float,
    ki: float,
    kd: float,
    dt: float,
    rc: float = 0.0,
) -> base.Agent:
  """A PID agent.

  Args:
    kp: P constant.
    ki: I constant.
    kd: D constant.
    dt: Time increment.
    rc: Delay constant.

  Returns:
    A constant agent.
  """

  decay = dt / (dt + rc)

  def init_fn():
    return PIDState()

  def action_fn(
      state: PIDState, obs: base.EnvironmentState
  ) -> tuple[PIDState, float | jnp.array]:
    p = obs
    i = state.I + decay * (obs - state.I)
    d = state.D + decay * (obs - state.P - state.D)

    action = kp * p + ki * i + kd * d
    return state.replace(P=p, I=i, D=d), action

  def update_fn(state: PIDState) -> PIDState:
    return state

  return base.Agent(init_fn, action_fn, update_fn)
