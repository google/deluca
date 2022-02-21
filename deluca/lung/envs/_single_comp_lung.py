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

"""Single comp lung."""
import deluca.core
from deluca.lung.core import LungEnv
import jax.numpy as jnp


class Observation(deluca.Obj):
  predicted_pressure: float = 0.0
  time: float = 0.0


class SimulatorState(deluca.Obj):
  steps: int = 0
  volume: float = 0.0
  time: float = 0.0


def PropValve(x):
  y = 3.0 * x
  flow_new = 1.0 * (jnp.tanh(0.03 * (y - 130)) + 1.0)
  flow_new = jnp.clip(flow_new, 0.0, 1.72)
  return flow_new


class SingleCompLung(LungEnv):
  """Single comp lung."""
  R: int = deluca.field(15, jaxed=False)
  C: int = deluca.field(50, jaxed=False)
  min_volume: float = deluca.field(0.2, jaxed=False)
  peep_value: float = deluca.field(5.0, jaxed=False)
  flow: float = deluca.field(0.0, jaxed=False)
  P0: float = deluca.field(10.0, jaxed=False)
  dt: float = deluca.field(0.03, jaxed=False)

  def reset(self):
    vol = self.min_volume
    obs = (vol / self.C) + self.P0

    # state = SimulatorState(0, vol)
    state = SimulatorState(steps=0, volume=vol, time=0.0)

    # print(type(state.steps))
    return state, Observation(obs, 0.0)

  def __call__(self, state, action):
    u_in, _ = action
    # calculate flow from u_in
    flow = jnp.clip(PropValve(u_in), 0, 2)
    # update volume by flow rate
    next_volume = (state.volume + flow * self.dt).squeeze()
    obs = ((flow * self.R) + (state.volume / self.C) +
           self.P0).squeeze()

    state = state.replace(
        steps=state.steps + 1, volume=next_volume, time=state.time + self.dt)

    return state, Observation(predicted_pressure=obs, time=state.time)
