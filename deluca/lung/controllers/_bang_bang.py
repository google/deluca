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

"""bang bang controller."""
import deluca.core
from deluca.lung.core import BreathWaveform
from deluca.lung.core import Controller
from deluca.lung.core import DEFAULT_DT
from deluca.lung.core import proper_time
import jax
import jax.numpy as jnp


class BangBang(Controller):
  """bang bang controller."""
  waveform: BreathWaveform = deluca.field(jaxed=False)
  min_action: float = deluca.field(0.0, jaxed=False)
  max_action: float = deluca.field(100.0, jaxed=False)

  def setup(self):
    if self.waveform is None:
      self.waveform = BreathWaveform.create()

  def __call__(self, controller_state, obs):
    pressure, t = obs.predicted_pressure, obs.time
    target = self.waveform.at(t)
    action = jax.lax.cond(pressure < target, lambda x: self.max_action,
                          lambda x: self.min_action, None)
    # update controller_state
    new_dt = jnp.max(
        jnp.array([DEFAULT_DT, t - proper_time(controller_state.time)]))
    new_time = t
    new_steps = controller_state.steps + 1
    controller_state = controller_state.replace(
        time=new_time, steps=new_steps, dt=new_dt)
    return controller_state, action
