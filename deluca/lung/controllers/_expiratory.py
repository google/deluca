# Copyright 2021 The Deluca Authors.
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
import numpy as np
import jax
import jax.numpy as jnp
import deluca
import deluca.core
from deluca.lung.core import Controller
from deluca.lung.core import BreathWaveform
from deluca.lung.core import Phase
from deluca.lung.core import DEFAULT_DT
from deluca.lung.core import proper_time


class ExpiratoryState(deluca.Obj):
  waveform: BreathWaveform = deluca.field(jaxed=False)

class Expiratory(Controller):

  def init(self, waveform=None):
    if waveform is None:
      waveform = BreathWaveform.create()
    state = ExpiratoryState(waveform=waveform)
    return state

  @jax.jit
  def __call__(self, state, obs, *args, **kwargs):
    pressure, time = obs.pressure, obs.time
    phase = state.waveform.phase(time)
    u_out = jnp.zeros_like(phase)
    u_out = jax.lax.cond(
        jax.numpy.logical_or(
            jnp.equal(phase, Phase.RAMP_DOWN.value),
            jnp.equal(phase, Phase.PEEP.value)), lambda x: 1, lambda x: x,
        u_out)

    return state, u_out
