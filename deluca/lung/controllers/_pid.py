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

from deluca.lung.core import Controller, ControllerState
from deluca.lung.core import BreathWaveform
from deluca.lung.core import DEFAULT_DT
import deluca.core
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from collections.abc import Callable


class PIDControllerState(deluca.Obj):
  waveform: deluca.Obj
  P: float = 0.
  I: float = 0.
  D: float = 0.
  steps: int = 0



# generic PID controller
class PID(Controller):
  K: jnp.ndarray = deluca.field(jnp.array([1., 2., 0.]), jaxed=False)
  RC: float = deluca.field(0.5, jaxed=False)

  def init(self, waveform=None):
    if waveform is None:
      waveform = BreathWaveform.create()
    state = PIDControllerState(waveform=waveform)
    return state

  @jax.jit
  def __call__(self, controller_state, obs):
    pressure, t, dt = obs.predicted_pressure, obs.time, obs.dt
    waveform = controller_state.waveform
    target = waveform.at(t)
    err = jnp.array(target - pressure)

    decay = jnp.array(dt / (dt + self.RC))

    P, I, D = controller_state.P, controller_state.I, controller_state.D
    next_P = err
    next_I = I + decay * (err - I)
    next_D = D + decay * (err - P - D)
    controller_state = controller_state.replace(P=next_P, I=next_I, D=next_D)

    next_coef = jnp.array([next_P, next_I, next_D])
    u_in = jnp.dot(next_coef, self.K)
    u_in = jax.lax.clamp(0.0, u_in.astype(jnp.float32), 100.0)

    # update controller_state
    new_steps = controller_state.steps + 1
    controller_state = controller_state.replace(steps=new_steps)

    return controller_state, u_in
