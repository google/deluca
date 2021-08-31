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
import functools

import jax
import jax.numpy as jnp
import flax.linen as nn

import deluca.core
from deluca.lung.core import Controller, ControllerState
from deluca.lung.core import BreathWaveform
from deluca.lung.core import DEFAULT_DT
from deluca.lung.core import proper_time


class PIDControllerState(deluca.Obj):
  P: float = 0.
  I: float = 0.
  D: float = 0.
  time: float = 0.
  steps: int = 0
  dt: float = DEFAULT_DT


class PID_network(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=1, use_bias=False, name=f"K")(x)
    return x


# generic PID controller
class PID(Controller):
  model: nn.module = deluca.field(PID_network, jaxed=False)
  params: jnp.array = deluca.field(jaxed=True)  # jnp.array([3.0, 4.0, 0.0]
  waveform: deluca.Obj = deluca.field(jaxed=False)
  RC: float = deluca.field(0.5, jaxed=False)
  dt: float = deluca.field(0.03, jaxed=False)

  def setup(self):
    self.model = PID_network()
    if self.params is None:
      self.params = self.model.init(jax.random.PRNGKey(0), jnp.ones([
          3,
      ]))["params"]
    # TODO: Handle dataclass initialization of jax objects
    if self.waveform is None:
      self.waveform = BreathWaveform.create()

  def init(self):
    state = PIDControllerState()
    return state

  @functools.partial(jax.jit, static_argnums=(2,))
  def __call__(self, state, obs):
    pressure, t = obs.pressure, obs.time
    target = self.waveform.at(t)
    err = target - pressure

    decay = self.dt / (self.dt + self.RC)

    P, I, D = state.P, state.I, state.D
    next_P = err
    next_I = I + decay * (err - I)
    next_D = D + decay * (err - P - D)

    next_coef = jnp.array([next_P, next_I, next_D])
    u_in = jnp.dot(next_coef, self.params)
    u_in = jax.lax.clamp(0.0, u_in.astype(jnp.float32), 100.0)

    # update state
    new_dt = obs.dt
    new_time = t
    new_steps = state.steps + 1
    state = state.replace(P=next_P, I=next_I, D=next_D,
        time=new_time, steps=new_steps, dt=new_dt)

    return state, u_in
