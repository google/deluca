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
from deluca.lung.core import proper_time
import deluca.core
import jax
import jax.numpy as jnp
import flax.linen as nn


class PIDControllerState(deluca.Obj):
  P: float = 0.
  I: float = 0.
  D: float = 0.
  time: float = float("inf")
  steps: int = 0
  dt: float = DEFAULT_DT


class PID_network(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=1, use_bias=False, name=f"K")(x)
    return x


# generic PID controller
class PID(Controller):
  model: nn.module = deluca.field(PID_network, trainable=False)
  params: jnp.array = deluca.field(trainable=True)  # jnp.array([3.0, 4.0, 0.0]
  waveform: deluca.Obj = deluca.field(BreathWaveform.create(), trainable=False)
  RC: float = deluca.field(0.5, trainable=False)
  dt: float = deluca.field(0.03, trainable=False)

  def setup(self):
    self.model = PID_network()
    if self.params is None:
      self.params = self.model.init(jax.random.PRNGKey(0), jnp.ones([
          3,
      ]))["params"]

  def init(self):
    state = PIDControllerState()
    return state

  def __call__(self, controller_state, obs):
    pressure, t = obs.predicted_pressure, obs.time
    target = self.waveform.at(t)
    err = jnp.array(target - pressure)

    decay = jnp.array(self.dt / (self.dt + self.RC))

    P, I, D = controller_state.P, controller_state.I, controller_state.D
    next_P = err
    next_I = I + decay * (err - I)
    next_D = D + decay * (err - P - D)
    controller_state = controller_state.replace(P=next_P, I=next_I, D=next_D)

    next_coef = jnp.array([next_P, next_I, next_D])
    u_in = self.model.apply({"params": self.params}, next_coef)
    u_in = jax.lax.clamp(0.0, u_in.astype(jnp.float64), 100.0)

    # update controller_state
    new_dt = jnp.max(
        jnp.array([DEFAULT_DT, t - proper_time(controller_state.time)]))
    new_time = t
    new_steps = controller_state.steps + 1
    controller_state = controller_state.replace(
        time=new_time, steps=new_steps, dt=new_dt)

    return controller_state, u_in
