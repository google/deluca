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

"""PID controller."""
import collections.abc

import deluca.core
from deluca.lung.core import BreathWaveform
from deluca.lung.core import Controller
from deluca.lung.core import DEFAULT_DT
from deluca.lung.core import proper_time
import flax.linen as nn
import jax
import jax.numpy as jnp


# TODO(dsuo,alexjyu): refactor with deluca PID
class PIDControllerState(deluca.Obj):
  waveform: deluca.Obj
  p: float = 0.
  i: float = 0.
  d: float = 0.
  time: float = float("inf")
  steps: int = 0
  dt: float = DEFAULT_DT


class PIDNetwork(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=1, use_bias=False, name="K")(x)
    return x


# generic PID controller
class PID(Controller):
  """PID controller."""
  model: nn.module = deluca.field(PIDNetwork, jaxed=False)
  params: jnp.array = deluca.field(jaxed=True)  # jnp.array([3.0, 4.0, 0.0]
  RC: float = deluca.field(0.5, jaxed=False)
  dt: float = deluca.field(0.03, jaxed=False)
  model_apply: collections.abc.Callable = deluca.field(jaxed=False)

  def setup(self):
    self.model = PIDNetwork()
    if self.params is None:
      self.params = self.model.init(jax.random.PRNGKey(0), jnp.ones([
          3,
      ]))["params"]
    # TODO(dsuo): Handle dataclass initialization of jax objects
    self.model_apply = jax.jit(self.model.apply)

  def init(self, waveform=None):
    if waveform is None:
      waveform = BreathWaveform.create()
    state = PIDControllerState(waveform=waveform)
    return state

  @jax.jit
  def __call__(self, controller_state, obs):
    pressure, t = obs.predicted_pressure, obs.time
    waveform = controller_state.waveform
    target = waveform.at(t)
    err = jnp.array(target - pressure)
    decay = jnp.array(self.dt / (self.dt + self.RC))
    p, i, d = controller_state.p, controller_state.i, controller_state.d
    next_p = err
    next_i = i + decay * (err - i)
    next_d = d + decay * (err - p - d)
    controller_state = controller_state.replace(p=next_p, i=next_i, d=next_d)
    next_coef = jnp.array([next_p, next_i, next_d])
    u_in = self.model_apply({"params": self.params}, next_coef)
    u_in = jax.lax.clamp(0.0, u_in.astype(jnp.float64), 100.0)

    # update controller_state
    new_dt = jnp.max(
        jnp.array([DEFAULT_DT, t - proper_time(controller_state.time)]))
    new_time = t
    new_steps = controller_state.steps + 1
    controller_state = controller_state.replace(
        time=new_time, steps=new_steps, dt=new_dt)
    return controller_state, u_in
