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

import deluca.core
from deluca.lung.controllers import PID
from deluca.lung.controllers._pid import PIDControllerState
from deluca.lung.core import Controller
from deluca.lung.core import DEFAULT_DT
from deluca.lung.core import proper_time
from deluca.lung.core import BreathWaveform
import flax.linen as fnn
import jax
import jax.numpy as jnp


class FeaturizerState(deluca.Obj):
  errs: jnp.array


class TriangleErrorFeaturizer(deluca.Obj):
  history_len: int
  coef: jnp.array = deluca.field(jaxed=False)

  def setup(self):
    self.coef = jnp.tril(jnp.ones((self.history_len, self.history_len)))
    self.coef /= jnp.expand_dims(jnp.arange(self.history_len, 0, -1), axis=0)

  def init(self):
    errs = jnp.array([0.0] * self.history_len)
    state = FeaturizerState(errs=errs)
    return state

  def __call__(self, featurizer_state, pressure, target, t, *args, **kwargs):
    errs = featurizer_state.errs
    errs = jnp.roll(errs, shift=-1)
    errs = errs.at[-1].set(target - pressure)
    featurizer_state = featurizer_state.replace(errs=errs)
    trajectory = jnp.expand_dims(errs, axis=(0, 1))
    return featurizer_state, trajectory @ self.coef


class LSTMControllerState(deluca.Obj):
  featurizer_state: FeaturizerState
  pid_state: PIDControllerState
  lstm_state: tuple
  time: float = float("inf")
  steps: int = 0
  dt: float = DEFAULT_DT


class LSTMnetwork(fnn.Module):
  num_layers: int = 1

  @fnn.compact
  def __call__(self, lstm_state, x):
    for _ in range(self.num_layers):
      lstm_state, x = fnn.LSTMCell()(lstm_state, x)
    x = jnp.squeeze(x, axis=0)
    x = fnn.Dense(features=1, use_bias=True, name="lstm_fc")(x)
    return lstm_state, x

  @staticmethod
  def initialize_carry(batch_size, hidden_size):
    # use dummy key since default state init fn is just zeros.
    return fnn.LSTMCell.initialize_carry(
        jax.random.PRNGKey(0), batch_size, hidden_size)


class LSTMWithPID(Controller):
  params: list = deluca.field(jaxed=True)
  history_len: int = deluca.field(jaxed=False)
  mult: float = deluca.field(jaxed=False)
  pid_K: list = deluca.field(jaxed=False)

  num_layers: int = deluca.field(1, jaxed=False)
  hidden_size: int = deluca.field(10, jaxed=False)
  lstm: fnn.module = deluca.field(LSTMnetwork, jaxed=False)
  batch_size: tuple = deluca.field((1, 1), jaxed=False)
  waveform: deluca.Obj = deluca.field(jaxed=False)
  pid_base: Controller = deluca.field(jaxed=False)
  featurizer: deluca.Obj = deluca.field(jaxed=False)

  def setup(self):
    self.featurizer = TriangleErrorFeaturizer.create(self.history_len)
    if self.waveform is None:
      self.waveform = BreathWaveform.create()
    pid_params = {"K": {"kernel": jnp.array(self.pid_K)}}
    self.pid_base = PID.create(waveform=self.waveform, params=pid_params)
    self.lstm = LSTMnetwork(num_layers=self.num_layers)
    init_controller_state = self.init()
    init_lstm_state = init_controller_state.lstm_state

    if self.params is None:
      self.params = self.lstm.init(
          jax.random.PRNGKey(0), init_lstm_state,
          jnp.expand_dims(jnp.ones([self.history_len]), axis=(0, 1)))["params"]

  def init(self):
    featurizer_state = self.featurizer.init()
    pid_state = self.pid_base.init()
    lstm_state = self.lstm.initialize_carry(self.batch_size, self.hidden_size)
    controller_state = LSTMControllerState(
        featurizer_state=featurizer_state,
        pid_state=pid_state,
        lstm_state=lstm_state)
    return controller_state

  def __call__(self, controller_state, obs):
    featurizer_state = controller_state.featurizer_state
    pid_state = controller_state.pid_state
    lstm_state = controller_state.lstm_state
    pressure, t = obs.predicted_pressure, obs.time
    target = self.waveform.at(t)
    pid_state, u_in_base = self.pid_base(pid_state, obs)
    featurizer_state, features = self.featurizer(featurizer_state, pressure,
                                                 target, t)
    # features = jax.lax.cond(len(features.shape) == 1,
    #                         lambda x: jnp.expand_dims(x, axis=0),
    #                         lambda x: x,
    #                         features)
    lstm_state, u_in_residual = self.lstm.apply({"params": self.params},
                                                lstm_state, features)
    u_in = u_in_base + u_in_residual * self.mult
    u_in = jax.lax.clamp(0.0, u_in.astype(jnp.float32), 100.0).squeeze()
    controller_state = controller_state.replace(
        featurizer_state=featurizer_state,
        pid_state=pid_state,
        lstm_state=lstm_state)
    # update controller_state
    new_dt = jnp.max(
        jnp.array([DEFAULT_DT, t - proper_time(controller_state.time)]))
    new_time = t
    new_steps = controller_state.steps + 1
    controller_state = controller_state.replace(
        time=new_time, steps=new_steps, dt=new_dt)
    return controller_state, u_in
