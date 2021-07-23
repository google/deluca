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

import os
from functools import partial
from enum import Enum
import jax
import jax.numpy as jnp
import deluca.core

ROOT = os.path.dirname(os.path.realpath(os.path.join(__file__, "../..")))

DEFAULT_DT = 0.03

DEFAULT_PRESSURE_RANGE = (5.0, 35.0)
DEFAULT_KEYPOINTS = [1e-8, 1.0, 1.5, 3.0]
DEFAULT_BPM = 20


class Phase(Enum):
  RAMP_UP = 1
  PIP = 2
  RAMP_DOWN = 3
  PEEP = 4


class BreathWaveform(deluca.Obj):
  """Waveform generator with shape |‾\_"""

  custom_range: tuple = deluca.field(trainable=False)
  lo: float = deluca.field(trainable=False)
  hi: float = deluca.field(trainable=False)
  fp: jnp.array = deluca.field(trainable=False)
  xp: jnp.array = deluca.field(trainable=False)
  dtype: jax._src.numpy.lax_numpy._ScalarMeta = deluca.field(
      jnp.float64, trainable=False)
  keypoints: jnp.array = deluca.field(trainable=False)
  bpm: int = deluca.field(DEFAULT_BPM, trainable=False)
  kernel: list = deluca.field(trainable=False)
  dt: float = deluca.field(0.01, trainable=False)
  period: float = deluca.field(trainable=False)

  def setup(self):
    self.lo, self.hi = self.custom_range or DEFAULT_PRESSURE_RANGE
    self.fp = jnp.array([self.lo, self.hi, self.hi, self.lo, self.lo])

    self.xp = jnp.zeros(len(self.fp), dtype=self.dtype)
    if self.keypoints:
      self.xp = jax.ops.index_update(
          self.xp, jax.ops.index[1:],
          jnp.array(self.keypoints, dtype=self.dtype))
    else:
      self.xp = jax.ops.index_update(
          self.xp, jax.ops.index[1:],
          jnp.array(DEFAULT_KEYPOINTS, dtype=self.dtype))
    self.xp = jax.ops.index_update(self.xp, -1, 60 / self.bpm)
    self.keypoints = self.xp
    self.keypoints = self.xp
    self.period = float(self.xp[-1])

    pad = 0
    num = int(1 / self.dt)
    if self.kernel is not None:
      pad = 60 / bpm / (num - 1)
      num += len(kernel) // 2 * 2

    tt = jnp.linspace(-pad, 60 / self.bpm + pad, num, dtype=self.dtype)
    self.fp = self.at(tt)
    self.xp = tt

    if self.kernel is not None:
      self.fp = jnp.convolve(self.fp, self.kernel, mode="valid")
      self.xp = jnp.linspace(0, 60 / bpm, int(1 / self.dt), dtype=dtype)

  @property
  def PIP(self):
    # TODO: what's custom range?
    if hasattr(self, "custom_range"):
      return self.custom_range[1]
    else:
      return jnp.max(self.fp)

  @property
  def PEEP(self):
    if hasattr(self, "custom_range"):
      return self.custom_range[0]
    else:
      return jnp.min(self.fp)

  def is_in(self, t):
    return self.elapsed(t) <= self.keypoints[2]

  def is_ex(self, t):
    return not self.is_in(t)

  def at(self, t):

    @partial(jax.jit, static_argnums=(3,))
    def static_interp(t, xp, fp, period):
      return jnp.interp(t, xp, fp, period=period)

    return static_interp(t, self.xp, self.fp, self.period).astype(self.dtype)

  def elapsed(self, t):
    return t % self.period

  # TODO: change all files where "if decay is None" to "if decay == float("inf")"
  # files affected: _mpc, _clipped_adv_deep, _clipped_deep, _deep_pid_residual,
  # _deep_pid_residual_clipped
  def decay(self, t):
    elapsed = self.elapsed(t)

    def false_func():
      result = jax.lax.cond(
          elapsed < self.keypoints[3], lambda x: 0.0, lambda x: 5 *
          (1 - jnp.exp(5 *
                       (self.keypoints[3] - elapsed))).astype(self.dtype), None)
      return result

    # float(inf) as substitute to None since cond requries same type output
    result = jax.lax.cond(elapsed < self.keypoints[2], lambda x: float("inf"),
                          lambda x: false_func(), None)
    return result

  def phase(self, t):
    return jnp.searchsorted(
        self.keypoints, jnp.mod(t, self.period), side="right")

  import os


class LungEnv(deluca.Env):
  # @property
  def time(self, state):
    return self.dt * state.steps

  @property
  def dt(self):
    return DEFAULT_DT

  @property
  def physical(self):
    return False

  def should_abort(self):
    return False

  def wait(self, duration):
    pass

  def cleanup(self):
    pass


class ControllerState(deluca.Obj):
  time: float = float("inf")
  steps: int = 0
  dt: float = DEFAULT_DT


def proper_time(t):
  return jax.lax.cond(t == float("inf"), lambda x: 0., lambda x: x, t)


class Controller(deluca.Agent):

  def init(self):
    return ControllerState()

  @property
  def max(self):
    return 100

  @property
  def min(self):
    return 0
