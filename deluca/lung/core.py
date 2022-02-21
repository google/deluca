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

"""Core."""
import functools
import os
from typing import Tuple

import deluca.core
import jax
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.realpath(os.path.join(__file__, "../..")))

DEFAULT_XP = [0.0, 1.0, 1.5, 3.0 - 1e-8, 3.0]
DEFAULT_DT = 0.03

# pylint: disable=protected-access
# pylint: disable=g-long-lambda


class BreathWaveform(deluca.Obj):
  r"""Waveform generator with shape |‾\_."""
  peep: float = deluca.field(5., jaxed=False)
  pip: float = deluca.field(35., jaxed=False)
  bpm: int = deluca.field(20, jaxed=False)
  fp: jnp.array = deluca.field(jaxed=False)
  xp: jnp.array = deluca.field(jaxed=False)
  in_bounds: Tuple[int, int] = deluca.field((0, 1), jaxed=False)
  ex_bounds: Tuple[int, int] = deluca.field((2, 4), jaxed=False)
  period: float = deluca.field(jaxed=False)
  dt: float = deluca.field(DEFAULT_DT, jaxed=False)
  dtype: jax._src.numpy.lax_numpy._ScalarMeta = deluca.field(
      jnp.float32, jaxed=False)

  def setup(self):
    if self.fp is None:
      self.fp = jnp.array([self.pip, self.pip, self.peep, self.peep, self.pip])
    if self.xp is None:
      self.xp = jnp.array(DEFAULT_XP)
    self.period = 60 / self.bpm

  def at(self, t):

    @functools.partial(jax.jit, static_argnums=(3,))
    def static_interp(t, xp, fp, period):
      return jnp.interp(t, xp, fp, period=period)

    return static_interp(t, self.xp, self.fp, self.period).astype(self.dtype)

  def elapsed(self, t):
    return t % self.period

  def is_in(self, t):
    return self.elapsed(t) <= self.xp[self.in_bounds[1]]

  def is_ex(self, t):
    return not self.is_in(t)


class LungEnv(deluca.Env):
  """Lung environment."""

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


# TODO(alexjyu): change predicted_pressure to pressure
class LungEnvState(deluca.Obj):
  t_in: int = 0
  steps: int = 0
  predicted_pressure: float = 0.0


class ControllerState(deluca.Obj):
  time: float = float("inf")
  steps: int = 0
  dt: float = DEFAULT_DT


def proper_time(t):
  return jax.lax.cond(t == float("inf"), lambda x: 0., lambda x: x, t)


class Controller(deluca.Agent):
  """Controller."""

  def init(self, waveform=None):
    if waveform is None:
      waveform = BreathWaveform.create()
    return ControllerState()

  @property
  def max(self):
    return 100

  @property
  def min(self):
    return 0

  def decay(self, waveform, t):
    elapsed = waveform.elapsed(t)

    def false_func():
      result = jax.lax.cond(
          elapsed < waveform.xp[waveform.ex_bounds[0]], lambda x: 0.0,
          lambda x: 5 * (1 - jnp.exp(5 * (waveform.xp[waveform.ex_bounds[
              0]] - elapsed))).astype(waveform.dtype), None)
      return result

    # float(inf) as substitute to None since cond requries same type output
    result = jax.lax.cond(elapsed < waveform.xp[waveform.in_bounds[1]],
                          lambda x: float("inf"), lambda x: false_func(), None)
    return result
