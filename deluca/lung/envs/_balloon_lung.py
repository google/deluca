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

"""Balloon Lung."""
import deluca.core
from deluca.lung.core import BreathWaveform
from deluca.lung.core import LungEnv
import jax
import jax.numpy as jnp


# pylint: disable=g-long-lambda


class Observation(deluca.Obj):
  predicted_pressure: float = 0.0
  time: float = 0.0


class SimulatorState(deluca.Obj):
  steps: int = 0
  time: float = 0.0
  volume: float = 0.0
  predicted_pressure: float = 0.0
  target: float = 0.0


def PropValve(x):
  y = 3.0 * x
  flow_new = 1.0 * (jnp.tanh(0.03 * (y - 130)) + 1.0)
  flow_new = jnp.clip(flow_new, 0.0, 1.72)
  return flow_new


def Solenoid(x):
  return x > 0


class BalloonLung(LungEnv):
  """Lung simulator based on the two-balloon experiment.

     Source: https://en.wikipedia.org/wiki/Two-balloon_experiment
     TODO:
      - time / dt
      - dynamics
      - waveform
      - phase
  """

  leak: bool = deluca.field(False, jaxed=False)
  peep_valve: float = deluca.field(5.0, jaxed=False)
  PC: float = deluca.field(40.0, jaxed=False)
  P0: float = deluca.field(0.0, jaxed=False)
  R: float = deluca.field(15.0, jaxed=False)
  C: float = deluca.field(10.0, jaxed=False)
  dt: float = deluca.field(0.03, jaxed=False)
  min_volume: float = deluca.field(1.5, jaxed=False)
  r0: float = deluca.field(jaxed=False)
  waveform: BreathWaveform = deluca.field(jaxed=False)
  # reward_fn: Callable = deluca.field(None, jaxed=False)
  reset_normalized_peep: float = deluca.field(0.0, jaxed=False)
  flow: float = deluca.field(0, jaxed=False)  # needed for run_controller

  def setup(self):
    if self.waveform is None:
      self.waveform = BreathWaveform.create()

    # dynamics hyperparameters
    # self.time = 0.0

    self.r0 = (3.0 * self.min_volume / (4.0 * jnp.pi))**(1.0 / 3.0)

    # reset states
    # self.reset()

  def reset(self):
    state = SimulatorState(
        steps=0,
        time=0.0,
        volume=self.min_volume,
        predicted_pressure=self.reset_normalized_peep,
        target=self.waveform.at(0.0))
    observation = Observation(
        predicted_pressure=self.reset_normalized_peep, time=0.0)
    return state, observation

  def __call__(self, state, action):
    """call function.

    Args:
      state: (steps, time, volume, predicted_pressure, target)
      action: (u_in, u_out)
    Returns:
      state: next state
      observation: next observation
    """
    volume, pressure = state.volume, state.predicted_pressure
    u_in, u_out = action

    flow = jnp.clip(PropValve(u_in) * self.R, 0.0, 2.0)
    flow -= jax.lax.cond(
        pressure > self.peep_valve,
        lambda x: jnp.clip(Solenoid(u_out), 0.0, 2.0) * 0.05 * pressure,
        lambda x: 0.0,
        flow,
    )

    volume += flow * self.dt
    volume += jax.lax.cond(
        self.leak,
        lambda x: ((self.dt / (5.0 + self.dt) *
                    (self.min_volume - volume))).squeeze(),
        lambda x: 0.0,
        0.0,
    )

    r = (3.0 * volume / (4.0 * jnp.pi))**(1.0 / 3.0)
    pressure = self.P0 + self.PC * (1.0 - (self.r0 / r)**6.0) / (
        self.r0**2.0 * r)
    # pressure = flow * self.R + volume / self.C + self.peep_valve

    state = state.replace(
        steps=state.time + 1,
        time=state.time + self.dt,
        volume=volume.squeeze(),
        predicted_pressure=pressure.squeeze(),
        target=self.waveform.at(state.time + self.dt))

    observation = Observation(
        predicted_pressure=state.predicted_pressure, time=state.time)
    return state, observation
