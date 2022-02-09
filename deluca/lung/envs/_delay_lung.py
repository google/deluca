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

"""Delay Lung."""
import deluca.core
from deluca.lung.core import BreathWaveform
from deluca.lung.core import LungEnv
import jax
import jax.numpy as jnp


class Observation(deluca.Obj):
  predicted_pressure: float = 0.0
  time: float = 0.0


class SimulatorState(deluca.Obj):
  in_history: jnp.array
  out_history: jnp.array
  steps: int = 0
  time: float = 0.0
  volume: float = 0.0
  predicted_pressure: float = 0.0
  pipe_pressure: float = 0.0
  target: float = 0.0


class DelayLung(LungEnv):
  """Delay lung."""
  min_volume: float = deluca.field(1.5, jaxed=False)
  R: int = deluca.field(10, jaxed=False)
  C: int = deluca.field(6, jaxed=False)
  delay: int = deluca.field(25, jaxed=False)
  inertia: float = deluca.field(0.995, jaxed=False)
  control_gain: float = deluca.field(0.02, jaxed=False)
  dt: float = deluca.field(0.03, jaxed=False)
  waveform: BreathWaveform = deluca.field(jaxed=False)
  r0: float = deluca.field(jaxed=False)
  flow: float = deluca.field(0.0, jaxed=False)  # Question: initial flow value?

  def setup(self):
    if self.waveform is None:
      self.waveform = BreathWaveform.create()
    self.r0 = (3 * self.min_volume / (4 * jnp.pi))**(1 / 3)

  def reset(self):
    state = SimulatorState(
        steps=0,
        time=0.0,
        volume=self.min_volume,
        predicted_pressure=0.0,
        pipe_pressure=0.0,
        target=self.waveform.at(0.0),
        in_history=jnp.zeros(self.delay),
        out_history=jnp.zeros(self.delay))
    observation = Observation(
        predicted_pressure=0.0, time=0.0)
    return state, observation

  def dynamics(self, state, action):
    """dynamics function.

    Args:
      state: (volume, pressure)
      action: (u_in, u_out)
    Returns:
      state: next state
    """
    flow = state.predicted_pressure / self.R
    volume = state.volume + flow * self.dt
    volume = jnp.maximum(volume, self.min_volume)

    r = (3.0 * volume / (4.0 * jnp.pi))**(1.0 / 3.0)
    lung_pressure = self.C * (1 - (self.r0 / r)**6) / (self.r0**2 * r)

    pipe_impulse = jax.lax.cond(
        state.time < self.delay,
        lambda x: 0.0,
        lambda x: self.control_gain * state.in_history[0],
        None,
    )
    peep = jax.lax.cond(state.time < self.delay, lambda x: 0.0,
                        lambda x: state.out_history[0], None)

    pipe_pressure = self.inertia * state.pipe_pressure + pipe_impulse
    pressure = jnp.maximum(0, pipe_pressure - lung_pressure)

    pipe_pressure = jax.lax.cond(peep, lambda x: x * 0.995, lambda x: x,
                                 pipe_pressure)

    state = state.replace(
        steps=state.time + 1,
        time=state.time + self.dt,
        volume=volume,
        predicted_pressure=pressure,
        pipe_pressure=pipe_pressure,
        target=self.waveform.at(state.time + self.dt))
    return state

  def __call__(self, state, action):
    u_in, u_out = action
    u_in = jax.lax.cond(u_in.squeeze() > 0.0,
                        lambda x: x,
                        lambda x: 0.0,
                        u_in.squeeze())

    next_in_history = jnp.roll(state.in_history, shift=1)
    next_in_history = next_in_history.at[0].set(u_in)
    next_out_history = jnp.roll(state.out_history, shift=1)
    next_out_history = next_out_history.at[0].set(u_out)

    state = state.replace(in_history=next_in_history,
                          out_history=next_out_history)

    next_state = self.dynamics(state, action)
    observation = Observation(
        predicted_pressure=state.predicted_pressure, time=state.time)

    return next_state, observation
