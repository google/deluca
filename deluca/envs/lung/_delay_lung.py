# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import jax
import jax.numpy as jnp

from deluca.envs.lung import BreathWaveform
from deluca.envs.lung.core import Lung


class DelayLung(Lung):
    def __init__(
        self,
        min_volume=1.5,
        R_lung=10,
        C_lung=6,
        delay=25,
        inertia=0.995,
        control_gain=0.02,
        dt=0.03,
        waveform=None,
        reward_fn=None,
    ):
        self.viewer = None
        self.delay = delay
        self.control_gain = control_gain
        self.inertia = inertia
        self.dt = dt
        self.C_lung = C_lung
        self.R_lung = R_lung
        self.min_volume = min_volume
        self.waveform = waveform or BreathWaveform()
        self.r0 = (3 * self.min_volume / (4 * jnp.pi)) ** (1 / 3)

        self.reset()

    def reset(self):
        self.volume = self.min_volume
        self.pipe_pressure = 0
        self.pressure = 0
        self.time = 0.0
        self.target = self.waveform.at(self.time)
        self.in_history = jnp.zeros(self.delay)
        self.out_history = jnp.zeros(self.delay)

        self.state = {
            "volume": self.volume,
            "pressure": self.pressure,
            "pipe_pressure": self.pipe_pressure,
        }
        return self.observation

    @property
    def observation(self):
        return {
            "measured": self.state["pressure"],
            "target": self.target,
            "dt": self.dt,
            "phase": self.waveform.phase(self.time),
        }

    def dynamics(self, state, action):
        """
        state: (volume, pressure)
        action: (u_in, u_out)
        """
        flow = self.state["pressure"] / self.R_lung
        volume = self.state["volume"] + flow * self.dt
        volume = jnp.maximum(volume, self.min_volume)

        r = (3.0 * self.volume / (4.0 * jnp.pi)) ** (1.0 / 3.0)
        lung_pressure = self.C_lung * (1 - (self.r0 / r) ** 6) / (self.r0 ** 2 * r)

        pipe_impulse = jax.lax.cond(
            self.time < self.delay,
            lambda x: 0.0,
            lambda x: self.control_gain * self.in_history[0],
            None,
        )
        peep = jax.lax.cond(
            self.time < self.delay, lambda x: 0.0, lambda x: self.out_history[0], None
        )

        pipe_pressure = self.inertia * state["pipe_pressure"] + pipe_impulse
        pressure = jnp.maximum(0, pipe_pressure - lung_pressure)

        pipe_pressure = jax.lax.cond(peep, lambda x: x * 0.995, lambda x: x, pipe_pressure)

        return {"volume": volume, "pressure": pressure, "pipe_pressure": pipe_pressure}

    def step(self, action):
        u_in, u_out = action
        u_in = jax.lax.cond(u_in > 0.0, lambda x: x, lambda x: 0.0, u_in)

        self.in_history = jnp.roll(self.in_history, shift=1)
        self.in_history = self.in_history.at[0].set(u_in)
        self.out_history = jnp.roll(self.out_history, shift=1)
        self.out_history = self.out_history.at[0].set(u_out)

        self.target = self.waveform.at(self.time)
        reward = -jnp.abs(self.target - self.state["pressure"])

        self.state = self.dynamics(self.state, action)
        self.time += 1

        return self.observation, reward, False, {}
