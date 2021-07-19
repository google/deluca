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


def PropValve(x):
    y = 3.0 * x
    flow_new = 1.0 * (jnp.tanh(0.03 * (y - 130)) + 1.0)
    flow_new = jnp.clip(flow_new, 0.0, 1.72)
    return flow_new


def Solenoid(x):
    return x > 0


class BalloonLung(Lung):
    """
    Lung simulator based on the two-balloon experiment

    Source: https://en.wikipedia.org/wiki/Two-balloon_experiment

    TODO:
    - time / dt
    - dynamics
    - waveform
    - phase
    """

    def __init__(
        self,
        leak=False,
        peep_valve=5.0,
        PC=40.0,
        P0=0.0,
        C=10.0,
        R=15.0,
        dt=0.03,
        waveform=None,
        reward_fn=None,
    ):
        self.viewer = None
        # dynamics hyperparameters
        self.min_volume = 1.5
        self.C = C
        self.R = R
        self.PC = PC
        self.P0 = 0.0
        self.leak = leak
        self.peep_valve = peep_valve
        self.time = 0.0
        self.dt = dt

        self.waveform = waveform or BreathWaveform()

        self.r0 = (3.0 * self.min_volume / (4.0 * jnp.pi)) ** (1.0 / 3.0)

        # reset states
        self.reset()

    def reset(self):
        self.time = 0.0
        self.target = self.waveform.at(self.time)
        self.state = self.dynamics({"volume": self.min_volume, "pressure": -1.0}, (0.0, 0.0))
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
        volume, pressure = state["volume"], state["pressure"]
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
            lambda x: (self.dt / (5.0 + self.dt) * (self.min_volume - volume)),
            lambda x: 0.0,
            0.0,
        )

        r = (3.0 * volume / (4.0 * jnp.pi)) ** (1.0 / 3.0)
        pressure = self.P0 + self.PC * (1.0 - (self.r0 / r) ** 6.0) / (self.r0 ** 2.0 * r)
        # pressure = flow * self.R + volume / self.C + self.peep_valve

        return {"volume": volume, "pressure": pressure}

    def step(self, action):
        self.target = self.waveform.at(self.time)
        reward = -jnp.abs(self.target - self.state["pressure"])
        self.state = self.dynamics(self.state, action)

        self.time += self.dt

        return self.observation, reward, False, {}
