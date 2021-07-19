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
import inspect

import dill as pickle
import jax.numpy as jnp

from deluca.envs.lung import BreathWaveform
from deluca.envs.lung.core import Lung


class LearnedLung(Lung):
    def __init__(
        self,
        weights,
        pressure_mean=0.0,
        pressure_std=1.0,
        PEEP=5,
        input_dim=3,
        history_len=5,
        dt=0.03,
        waveform=None,
        reward_fn=None,
    ):
        self.viewer = None
        self.weights = weights
        self.pressure_mean = pressure_mean
        self.pressure_std = pressure_std
        self.input_dim = input_dim
        self.history_len = history_len
        self.dt = dt
        self.PEEP = PEEP
        self.waveform = waveform or BreathWaveform()
        self.reset()

    def reset(self):
        # pressure
        self.pressure = 0
        self.time = 0

        # NOTE: we don't combine this into a single trajectory because we must
        # roll these arrays at different times (e.g., u_ins, u_outs before
        # running through model, normalized pressures after)
        u_ins = jnp.zeros(self.history_len)
        u_outs = jnp.ones(self.history_len)
        normalized_pressures = (
            jnp.ones(self.history_len) * (self.PEEP - self.pressure_mean) / self.pressure_std
        )
        self.state = {
            "u_ins": u_ins,
            "u_outs": u_outs,
            "normalized_pressures": normalized_pressures,
        }

        return self.observation

    @classmethod
    def from_torch(cls, path):
        torch_sim = pickle.load(open(path, "rb"))
        args = {
            key: val
            for key, val in torch_sim.items()
            if key in inspect.signature(LearnedLung.__init__).parameters.keys()
        }
        sim = cls(**args)

        weights = []
        for weight in sim.weights:
            weights.append(jnp.array(weight))

        sim.weights = weights

        return sim

    @property
    def observation(self):
        return {
            "measured": self.pressure,
            "target": self.waveform.at(self.time),
            "dt": self.dt,
            "phase": self.waveform.phase(self.time),
        }

    def dynamics(self, state, action):
        """
        pressure: (u_in, u_out, normalized pressure) histories
        action: (u_in, u_out)
        """

        u_ins, u_outs, normalized_pressures = (
            state["u_ins"],
            state["u_outs"],
            state["normalized_pressures"],
        )
        u_in, u_out = action

        u_in /= 50.0
        u_out = u_out * 2.0 - 1.0

        u_ins = jnp.roll(u_ins, shift=-1)
        u_ins = u_ins.at[-1].set(u_in)
        u_outs = jnp.roll(u_outs, shift=-1)
        u_outs = u_outs.at[-1].set(u_out)

        normalized_pressure = jnp.concatenate((u_ins, u_outs, normalized_pressures))
        for i in range(0, len(self.weights), 2):
            normalized_pressure = self.weights[i] @ normalized_pressure + self.weights[i + 1]
            if i <= len(self.weights) - 4:
                normalized_pressure = jnp.tanh(normalized_pressure)

        normalized_pressures = jnp.roll(normalized_pressures, shift=-1)
        normalized_pressures = normalized_pressures.at[-1].set(normalized_pressure.squeeze())

        return {"u_ins": u_ins, "u_outs": u_outs, "normalized_pressures": normalized_pressures}

    def step(self, action):
        self.state = self.dynamics(self.state, action)
        self.pressure = (
            self.state["normalized_pressures"][-1] * self.pressure_std
        ) + self.pressure_mean
        self.pressure = jnp.clip(self.pressure, 0.0, 100.0)

        self.target = self.waveform.at(self.time)
        reward = -jnp.abs(self.target - self.pressure)

        self.time += self.dt

        return self.observation, reward, False, {}
