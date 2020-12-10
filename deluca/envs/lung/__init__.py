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
# TODO
# - interp smh
import jax.numpy as jnp

from deluca.core import JaxObject

DEFAULT_PRESSURE_RANGE = (5.0, 35.0)
DEFAULT_KEYPOINTS = [1e-8, 1.0, 1.5, 3.0]


class BreathWaveform(JaxObject):
    """Waveform generator with shape /‾\_"""

    def __init__(self, range=None, keypoints=None):
        self.lo, self.hi = range or DEFAULT_PRESSURE_RANGE
        self.xp = jnp.asarray([0] + (keypoints or DEFAULT_KEYPOINTS))
        self.fp = jnp.asarray([self.lo, self.hi, self.hi, self.lo, self.lo])
        self.period = self.xp[-1]

    def at(self, t):
        # return jnp.interp(t, self.xp, self.fp, period=self.period)
        return jnp.interp(t, self.xp, self.fp, period=3)

    def phase(self, t):
        return jnp.searchsorted(self.xp, t % self.period, side="right")


__all__ = ["BreathWaveform"]
