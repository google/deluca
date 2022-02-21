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

from deluca.core import Agent
from deluca.core import field
from deluca.core import Obj


class PIDState(Obj):
    P: float = field(0.0, jaxed=True)
    I: float = field(0.0, jaxed=True)
    D: float = field(0.0, jaxed=True)


class PID(Agent):
    K_P: float = field(0.0, jaxed=True)
    K_I: float = field(0.0, jaxed=True)
    K_D: float = field(0.0, jaxed=True)
    RC: float = field(0.5, jaxed=False)
    dt: float = field(0.03, jaxed=False)
    decay: float = field(jaxed=False)

    def init(self):
        return PIDState()

    def setup(self):
        self.decay = self.dt / (self.dt + self.RC)

    def __call__(self, state, obs):
        """Assume observation is err"""
        P = obs
        I = state.I + self.decay * (obs - state.I)
        D = state.D + self.decay * (obs - state.P - state.D)

        action = self.K_P * P + self.K_I * I + self.K_D * D

        return state.replace(P=P, I=I, D=D), action
