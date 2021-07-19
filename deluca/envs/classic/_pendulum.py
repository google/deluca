# Copyright 2021 Google LLC
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
import jax.numpy as jnp

from deluca.core import Env
from deluca.core import field
from deluca.core import Obj


class PendulumState(Obj):
    arr: jnp.ndarray = field(jaxed=True)
    h: int = field(0, jaxed=True)


class Pendulum(Env):
    m: float = field(1.0, jaxed=False)
    l: float = field(1.0, jaxed=False)
    g: float = field(9.81, jaxed=False)
    max_torque: float = field(1.0, jaxed=False)
    dt: float = field(0.02, jaxed=False)
    H: int = field(300, jaxed=False)
    goal_state: jnp.ndarray = field(jnp.array([0.0, -1.0, 0.0]), jaxed=False)

    def init(self):
        return PendulumState(arr=jnp.array([0.0, 1.0, 0.0]))

    def __call__(self, state, action):
        sin, cos, thdot = state.arr
        action = self.max_torque * jnp.tanh(action[0])
        th = jnp.arctan2(sin, cos)
        newthdot = th + (
            -3.0 * self.g / (2.0 * self.l) * jnp.sin(th + jnp.pi)
            + 3.0 / (self.m * self.l ** 2) * action
        )
        newth = th + newthdot * self.dt
        newsin, newcos = jnp.sin(newth), jnp.cos(newth)
        arr = jnp.array([newsin, newcos, newthdot])
        return PendulumState(arr=arr, h=state.h + 1), arr
