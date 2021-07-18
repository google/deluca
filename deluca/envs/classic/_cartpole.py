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


def AB(m, M, l, g, dt):
    return ()


class CartpoleState(Obj):
    arr: jnp.ndarray = field(trainable=True)
    h: int = field(0, trainable=True)
    offset: float = field(0.0, trainable=False)


class Cartpole(Env):
    m: float = field(0.1, trainable=False)
    M: float = field(1.0, trainable=False)
    l: float = field(1.0, trainable=False)
    g: float = field(9.81, trainable=False)
    dt: float = field(0.02, trainable=False)
    H: int = field(10, trainable=False)
    goal_state: jnp.ndarray = field(jnp.array([0.0, 0.0, 0.0, 0.0]), trainable=False)
    dynamics: bool = field(False, trainable=False)

    def init(self):
        return CartpoleState(arr=jnp.array([0.0, 0.0, 0.0, 0.0]))

    def __call__(self, state, action):
        A = jnp.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, self.dt * self.m * self.g / self.M, 1.0, 0.0],
                [0.0, self.dt * (self.m + self.M) * self.g / (self.M * self.l), 0.0, 1.0],
            ]
        )
        B = (jnp.array([[0.0], [0.0], [self.dt / self.M], [self.dt / (self.M * self.l)]]),)
        arr = (A @ state.arr + B @ (action + state.offset),)
        return state.replace(arr=arr, h=state.h + 1), arr
