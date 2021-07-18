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


class ReacherState(Obj):
    arr: jnp.ndarray = field(trainable=True)
    h: int = field(0, trainable=True)


class Reacher(Env):
    m1: float = field(1.0, trainable=False)
    m2: float = field(1.0, trainable=False)
    l1: float = field(1.0, trainable=False)
    l2: float = field(1.0, trainable=False)
    g: float = field(9.81, trainable=False)
    max_torque: float = field(1.0, trainable=False)
    dt: float = field(0.01, trainable=False)
    H: int = field(200, trainable=False)
    goal_coord: jnp.ndarray = field(jnp.array([0.0, 1.8]), trainable=False)

    def init(self):
        initial_th = (jnp.pi / 4, jnp.pi / 2)
        return ReacherState(
            arr=jnp.array(
                [
                    *initial_th,
                    0.0,
                    0.0,
                    self.l1 * jnp.cos(initial_th[0])
                    + self.l2 * jnp.cos(initial_th[0] + initial_th[1])
                    - self.goal_coord[0],
                    self.l1 * jnp.sin(initial_th[0])
                    + self.l2 * jnp.sin(initial_th[0] + initial_th[1])
                    - self.goal_coord[1],
                ]
            )
        )

    def __call__(self, state, action):
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        th1, th2, dth1, dth2, Dx, Dy = state.arr
        t1, t2 = action

        a11 = (m1 + m2) * l1 ** 2 + m2 * l2 ** 2 + 2 * m2 * l1 * l2 * jnp.cos(th2)
        a12 = m2 * l2 ** 2 + m2 * l1 * l2 * jnp.cos(th2)
        a22 = m2 * l2 ** 2
        b1 = (
            t1
            + m2 * l1 * l2 * (2 * dth1 + dth2) * dth2 * jnp.sin(th2)
            - m2 * l2 * g * jnp.sin(th1 + th2)
            - (m1 + m2) * l1 * g * jnp.sin(th1)
        )
        b2 = t2 - m2 * l1 * l2 * dth1 ** 2 * jnp.sin(th2) - m2 * l2 * g * jnp.sin(th1 + th2)
        A, b = jnp.array([[a11, a12], [a12, a22]]), jnp.array([b1, b2])
        ddth1, ddth2 = jnp.linalg.inv(A) @ b

        th1, th2 = th1 + dth1 * self.dt, th2 + dth2 * self.dt
        dth1, dth2 = dth1 + ddth1 * self.dt, dth2 + ddth2 * self.dt
        Dx, Dy = (
            l1 * jnp.cos(th1) + l2 * jnp.cos(th1 + th2) - self.goal_coord[0],
            l1 * jnp.sin(th1) + l2 * jnp.sin(th1 + th2) - self.goal_coord[1],
        )

        arr = jnp.array([th1, th2, dth1, dth2, Dx, Dy])
        new_state = state.replace(arr=arr, h=state.h + 1)

        return new_state, arr
