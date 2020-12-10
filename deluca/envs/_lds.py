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

from deluca.envs.core import Env
from deluca.utils import Random


class LDS(Env):
    def __init__(self, state_size=1, action_size=1, A=None, B=None, C=None, seed=0):

        if A is not None:
            assert (
                A.shape[0] == state_size and A.shape[1] == state_size
            ), "ERROR: Your input dynamics matrix does not have the correct shape."

        if B is not None:
            assert (
                B.shape[0] == state_size and B.shape[1] == action_size
            ), "ERROR: Your input dynamics matrix does not have the correct shape."

        self.random = Random(seed)

        self.state_size, self.action_size = state_size, action_size

        self.A = (
            A
            if A is not None
            else jax.random.normal(self.random.generate_key(), shape=(state_size, state_size))
        )

        self.B = (
            B
            if B is not None
            else jax.random.normal(self.random.generate_key(), shape=(state_size, action_size))
        )

        self.C = (
            C
            if C is not None
            else jax.numpy.identity(self.state_size)
        )

        self.t = 0

        self.reset()

    def step(self, action):
        self.state = self.A @ self.state + self.B @ action
        self.obs = self.C @ self.state

    @jax.jit
    def dynamics(self, state, action):
        new_state = self.A @ state + self.B @ action
        return new_state

    def reset(self):
        self.state = jax.random.normal(self.random.generate_key(), shape=(self.state_size, 1))
        self.obs = self.C @ self.state
