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

import jax.numpy as jnp

from deluca.core import Agent
from deluca.core import field


class Zero(Agent):
    d_action: int = 1
    #    agent_state: jnp.ndarray = field(default_factory=lambda: jnp.ndarray([[1.0]]), jaxed=False)

    def init(self, d_action):
        self.d_action = d_action
        #        self.agent_state = None
        return None

    def __call__(self, obs):
        return jnp.zeros((self.d_action, 1))

    def update(self, y: jnp.array, u: jnp.ndarray) -> None:
        return None
