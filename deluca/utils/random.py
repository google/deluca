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
import jax
import jax.numpy as jnp

from deluca.core import JaxObject


class Random(JaxObject):
    def __init__(self, seed=0) -> None:
        """Initialize instance"""
        self.set_key(seed)

    def set_key(self, seed=None) -> None:
        self.key = jax.random.PRNGKey(seed)

    def get_key(self) -> jnp.ndarray:
        return self.key

    def generate_key(self) -> jnp.ndarray:
        """Generates random subkey"""
        self.key, subkey = jax.random.split(self.key)
        return subkey
