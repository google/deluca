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

"""Tests for transform."""

from absl.testing import absltest
import chex
from deluca.lung.utils.data.transform import ShiftScaleTransform
import jax.numpy as jnp


class TransformTest(chex.TestCase):

    def setUp(self):
        super().setUp()
        x = jnp.array([jnp.arange(5)])
        self.transform = ShiftScaleTransform(x)

    def test_call(self):
        v = jnp.array([2 for i in range(5)])
        assert jnp.allclose(self.transform(v), jnp.zeros((5,)))

    def test_inverse(self):
        v = jnp.zeros((5,))
        assert jnp.allclose(self.transform.inverse(v), jnp.array([2 for i in range(5)]))


if __name__ == "__main__":
    absltest.main()
