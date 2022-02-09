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

"""Tests for analyzer."""

from absl.testing import absltest
import chex
from deluca.lung.utils.data.analyzer import Analyzer
import jax.numpy as jnp


class AnalyzerTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    result = {}
    u_in = jnp.array([1 for i in range(29)] +
                     [0 for i in range(29)] +
                     [1 for i in range(29)] + [0])
    u_out = jnp.array([0 for i in range(29)] +
                      [1 for i in range(29)] +
                      [0 for i in range(29)] + [1])
    pressure = jnp.array([1 for i in range(29)] +
                         [0 for i in range(29)] +
                         [1 for i in range(29)] + [0])
    target = jnp.array([1 for i in range(29)] +
                       [0 for i in range(29)] +
                       [1 for i in range(29)] + [0])
    timeseries = {
        'timestamp': jnp.array([0.03 * i for i in range(88)]),
        'pressure': pressure,
        'flow': jnp.array([0 for i in range(88)]),
        'target': target,
        'u_in': u_in,
        'u_out': u_out
    }
    result['timeseries'] = timeseries
    self.analyzer = Analyzer(result)

  def test_get_breaths(self):
    print(self.analyzer.get_breaths())
    u_in = jnp.array([1 for i in range(29)])
    pressure = jnp.array([1 for i in range(29)])
    for (u, p) in self.analyzer.get_breaths():
      assert jnp.allclose(u, u_in)
      assert jnp.allclose(p, pressure)

  # Note: in order to find the second breath, its required we end on u_out = 1
  def test_get_breath_indices(self):
    print(self.analyzer.get_breath_indices(use_cached=False))
    self.assertEqual(self.analyzer.get_breath_indices(use_cached=False),
                     [(0, 29), (58, 87)])

  def test_losses_per_breath(self):
    target = jnp.array([1 for i in range(29)])
    print(self.analyzer.losses_per_breath(target))
    self.assertEqual(
        list(self.analyzer.losses_per_breath(target)),
        [0., 0.])

  def test_default_metric(self):
    target = jnp.array([1 for i in range(29)])
    print(self.analyzer.default_metric(target=target))
    self.assertEqual(
        float(self.analyzer.default_metric(target=target)),
        0.)


if __name__ == '__main__':
  absltest.main()
