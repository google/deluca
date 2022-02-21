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

"""Tests for breath_dataset."""
from absl.testing import absltest
import chex
from deluca.lung.utils.data.breath_dataset import BreathDataset
from deluca.lung.utils.data.breath_dataset import get_initial_pressure
from deluca.lung.utils.data.breath_dataset import get_shuffled_and_batched_data
import jax
import jax.numpy as jnp


class BreathDatasetTest(chex.TestCase):

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
    paths = [result for i in range(10)]
    self.dataset = BreathDataset.from_paths(paths, clip=(0, None))

  def test_get_shuffled_and_batched_data(self):
    prng_key = jax.random.PRNGKey(0)
    batch_size = 2
    x, y, prng_key = get_shuffled_and_batched_data(
        self.dataset, batch_size, 'train', prng_key)
    u_in = jnp.array([1 for i in range(29)])
    p = jnp.array([1 for i in range(29)])
    self.assertEqual(x.shape, (9, 2, 29))
    self.assertEqual(y.shape, (9, 2, 29))
    assert jnp.allclose(u_in, x[0, 0])
    assert jnp.allclose(p, y[0, 0])

  def test_get_initial_pressure(self):
    p_0 = get_initial_pressure(self.dataset, 'train')
    jnp.isnan(p_0)  # nan since std = 0.

if __name__ == '__main__':
  absltest.main()
