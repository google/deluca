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

"""Tests for test_simulator."""

from absl.testing import absltest
import chex
from deluca.lung.envs._learned_lung import LearnedLung
from deluca.lung.utils.data.breath_dataset import BreathDataset
from deluca.lung.utils.data.breath_dataset import get_initial_pressure
from deluca.lung.utils.scripts.test_simulator import test_simulator
import flax.linen as nn
import jax
import jax.numpy as jnp


class TestSimulatorTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    result = {}
    u_in = jnp.array([1 for i in range(29)] + [0 for i in range(29)] +
                     [1 for i in range(29)] + [0])
    u_out = jnp.array([0 for i in range(29)] + [1 for i in range(29)] +
                      [0 for i in range(29)] + [1])
    pressure = jnp.array([1 for i in range(29)] + [0 for i in range(29)] +
                         [1 for i in range(29)] + [0])
    target = jnp.array([1 for i in range(29)] + [0 for i in range(29)] +
                       [1 for i in range(29)] + [0])
    u_in = jnp.abs(u_in + jax.random.normal(jax.random.PRNGKey(0), (88,)))
    u_out = jnp.abs(u_out + jax.random.normal(jax.random.PRNGKey(1), (88,)))
    pressure = jnp.abs(pressure +
                       jax.random.normal(jax.random.PRNGKey(2), (88,)))
    target = jnp.abs(target + jax.random.normal(jax.random.PRNGKey(3), (88,)))

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

    params = None
    u_window = 5
    p_window = 3
    num_boundary_models = 0
    boundary_out_dim = 1
    boundary_hidden_dim = 100
    reset_scaled_peep = get_initial_pressure(self.dataset, 'train')
    default_model_name = 'MLP'
    default_model_parameters = {
        'hidden_dim': 10,
        'out_dim': 1,
        'n_layers': 4,
        'droprate': 0.0,
        'activation_fn': nn.relu
    }
    self.sim = LearnedLung.create(
        params=params,
        u_window=u_window,
        p_window=p_window,
        u_normalizer=self.dataset.u_normalizer,
        p_normalizer=self.dataset.p_normalizer,
        default_model_name=default_model_name,
        default_model_parameters=default_model_parameters,
        num_boundary_models=num_boundary_models,
        boundary_out_dim=boundary_out_dim,
        boundary_hidden_dim=boundary_hidden_dim,
        reset_normalized_peep=reset_scaled_peep,
    )

  # compute open loop score
  def test_test_simulator(self):
    open_loop_summary = test_simulator(self.sim, self.dataset)
    score = open_loop_summary['mae'].mean()
    self.assertTrue(jnp.allclose(float(score), 0.82767105))


if __name__ == '__main__':
  absltest.main()
