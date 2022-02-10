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

"""Tests for run_controller."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from deluca.lung.controllers._deep_cnn import DeepCnn
from deluca.lung.controllers._expiratory import Expiratory
from deluca.lung.core import BreathWaveform
from deluca.lung.envs._learned_lung import LearnedLung
from deluca.lung.utils.data.breath_dataset import BreathDataset
from deluca.lung.utils.data.breath_dataset import get_initial_pressure
from deluca.lung.utils.scripts.run_controller import loop_over_tt
from deluca.lung.utils.scripts.run_controller import run_controller
from deluca.lung.utils.scripts.run_controller import run_controller_scan
import flax.linen as nn
import jax
import jax.numpy as jnp

# pylint: disable=pointless-string-statement


class RunControllerTest(chex.TestCase):

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

    learning_rate = 1e-3
    weight_decay = 1e-4
    h = 100
    history_len = 10
    kernel_size = 5

    self.optimizer_params = {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }

    self.controller = DeepCnn.create(
        H=h,
        history_len=history_len,
        kernel_size=kernel_size,
        use_model_apply_jit=False,
        use_leaky_clamp=True)

    self.model_parameters = {
        'H': h,
        'history_len': history_len,
        'kernel_size': kernel_size,
    }
    self.expected_u_ins = jnp.array([
        -0.13033567, -0.20070392, -0.24765663, -0.28653696, -0.30381608,
        -0.32553822, -0.33032474, -0.31910658, -0.3274433, -0.33562654,
        -0.3355351, -0.33554694, -0.3354946, -0.33546093, -0.33545992,
        -0.33545637, -0.33545426, -0.33545238, -0.33545154, -0.33545196,
        -0.335452, -0.33545145, -0.33545142, -0.33545142, -0.33545142,
        -0.33545142, -0.33545142, -0.33545142, -0.33545142
    ])
    self.expected_pressures = jnp.array([
        1.0337607, 1.2874004, 1.3058296, 1.3317846, 1.4556241, 1.4488808,
        1.4532007, 1.4582397, 1.4606066, 1.4610327, 1.4608891, 1.4605807,
        1.4616922, 1.4622655, 1.462111, 1.4621205, 1.4621186, 1.4621154,
        1.4621159, 1.4621155, 1.4621152, 1.4621152, 1.4621152, 1.4621152,
        1.4621152, 1.462115, 1.462115, 1.462115, 1.462115
    ])

  # test jitted and unjitted variants of inner loop
  @parameterized.product(variant=[chex.with_jit, chex.without_jit])
  def test_loop_over_tt(self, variant):
    dt = 0.03
    horizon = 29
    waveform = BreathWaveform.create()
    expiratory = Expiratory.create(waveform=waveform)
    controller_state = self.controller.init()
    expiratory_state = expiratory.init()
    state, obs = self.sim.reset()
    """jit_loop_over_tt = jax.jit(

        functools.partial(
            loop_over_tt,
            controller=self.controller,
            expiratory=expiratory,
            env=self.sim,
            dt=dt))
    """
    variant_loop_over_tt = variant(
        functools.partial(
            loop_over_tt,
            controller=self.controller,
            expiratory=expiratory,
            env=self.sim,
            dt=dt))

    _, (_, u_ins, _, pressures,
        _) = jax.lax.scan(variant_loop_over_tt,
                          (state, obs, controller_state, expiratory_state, 0),
                          jnp.arange(horizon))
    print('test_loop_over_tt u_ins:' + str(list(u_ins)))
    print('test_loop_over_tt pressure:' + str(list(pressures)))

    self.assertTrue(jnp.allclose(u_ins, self.expected_u_ins))
    self.assertTrue(jnp.allclose(pressures, self.expected_pressures))

  def test_run_controller_scan(self):
    result = run_controller_scan(
        self.controller,
        T=29,
        dt=0.03,
        abort=60,
        env=self.sim,
        waveform=None,
        use_tqdm=False,
        directory=None,
        init_controller=False,
    )['timeseries']
    print('test_run_controller_scan u_in:' + str(list(result['u_in'])))
    print('test_run_controller_scan pressure:' + str(list(result['pressure'])))
    self.assertTrue(jnp.allclose(result['u_in'], self.expected_u_ins))
    self.assertTrue(jnp.allclose(result['pressure'], self.expected_pressures))

  def test_run_controller(self):
    result = run_controller(
        self.controller,
        T=29,
        dt=0.03,
        abort=60,
        env=self.sim,
        waveform=None,
        use_tqdm=False,
        directory=None,
    )['timeseries']
    print('test_run_controller u_in:' + str(list(result['u_in'])))
    print('test_run_controller pressure:' + str(list(result['pressure'])))
    self.assertTrue(jnp.allclose(result['u_in'], self.expected_u_ins))
    self.assertTrue(jnp.allclose(result['pressure'], self.expected_pressures))


if __name__ == '__main__':
  absltest.main()
