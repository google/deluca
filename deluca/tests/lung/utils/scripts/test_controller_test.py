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

"""Tests for test_controller."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from deluca.lung.controllers._bang_bang import BangBang
from deluca.lung.controllers._deep_cnn import DeepCnn
from deluca.lung.controllers._deep_mlp import DeepMlp
from deluca.lung.controllers._pid import PID
from deluca.lung.controllers._predestined import Predestined
from deluca.lung.controllers._residual import CompositeController
from deluca.lung.envs._balloon_lung import BalloonLung
from deluca.lung.envs._delay_lung import DelayLung
from deluca.lung.envs._learned_lung import LearnedLung
from deluca.lung.envs._single_comp_lung import SingleCompLung
from deluca.lung.utils.data.breath_dataset import BreathDataset
from deluca.lung.utils.data.breath_dataset import get_initial_pressure
from deluca.lung.utils.scripts.test_controller import test_controller
import flax.linen as nn
import jax
import jax.numpy as jnp


class TestControllerParameterizedTest(chex.TestCase):

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
    self.sims = {}
    self.controllers = {}

    self.learned_lung = LearnedLung.create(
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
    self.sims['learned_lung'] = self.learned_lung

    self.balloon_lung = BalloonLung.create()
    self.sims['balloon_lung'] = self.balloon_lung

    self.single_comp_lung = SingleCompLung.create()
    self.sims['single_comp_lung'] = self.single_comp_lung

    self.delay_lung = DelayLung.create()
    self.sims['delay_lung'] = self.delay_lung

    learning_rate = 1e-3
    weight_decay = 1e-4
    h = 100
    history_len = 10
    kernel_size = 5

    self.optimizer_params = {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }

    self.deep_cnn = DeepCnn.create(
        H=h,
        history_len=history_len,
        kernel_size=kernel_size,
        use_model_apply_jit=False,
        use_leaky_clamp=True)

    self.controllers['deep_cnn'] = self.deep_cnn

    self.bang_bang = BangBang.create()
    self.controllers['bang_bang'] = self.bang_bang

    hidden_dim = 200
    n_layers = 4
    activation_fn_name = 'leaky_relu'
    droprate = 0.0
    back_history_len = 29
    fwd_history_len = 2
    horizon = 29
    self.deep_mlp = DeepMlp.create(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        droprate=droprate,
        activation_fn_name=activation_fn_name,
        back_history_len=back_history_len,
        fwd_history_len=fwd_history_len,
        horizon=horizon,
        use_model_apply_jit=False)
    self.controllers['deep_mlp'] = self.deep_mlp

    # self.expiratory = Expiratory.create()
    # self.controllers['expiratory'] = self.expiratory

    self.pid = PID.create()
    self.controllers['pid'] = self.pid

    self.predestined = Predestined.create(u_ins=jnp.zeros(horizon,))
    self.controllers['predestined'] = self.predestined

    self.residual = CompositeController(base_controller=self.pid,
                                        resid_controller=self.deep_cnn)

    self.controllers['residual'] = self.residual

  # test all lungs vs. all controllers
  @parameterized.product(
      sim_name=['learned_lung',
                'balloon_lung',
                'delay_lung',
                'single_comp_lung'],
      controller_name=[
          'bang_bang', 'deep_cnn', 'deep_mlp', 'pid',
          'predestined', 'residual'
      ])
  def test_sim_controller(self, sim_name, controller_name):
    print('sim_name:' + str(sim_name))
    print('controller_name:' + str(controller_name))
    sim = self.sims[sim_name]
    controller = self.controllers[controller_name]
    pips = [10, 15, 20, 25, 30, 35]
    peep = 5
    score = test_controller(controller, sim, pips, peep)
    print('score:' + str(score))

  def test_train_controller(self):
    pips = [10, 15, 20, 25, 30, 35]
    peep = 5
    score = test_controller(self.deep_cnn, self.learned_lung, pips, peep)
    print('score:' + str(score))
    self.assertTrue(jnp.allclose(float(score), 21.095024))


if __name__ == '__main__':
  absltest.main()
