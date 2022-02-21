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

"""Tests for train_controller."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from deluca.lung.controllers._deep_cnn import DeepCnn
from deluca.lung.envs._learned_lung import LearnedLung
from deluca.lung.utils.data.breath_dataset import BreathDataset
from deluca.lung.utils.data.breath_dataset import get_initial_pressure
from deluca.lung.utils.scripts.train_controller import rollout
from deluca.lung.utils.scripts.train_controller import rollout_parallel
from deluca.lung.utils.scripts.train_controller import train_controller
import flax.linen as nn
import jax
import jax.numpy as jnp


class TrainControllerParameterizedTest(chex.TestCase):

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

  # test all training modes
  @parameterized.named_parameters(('test1', 'parallel', 'multipip'),
                                  ('test2', 'sequential', 'singular'))
  def test_train_controller(self, pip_feed, mode):
    _, train_loss, score = train_controller(
        self.controller,
        self.sim,
        pip_feed=pip_feed,
        mode=mode,
        duration=0.87,
        dt=0.03,
        epochs=50,
        optimizer_params=self.optimizer_params,
        scheduler='Cosine',
        tensorboard_dir=None,
        model_parameters=self.model_parameters,
        print_loss=1)
    print('pip_feed:' + str(pip_feed))
    print('mode:' + str(mode))
    print('train_loss:' + str(train_loss))
    print('score:' + str(score))

    # TODO(dsuo) remove one of parallel/sequential if results same
    if (pip_feed, mode) == ('parallel', 'multipip'):
      self.assertTrue(jnp.allclose(float(score), 20.96012))
    elif (pip_feed, mode) == ('parallel', 'singular'):
      self.assertTrue(jnp.allclose(float(score), 33.595024))
    elif (pip_feed, mode) == ('sequential', 'multipip'):
      self.assertTrue(jnp.allclose(score, 21.095024))
    elif (pip_feed, mode) == ('sequential', 'singular'):
      self.assertTrue(jnp.allclose(score, 33.380775))

  @parameterized.product(variant=[chex.with_jit, chex.without_jit])
  def test_rollout(self, variant):
    peep = 0
    pip = 1
    use_noise = False
    loss_fn = lambda x, y: (jnp.abs(x - y)).mean()
    loss = jnp.array(0.)
    duration = 0.87
    dt = 0.03
    tt = jnp.linspace(0, duration, int(duration / dt))
    variant_rollout = variant(rollout, static_argnums=(6,))
    value, _ = jax.value_and_grad(variant_rollout)(self.controller, self.sim,
                                                   tt, use_noise, peep, pip,
                                                   loss_fn, loss)
    print('test_rollout loss:' + str(value))
    self.assertTrue(jnp.allclose(float(value), 11.748882))

  @parameterized.product(variant=[chex.with_jit, chex.without_jit])
  def test_rollout_parallel(self, variant):
    peep = 0
    pips = [1, 2]
    use_noise = False
    loss_fn = lambda x, y: (jnp.abs(x - y)).mean()
    duration = 0.87
    dt = 0.03
    tt = jnp.linspace(0, duration, int(duration / dt))
    variant_rollout = variant(rollout_parallel, static_argnums=(6,))
    value, _ = jax.value_and_grad(variant_rollout)(self.controller, self.sim,
                                                   tt, use_noise, peep,
                                                   jnp.array(pips), loss_fn)
    print('test_rollout_parallel loss:' + str(value))
    self.assertTrue(jnp.allclose(float(value), 14.495776))


if __name__ == '__main__':
  absltest.main()
