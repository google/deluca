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

"""Tests for train_simulator."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from deluca.lung.envs._learned_lung import LearnedLung
from deluca.lung.utils.data.breath_dataset import BreathDataset
from deluca.lung.utils.data.breath_dataset import get_initial_pressure
from deluca.lung.utils.data.breath_dataset import get_shuffled_and_batched_data
from deluca.lung.utils.scripts.train_simulator import loop_over_loader
from deluca.lung.utils.scripts.train_simulator import map_rollout_over_batch
from deluca.lung.utils.scripts.train_simulator import predict_and_update_state
from deluca.lung.utils.scripts.train_simulator import rollout
from deluca.lung.utils.scripts.train_simulator import train_simulator
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


class TrainSimulatorParameterizedTest(chex.TestCase):

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
    transition_threshold = 3
    self.sim = LearnedLung.create(
        params=params,
        u_window=u_window,
        p_window=p_window,
        u_normalizer=self.dataset.u_normalizer,
        p_normalizer=self.dataset.p_normalizer,
        transition_threshold=transition_threshold,
        default_model_name=default_model_name,
        default_model_parameters=default_model_parameters,
        num_boundary_models=num_boundary_models,
        boundary_out_dim=boundary_out_dim,
        boundary_hidden_dim=boundary_hidden_dim,
        reset_normalized_peep=reset_scaled_peep,
    )

  # test train simulator on both lr schedulers
  @parameterized.named_parameters(('Cosine', 'Cosine'),
                                  ('ReduceLROnPlateau', 'ReduceLROnPlateau'))
  def test_train_simulator(self, scheduler):
    r = 5
    c = 10
    batch_size = 2
    epochs = 50
    activation_fn_name = 'relu'
    self.sim, test_loss = train_simulator(
        self.dataset,
        self.sim,
        self.sim.num_boundary_models,
        activation_fn_name,
        r,
        c,
        batch_size=batch_size,
        epochs=epochs,
        patience=10,
        scheduler=scheduler,
        print_loss=1,
        use_tensorboard=False,
        mode='train')
    print('test_loss:' + str(test_loss))
    print('scheduler: ' + str(scheduler))
    if scheduler == 'Cosine':
      self.assertTrue(jnp.allclose(float(test_loss), 0.60894716, atol=1e-3))
    elif scheduler == 'ReduceLROnPlateau':
      self.assertTrue(jnp.allclose(float(test_loss), 0.60894716, atol=1e-3))

  @parameterized.product(variant=[chex.with_jit, chex.without_jit])
  def test_map_rollout_over_batch(self, variant):
    x_train, y_train = self.dataset.data['train']
    variant_map_rollout_over_batch = variant(
        map_rollout_over_batch, static_argnums=(2,))
    train_loss = variant_map_rollout_over_batch(self.sim, (x_train, y_train),
                                                rollout)
    print('test_map_rollout_over_batch loss:' + str(train_loss))
    self.assertTrue(jnp.allclose(float(train_loss), 0.9024319))

  def test_true_func(self):
    # testing with transition threshold 3
    model = self.sim
    x_train, y_train = self.dataset.data['train']
    data_zipped = jnp.array(list(zip(x_train, y_train)))  # (batch_size, 2, N)
    data = data_zipped[0]

    start_idx = 0
    end_idx = 6
    state, _ = model.reset()
    new_u_history = jnp.zeros((model.u_history_len,))
    new_p_history = jnp.zeros((model.p_history_len,))
    state = state.replace(u_history=new_u_history, p_history=new_p_history)

    loss_init = jnp.abs(
        model.p_normalizer(state.predicted_pressure) -
        model.p_normalizer(data[1, 0]))
    state_loss_model_data = (state, loss_init, model, data)
    print('state:' + str(state))
    for i in range(start_idx, end_idx):
      print('-------------------------')
      print('i:' + str(i))
      state_loss_model_data = predict_and_update_state(i, state_loss_model_data)
      state = state_loss_model_data[0]
      print('resulting state:' + str(state))
      if i >= model.transition_threshold:
        print('model.p_normalizer(data[1, i]):' +
              str(model.p_normalizer(data[1, i])))
        print('state.p_history[-2]):' + str(state.p_history[-2]))
        self.assertTrue(
            jnp.allclose(model.p_normalizer(data[1, i]), state.p_history[-2]))

  def test_false_func(self):
    # testing with transition threshold 3
    model = self.sim
    x_train, y_train = self.dataset.data['train']
    data_zipped = jnp.array(list(zip(x_train, y_train)))  # (batch_size, 2, N)
    data = data_zipped[0]

    start_idx = 0
    end_idx = 6
    state, _ = model.reset()
    new_u_history = jnp.zeros((model.u_history_len,))
    new_p_history = jnp.zeros((model.p_history_len,))
    state = state.replace(u_history=new_u_history, p_history=new_p_history)

    loss_init = jnp.abs(
        model.p_normalizer(state.predicted_pressure) -
        model.p_normalizer(data[1, 0]))
    state_loss_model_data = (state, loss_init, model, data)
    print('state:' + str(state))
    for i in range(start_idx, end_idx):
      print('-------------------------')
      print('i:' + str(i))
      old_state = state
      state_loss_model_data = predict_and_update_state(i, state_loss_model_data)
      state = state_loss_model_data[0]
      print('resulting state:' + str(state))
      if i < model.transition_threshold:
        print('old_state.p_history[-1]:' + str(old_state.p_history[-1]))
        print('state.p_history[-2]):' + str(state.p_history[-2]))
        self.assertTrue(
            jnp.allclose(old_state.p_history[-1], state.p_history[-2]))

  def test_predict_and_update_state(self):
    # testing with transition threshold 3
    model = self.sim
    x_train, y_train = self.dataset.data['train']
    data_zipped = jnp.array(list(zip(x_train, y_train)))  # (batch_size, 2, N)
    data = data_zipped[0]

    start_idx = 0
    end_idx = 6
    state, _ = model.reset()
    new_u_history = jnp.zeros((model.u_history_len,))
    new_p_history = jnp.zeros((model.p_history_len,))
    state = state.replace(u_history=new_u_history, p_history=new_p_history)

    loss_init = jnp.abs(
        model.p_normalizer(state.predicted_pressure) -
        model.p_normalizer(data[1, 0]))
    state_loss_model_data = (state, loss_init, model, data)
    print('state:' + str(state))
    for i in range(start_idx, end_idx):
      print('-------------------------')
      print('i:' + str(i))
      old_state = state
      state_loss_model_data = predict_and_update_state(i, state_loss_model_data)
      state = state_loss_model_data[0]
      print('resulting state:' + str(state))
      if i < model.transition_threshold:
        print('old_state.p_history[-1]:' + str(old_state.p_history[-1]))
        print('state.p_history[-2]):' + str(state.p_history[-2]))
        self.assertTrue(
            jnp.allclose(old_state.p_history[-1], state.p_history[-2]))
      else:
        print('model.p_normalizer(data[1, i]):' +
              str(model.p_normalizer(data[1, i])))
        print('state.p_history[-2]):' + str(state.p_history[-2]))
        self.assertTrue(
            jnp.allclose(model.p_normalizer(data[1, i]), state.p_history[-2]))

  @parameterized.product(variant=[chex.with_jit, chex.without_jit])
  def test_rollout(self, variant):
    x_train, y_train = self.dataset.data['train']
    data_zipped = jnp.array(list(zip(x_train, y_train)))  # (batch_size, 2, N)
    variant_rollout = variant(rollout)
    train_loss = variant_rollout(self.sim, data_zipped[0])
    print('test_rollout loss:' + str(train_loss))
    self.assertTrue(jnp.allclose(float(train_loss), 0.9024319))

  @parameterized.product(variant=[chex.with_jit, chex.without_jit])
  def test_loop_over_loader(self, variant):
    x_train, _ = self.dataset.data['train']
    batch_size = 2
    epochs = 1

    # set up optimizer and lr scheduler
    scheduler = 'Cosine'
    optimizer_params = {'learning_rate': 1e-3, 'weight_decay': 1e-4}
    lr_mult = 1.0
    steps_per_epoch = float(x_train.shape[0] / batch_size)
    decay_steps = int((epochs + 1) * steps_per_epoch)
    print('steps_per_epoch: %s', str(steps_per_epoch))
    print('decay_steps: %s', str(decay_steps))
    cosine_scheduler_fn = optax.cosine_decay_schedule(
        init_value=optimizer_params['learning_rate'], decay_steps=decay_steps)
    optimizer_params['learning_rate'] = cosine_scheduler_fn
    print('optimizer_params: %s', str(optimizer_params))
    optim = optax.adamw(**optimizer_params)
    optim_state = optim.init(self.sim)

    loop_over_loader_partial = functools.partial(
        loop_over_loader, optim=optim, rollout_fn=rollout, scheduler=scheduler)

    variant_loop_over_loader = variant(loop_over_loader_partial)

    prng_key = jax.random.PRNGKey(0)
    x, y, prng_key = get_shuffled_and_batched_data(self.dataset, batch_size,
                                                   'train', prng_key)

    (self.sim, optim_state, lr_mult,
     loss), _ = jax.lax.scan(variant_loop_over_loader,
                             (self.sim, optim_state, lr_mult, 0.), (x, y))

    print('test_loop_over_loader loss:' + str(loss))
    self.assertTrue(jnp.allclose(float(loss), 0.89504164))


if __name__ == '__main__':
  absltest.main()
