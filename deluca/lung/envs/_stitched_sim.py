# Copyright 2021 The Deluca Authors.
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

from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

import deluca.core
from deluca.lung.core import LungEnv
from deluca.lung.utils.data.transform import ShiftScaleTransform

from deluca.lung.utils.nn import SNN
from deluca.lung.utils.nn import ShallowBoundaryModel


class StitchedSimObservation(deluca.Obj):
  predicted_pressure: float = 0.0
  time: float = 0.0


class SimulatorState(deluca.Obj):
  u_history: jnp.ndarray
  p_history: jnp.ndarray
  t_in: int = 0
  steps: int = 0
  predicted_pressure: float = 0.0


class StitchedSim(LungEnv):
  params: list = deluca.field(jaxed=True)

  init_rng: jnp.array = deluca.field(jaxed=False)
  u_window: int = deluca.field(5, jaxed=False)
  p_window: int = deluca.field(3, jaxed=False)
  u_history_len: int = deluca.field(5, jaxed=False)
  p_history_len: int = deluca.field(5, jaxed=False)
  u_scaler: ShiftScaleTransform = deluca.field(jaxed=False)
  p_scaler: ShiftScaleTransform = deluca.field(jaxed=False)
  to_round: bool = deluca.field(False, jaxed=False)
  seed: int = deluca.field(0, jaxed=False)
  flow: int = deluca.field(0, jaxed=False)

  default_out_dim: int = deluca.field(1, jaxed=False)
  default_hidden_dim: int = deluca.field(100, jaxed=False)
  default_n_layers: int = deluca.field(4, jaxed=False)
  default_dropout_prob: float = deluca.field(0.0, jaxed=False)

  num_boundary_models: int = deluca.field(5, jaxed=False)
  boundary_out_dim: int = deluca.field(1, jaxed=False)
  boundary_hidden_dim: int = deluca.field(100, jaxed=False)
  reset_scaled_peep: float = deluca.field(0.0, jaxed=False)

  default_model: nn.module = deluca.field(SNN, jaxed=False)
  boundary_models: list = deluca.field(default_factory=list, jaxed=False)
  ensemble_models: list = deluca.field(default_factory=list, jaxed=False)

  def setup(self):
    self.u_history_len = max(self.u_window, self.num_boundary_models)
    self.p_history_len = max(self.p_window, self.num_boundary_models)

    self.default_model = SNN(
        out_dim=self.default_out_dim,
        hidden_dim=self.default_hidden_dim,
        n_layers=self.default_n_layers,
    )
    default_params = self.default_model.init(
        jax.random.PRNGKey(0),
        jnp.ones([self.u_history_len + self.p_history_len]))["params"]

    self.boundary_models = [
        ShallowBoundaryModel(
            out_dim=self.boundary_out_dim,
            hidden_dim=self.boundary_hidden_dim,
            model_num=i + 1) for i in range(self.num_boundary_models)
    ]
    boundary_params = [
        self.boundary_models[i].init(
            jax.random.PRNGKey(0),
            jnp.ones([self.u_history_len + self.p_history_len]))["params"]
        for i in range(self.num_boundary_models)
    ]

    self.ensemble_models = self.boundary_models + [self.default_model]
    # print("ENSEMBLE MODELS:")
    # print(self.ensemble_models)
    # without ensemble i.e. flattened params
    if self.params is None:
      self.params = boundary_params + [default_params]
    # print("TREE MAP")
    # print(jax.tree_map(lambda x: x.shape, self.params))

  def reset(self):
    scaled_peep = self.reset_scaled_peep
    state = SimulatorState(
        u_history=jnp.zeros([self.u_history_len]),
        p_history=jnp.hstack(
            [jnp.zeros([self.p_history_len]),
             jnp.array([scaled_peep])]),
        t_in=0,
        steps=0,
        predicted_pressure=scaled_peep,
    )
    state = self.sync_unscaled_pressure(state)
    obs = StitchedSimObservation(
        predicted_pressure=state.predicted_pressure, time=self.time(state))
    return state, obs

  def update_history(self, history, value):
    history = jnp.roll(history, shift=-1)
    history = history.at[-1].set(value)
    return history

  def sync_unscaled_pressure(self, state):
    scaled_pressure = self.p_scaler.inverse(state.predicted_pressure).squeeze()
    return state.replace(predicted_pressure=scaled_pressure)

  def __call__(self, state, action):
    u_in, u_out = action
    model_idx = jnp.min(jnp.array([state.t_in, self.num_boundary_models]))
    funcs = [
        partial(self.ensemble_models[i].apply, {"params": self.params[i]})
        for i in range(self.num_boundary_models + 1)
    ]

    def true_func(state):

      def call_if_t_in_positive(state):
        new_state, _ = self.reset()
        return new_state

      state = jax.lax.cond(state.t_in > 0, call_if_t_in_positive, lambda x: x,
                           state)
      return state

    def false_func(state, u_in, model_idx):
      if self.to_round:
        u_in_scaled = self.u_scaler(jnp.round(u_in)).squeeze()
      else:
        u_in_scaled = self.u_scaler(u_in).squeeze()
      state = state.replace(
          u_history=self.update_history(state.u_history, u_in_scaled),
          t_in=state.t_in + 1)
      boundary_features = jnp.hstack([
          state.u_history[-self.u_history_len:],
          state.p_history[-self.p_history_len:]
      ])
      default_features = jnp.hstack(
          [state.u_history[-self.u_window:], state.p_history[-self.p_window:]])
      default_pad_len = (self.u_history_len + self.p_history_len) - (
          self.u_window + self.p_window)
      default_features = jnp.hstack(
          [jnp.zeros((default_pad_len,)), default_features])
      features = jax.lax.cond(
          model_idx == self.num_boundary_models,
          lambda x: default_features,
          lambda x: boundary_features,
          None,
      )

      scaled_pressure = jax.lax.switch(model_idx, funcs, features)[0]

      state = state.replace(predicted_pressure=scaled_pressure.squeeze())
      new_p_history = self.update_history(state.p_history,
                                          scaled_pressure.squeeze())
      state = state.replace(
          p_history=new_p_history
      )  # just for inference. This is undone during training
      state = self.sync_unscaled_pressure(state)  # unscale predicted pressure
      return state

    partial_false_func = partial(false_func, u_in=u_in, model_idx=model_idx)
    state = jax.lax.cond(u_out == 1, true_func, partial_false_func, state)
    obs = StitchedSimObservation(
        predicted_pressure=state.predicted_pressure, time=self.time(state))
    return state.replace(steps=state.steps + 1), obs


def get_X_y_for_next_epoch_tf(loader, batch_size):
  # loader is a tensorflow.data.Dataset
  loader = loader.shuffle(buffer_size=len(loader), seed=0)
  loader = loader.batch(batch_size=batch_size, drop_remainder=True)
  loader_list = list(loader)
  unzipped_loader = list(zip(*loader_list))
  X = jnp.array([jnp.array(z.numpy()) for z in unzipped_loader[0]])
  y = jnp.array([jnp.array(z.numpy()) for z in unzipped_loader[1]])
  return X, y


def rollout(model, data):
  # data.shape == (2, N)
  start_idx = 0
  end_idx = len(data[0])
  state, _ = model.reset()
  new_u_history = jnp.zeros((model.u_history_len,))
  new_p_history = jnp.zeros((model.p_history_len,))
  state = state.replace(u_history=new_u_history, p_history=new_p_history)
  state_loss_init = (state, 0.0)

  def predict_and_update_state(i, state_loss):
    state, loss = state_loss
    u_in, pressure = data[0, i], data[1, i]  # unscaled u_in and pressure
    new_p_history = jnp.append(state.p_history,
                               model.p_scaler(pressure).squeeze())
    state = state.replace(p_history=new_p_history)
    next_state, _ = model(state=state, action=(u_in, 0))
    # drop predicted_pressure so that u_history.shape = u_history_len+1, p_history.shape = p_history_len+1
    truncated_p_history = next_state.p_history[:-1]
    next_state = next_state.replace(p_history=truncated_p_history)
    pred = model.p_scaler(next_state.predicted_pressure)  # scaled gradient step
    return (
        next_state,
        loss + jnp.abs(model.p_scaler(data[1, i + 1]) - pred),
    )  # scaled gradient step

  (state, total_loss) = jax.lax.fori_loop(start_idx, end_idx,
                                          predict_and_update_state,
                                          state_loss_init)
  return total_loss / (end_idx - start_idx)


def loop_over_loader(model_optimState, X_Y, optim, rollout):
  """
    rollout has signature (model, data) -> loss where data.shape = (2, N)
    X_batch.shape = Y_batch.shape = (num_batches, batch_size, N=29)
  """
  # print('=================== ENTER loop_over_loader ======================')
  X_batch, y_batch = X_Y
  model, optim_state = model_optimState
  value, grad = jax.value_and_grad(map_rollout_over_batch)(model,
                                                           (X_batch, y_batch),
                                                           rollout)
  updates, optim_state = optim.update(grad, optim_state, model)
  model = optax.apply_updates(model, updates)
  return (model, optim_state), None


def map_rollout_over_batch(model, data, rollout):
  """
    rollout has signature (model, data) -> loss where data.shape = (2, N)
    data.shape = ((batch_size, N), (batch_size, N))
    """
  rollout_partial = lambda xs: partial(rollout, model=model)(data=xs)
  data_zipped = jnp.array(list(zip(data[0], data[1])))  # (batch_size, 2, N)
  losses = jax.vmap(rollout_partial)(data_zipped)
  return jnp.array(losses).mean()


# TODO: add scheduler and scheduler_params
def stitched_sim_train(
    munger,
    model,
    num_boundary_models,
    # 0 to num_boundary_models-1 are boundary models, and num_boundary_models is #
    # default_model
    train_key="train",
    test_key="test",
    batch_size=512,
    epochs=500,
    optimizer=optax.adam,
    optimizer_params={"learning_rate": 1e-3},
    # scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    # scheduler_params={"factor": 0.9, "patience": 15},
    loss_fn=lambda x, y: (jnp.abs(x - y)).mean(),
    # loss_fn_params={},
    print_loss=100,
):
  # Unpack dataloader
  loader = munger.get_dataloader_tf(
      key=train_key, shuffle=True, scale_data=False, batch_size=batch_size)

  # evaluate on these at end of epoch
  X_train, y_train = munger.get_scaled_data(train_key, scale_data=False)
  X_test, y_test = munger.get_scaled_data(test_key, scale_data=False)

  # set up model and optimizer
  optim = optimizer(**optimizer_params)
  optim_state = optim.init(model)

  loop_over_loader_partial = partial(
      loop_over_loader, optim=optim, rollout=rollout)
  for epoch in range(epochs + 1):
    if epoch % 25 == 0:
      print("epoch:" + str(epoch))
    X, y = get_X_y_for_next_epoch_tf(loader, batch_size)

    (model, optim_state), _ = jax.lax.scan(loop_over_loader_partial,
                                           (model, optim_state), (X, y))

    if epoch % print_loss == 0:
      # expensive end-of-epoch eval, just for intuition
      print("X_train.shape:" + str(X_train.shape))
      print("y_train.shape:" + str(y_train.shape))
      train_loss = map_rollout_over_batch(model, (X_train, y_train),
                                          rollout)
      # cross-validation
      print("X_test.shape:" + str(X_test.shape))
      print("y_test.shape:" + str(y_test.shape))
      test_loss = map_rollout_over_batch(model, (X_test, y_test),
                                         rollout)
      print(
          f"Epoch {epoch:2d}: train={train_loss.item():.5f}, test={test_loss.item():.5f}"
      )
      print("-----------------------------------")
  print("finished looping over epochs")
  return model
