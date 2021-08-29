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
from absl import logging
from typing import Dict, Any
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import copy
from flax.metrics import tensorboard
import deluca.core
from deluca.lung.core import LungEnv
from deluca.lung.utils.data.transform import ShiftScaleTransform

from deluca.lung.utils.nn import SNN
from deluca.lung.utils.nn import MLP
from deluca.lung.utils.nn import CNN
from deluca.lung.utils.nn import LSTM
from deluca.lung.utils.nn import ShallowBoundaryModel
from google3.pyglib import gfile


class StitchedSimObservation(deluca.Obj):
  predicted_pressure: float = 0.0
  time: float = 0.0


class SimulatorState(deluca.Obj):
  u_history: jnp.ndarray
  p_history: jnp.ndarray
  t_in: int = 0
  steps: int = 0
  predicted_pressure: float = 0.0


class GeneralizedStitchedSim_open_loop(LungEnv):
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
  transition_threshold: int = deluca.field(0, jaxed=False)

  default_model_parameters: Dict[str, Any] = deluca.field(jaxed=False)
  num_boundary_models: int = deluca.field(5, jaxed=False)
  boundary_out_dim: int = deluca.field(1, jaxed=False)
  boundary_hidden_dim: int = deluca.field(100, jaxed=False)
  reset_scaled_peep: float = deluca.field(0.0, jaxed=False)

  default_model_name: str = deluca.field("SNN", jaxed=False)
  default_model: nn.module = deluca.field(jaxed=False)
  boundary_models: list = deluca.field(default_factory=list, jaxed=False)
  ensemble_models: list = deluca.field(default_factory=list, jaxed=False)

  def setup(self):
    self.u_history_len = max(self.u_window, self.num_boundary_models)
    self.p_history_len = max(self.p_window, self.num_boundary_models)
    if self.default_model_name == "SNN":
      self.default_model = SNN(
          out_dim=self.default_model_parameters["out_dim"],
          hidden_dim=self.default_model_parameters["hidden_dim"],
          n_layers=self.default_model_parameters["n_layers"],
          droprate=self.default_model_parameters["droprate"])
    elif self.default_model_name == "MLP":
      self.default_model = MLP(
          hidden_dim=self.default_model_parameters["hidden_dim"],
          out_dim=self.default_model_parameters["out_dim"],
          n_layers=self.default_model_parameters["n_layers"],
          droprate=self.default_model_parameters["droprate"],
          activation_fn=self.default_model_parameters["activation_fn"])
    elif self.default_model_name == "CNN":
      self.default_model = CNN(
          n_layers=self.default_model_parameters["n_layers"],
          out_channels=self.default_model_parameters["out_channels"],
          kernel_size=self.default_model_parameters["kernel_size"],
          strides=self.default_model_parameters["strides"],
          out_dim=self.default_model_parameters["out_dim"],
          activation_fn=self.default_model_parameters["activation_fn"])
    elif self.default_model_name == "LSTM":
      self.default_model = LSTM(
          n_layers=self.default_model_parameters["n_layers"],
          hidden_dim=self.default_model_parameters["hidden_dim"],
          out_dim=self.default_model_parameters["out_dim"],
          bptt=self.default_model_parameters["bptt"],
          activation_fn=self.default_model_parameters["activation_fn"])
          # seq_len=self.default_model_parameters["seq_len"])
    if self.default_model_name == "SNN" or self.default_model_name == "MLP":
      default_params = self.default_model.init(
          jax.random.PRNGKey(0),
          jnp.ones([self.u_history_len + self.p_history_len]))["params"]
    elif self.default_model_name == "CNN":
      default_params = self.default_model.init(
          jax.random.PRNGKey(0),
          jnp.ones([1, self.u_history_len + self.p_history_len, 1]))["params"]
      logging.info('[1, self.u_history_len + self.p_history_len, 1]:' + str([1, self.u_history_len + self.p_history_len, 1]))
    # TODO: edit rollout to init lstm_state and pass it to __call__
    elif self.default_model_name == "LSTM":
      # init_lstm_state = self.default_model.initialize_carry(batch_dims,
      #                                                       self.default_model_parameters["hidden_dim"])
      default_params = self.default_model.init(
          jax.random.PRNGKey(0),
          jnp.ones((1, self.u_history_len + self.p_history_len, 1)))["params"]

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
    # logging.info("ENSEMBLE MODELS:")
    # logging.info(self.ensemble_models)
    # without ensemble i.e. flattened params
    if self.params is None:
      self.params = boundary_params + [default_params]
    logging.info("TREE MAP")
    logging.info(jax.tree_map(lambda x: x.shape, self.params))

  def reset(self):
    scaled_peep = self.reset_scaled_peep
    state = SimulatorState(
        u_history=jnp.zeros([self.u_history_len]),
        p_history=jnp.hstack(
            [jnp.zeros([self.p_history_len-1]),
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
      # logging.info("default_features 1:" + str(default_features))
      default_pad_len = (self.u_history_len + self.p_history_len) - (
          self.u_window + self.p_window)
      default_features = jnp.hstack(
          [jnp.zeros((default_pad_len,)), default_features])
      # logging.info("default_features 2:" + str(default_features))
      if self.default_model_name == "CNN":
        boundary_features = jnp.expand_dims(boundary_features, axis=[0, 2])
        default_features = jnp.expand_dims(default_features, axis=[0, 2])
      if self.default_model_name == "LSTM":
        boundary_features = jnp.expand_dims(boundary_features, axis=[0, 2])
        default_features = jnp.expand_dims(default_features, axis=[0, 2])
      features = jax.lax.cond(
          model_idx == self.num_boundary_models,
          lambda x: default_features,
          lambda x: boundary_features,
          None,
      )
      if self.default_model_name != "LSTM":
        scaled_pressure = jax.lax.switch(model_idx, funcs, features)
      else:
        scaled_pressure = jax.lax.switch(model_idx, funcs, features)

      state = state.replace(predicted_pressure=scaled_pressure)
      new_p_history = self.update_history(state.p_history,
                                          scaled_pressure)
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
  end_idx = len(data[0])-1 # need minus 1 since we predict for i+1 at idx i
  state, _ = model.reset()
  new_u_history = jnp.zeros((model.u_history_len,))
  new_p_history = jnp.zeros((model.p_history_len,))
  state = state.replace(u_history=new_u_history, p_history=new_p_history)

  loss_init = jnp.abs(model.p_scaler(state.predicted_pressure) - model.p_scaler(data[1, 0]))
  state_loss_init = (state, loss_init)

  def predict_and_update_state(i, state_loss):
    state, loss = state_loss
    u_in, pressure = data[0, i], data[1, i]  # unscaled u_in and pressure
    # logging.info("---------------------------")
    # logging.info("i:" + str(i))
    # logging.info("state:" + str(state))
    # logging.info("scaled u_in:" + str(model.u_scaler(u_in).squeeze()))
    # logging.info("true_pressure to add:" + str(model.p_scaler(pressure).squeeze()))
    # if i >= model.transition_threshold:
    def true_func(state, u_in, pressure):
      # new_p_history = jnp.append(state.p_history,
      #                            model.p_scaler(pressure).squeeze())
      # new_p_history = model.update_history(state.p_history, model.p_scaler(pressure).squeeze())
      new_p_history = state.p_history.at[-1].set(
          model.p_scaler(pressure).squeeze())
      # logging.info("new_p_history:" + str(new_p_history))
      state = state.replace(p_history=new_p_history)
      next_state, _ = model(state=state, action=(u_in, 0))
      # drop predicted_pressure since we are training with known true pressure
      # then, u_history.shape = u_history_len+1, p_history.shape = p_history_len+1
      # truncated_p_history = next_state.p_history[:-1]
      # next_state = next_state.replace(p_history=truncated_p_history)
      return next_state
    # else:  # use predicted pressures for p_history
    def false_func(state, u_in):
      next_state, _ = model(state=state, action=(u_in, 0))
      return next_state
    partial_true_func = partial(true_func, u_in=u_in, pressure=pressure)
    partial_false_func = partial(false_func, u_in=u_in)
    next_state = jax.lax.cond(i >= model.transition_threshold,
                              partial_true_func,
                              partial_false_func,
                              state)
    # logging.info('next_state:' + str(next_state))
    pred = model.p_scaler(next_state.predicted_pressure)  # scaled gradient step
    # logging.info("pred:" + str(pred))

    # for debugging test_loss vs. open_loop_score
    return (next_state, loss + jnp.abs(model.p_scaler(data[1, i + 1]) - pred))
    # scaled gradient step


  (state, total_loss) = jax.lax.fori_loop(start_idx, end_idx,
                                          predict_and_update_state,
                                          state_loss_init)

  '''state_loss = state_loss_init
  for i in range(start_idx, end_idx):
    state_loss = predict_and_update_state(i, state_loss)
  (_, total_loss) = state_loss'''

  return total_loss / len(data[0])


def loop_over_loader(model_optimState_lrMult_loss, X_Y, optim, rollout, scheduler):
  """
    rollout has signature (model, data) -> loss where data.shape = (2, N)
    X_batch.shape = Y_batch.shape = (num_batches, batch_size, N=29)
    lrMult is the multiplier for the scheduler
  """
  # logging.info('=================== ENTER loop_over_loader ======================')
  X_batch, y_batch = X_Y
  model, optim_state, lr_mult, loss = model_optimState_lrMult_loss
  loss, grad = jax.value_and_grad(map_rollout_over_batch)(model,
                                                          (X_batch, y_batch),
                                                          rollout)
  updates, optim_state = optim.update(grad, optim_state, model)
  if scheduler == "ReduceLROnPlateau":
    updates = jax.tree_map(lambda g: lr_mult * g, updates)
  model = optax.apply_updates(model, updates)
  return (model, optim_state, lr_mult, loss), None


def map_rollout_over_batch(model, data, rollout):
  """
    rollout has signature (model, data) -> loss where data.shape = (2, N)
    data.shape = ((batch_size, N), (batch_size, N))
    """
  rollout_partial = lambda xs: partial(rollout, model=model)(data=xs)
  data_zipped = jnp.array(list(zip(data[0], data[1])))  # (batch_size, 2, N)
  losses = jax.vmap(rollout_partial)(data_zipped)
  return jnp.array(losses).mean()


def generalized_stitched_sim_train_open_loop(
    munger,
    model,
    num_boundary_models,
    activation_fn_name,
    R,
    C,
    # idx 0 to num_boundary_models-1 are boundary models,
    # idx num_boundary_models is default_model
    train_key="train",
    test_key="test",
    batch_size=512,
    epochs=500,
    optimizer=optax.adamw,
    optimizer_params={
        "learning_rate": 1e-3,
        "weight_decay": 1e-4
    },
    patience=10,
    lr_decay_factor=0.1,
    scheduler="ReduceLROnPlateau",  # or "Cosine"
    loss_fn=lambda x, y: (jnp.abs(x - y)).mean(),
    print_loss=100,
    use_tensorboard=False,
    mode='train',
):
  # Unpack dataloader
  loader = munger.get_dataloader_tf(
      key=train_key, shuffle=True, scale_data=False, batch_size=batch_size)

  # evaluate on these at end of epoch
  X_train, y_train = munger.get_scaled_data(train_key, scale_data=False)
  X_test, y_test = munger.get_scaled_data(test_key, scale_data=False)

  # set up optimizer and lr scheduler
  lr_mult = 1.0
  if scheduler == "ReduceLROnPlateau":
    optim = optimizer(**optimizer_params)
    patience_cnt = 0
    prev_loss = float("inf")
  elif scheduler == "Cosine":
    steps_per_epoch = float(X_train.shape[0] / batch_size)
    decay_steps = int((epochs + 1) * steps_per_epoch)
    logging.info("steps_per_epoch:" + str(steps_per_epoch))
    logging.info("decay_steps:" + str(decay_steps))
    cosine_scheduler_fn = optax.cosine_decay_schedule(
        init_value=optimizer_params["learning_rate"], decay_steps=decay_steps)
    learning_rate = optimizer_params["learning_rate"]
    optimizer_params["learning_rate"] = cosine_scheduler_fn
    logging.info("optimizer_params:" + str(optimizer_params))
    optim = optimizer(**optimizer_params)
  optim_state = optim.init(model)

  loop_over_loader_partial = partial(
      loop_over_loader, optim=optim, rollout=rollout, scheduler=scheduler)

  # Tensorboard writer
  if use_tensorboard:
    config = copy.deepcopy(model.default_model_parameters)
    del config["activation_fn"]
    config["activation_fn_name"] = activation_fn_name

    default_model_name = model.default_model_name
    if mode == "train":
      config["learning_rate"] = learning_rate
      config["weight_decay"] = optimizer_params["weight_decay"]
      dirname = "R" + str(R) + "_C" + str(C) + "_learning_rate" + str(
          learning_rate) + "_weight_decay" + str(
              optimizer_params["weight_decay"]) + "/"
      file_name = str(config)
      workdir = "/cns/lu-d/home/brain-pton/deluca_lung/model_ckpts/simulator_search_transition_threshold/" + default_model_name + "_boundary_models0_grid_results_analysis/" + dirname
    elif mode == "featurization":
      config["batch_size"] = batch_size
      dirname = "R" + str(R) + "_C" + str(C) + "_" + str(config) + "/"
      file_name = "u_window" + str(model.u_window) + "_p_window" + str(
          model.p_window)
      workdir = "/cns/lu-d/home/brain-pton/deluca_lung/model_ckpts/simulator_search_transition_threshold/" + default_model_name + "_boundary_models0_featurization_analysis/" + dirname
    elif mode == "stitchedSim":
      config["u_window"] = model.u_window
      config["p_window"] = model.p_window
      config["batch_size"] = batch_size
      dirname = "R" + str(R) + "_C" + str(C) + "_" + str(config) + "/"
      s1 = "num_boundary_models" + str(model.num_boundary_models)
      s2 = "_transition_threshold" + str(model.transition_threshold)
      s3 = "_learning_rate" + str(learning_rate)
      s4 = "_weight_decay" + str(optimizer_params["weight_decay"])
      file_name = s1 + s2 + s3 + s4
      workdir = "/cns/lu-d/home/brain-pton/deluca_lung/model_ckpts/simulator_search_transition_threshold/" + default_model_name + "_boundary_models0_stitchedSim_analysis/" + dirname
    gfile.MakeDirs(os.path.dirname(workdir))
    write_path = workdir + file_name
    summary_writer = tensorboard.SummaryWriter(write_path)
    summary_writer.hparams(dict(config))

  # Main Training Loop
  for epoch in range(epochs + 1):
    if epoch % 10 == 0:
      logging.info("epoch:" + str(epoch))
    X, y = get_X_y_for_next_epoch_tf(loader, batch_size)
    # logging.info("X.shape:" + str(X.shape))
    # logging.info("y.shape: " + str(y.shape))

    (model, optim_state, lr_mult,
     loss), _ = jax.lax.scan(loop_over_loader_partial,
                             (model, optim_state, lr_mult, 0.), (X, y))
    '''for i in range(X.shape[0]):
      carry = (model, optim_state, lr_mult, 0.)
      carry, _ = loop_over_loader_partial(carry, (X[i], y[i]))
    model, optim_state, lr_mult, loss = carry'''
    if scheduler == "ReduceLROnPlateau":
      if loss > prev_loss:
        patience_cnt = patience_cnt + 1
      else:
        patience_cnt = 0
      if patience_cnt == patience:
        lr_mult = lr_mult * lr_decay_factor
        patience_cnt = 0
      prev_loss = loss
    # elif scheduler == "Cosine":
    #   lr_mult = cosine_scheduler_fn(epoch)

    if epoch % print_loss == 0:
      if scheduler == "ReduceLROnPlateau":
        logging.info("loss:" + str(loss))
        logging.info("prev_loss:" + str(prev_loss))
        logging.info("patience_cnt:" + str(patience_cnt))
        logging.info("lr_mult:" + str(lr_mult))
      # expensive end-of-epoch eval, just for intuition
      # logging.info("X_train.shape:" + str(X_train.shape))
      # logging.info("y_train.shape:" + str(y_train.shape))
      train_loss = map_rollout_over_batch(model, (X_train, y_train), rollout)
      # cross-validation
      # logging.info("X_test.shape:" + str(X_test.shape))
      # logging.info("y_test.shape:" + str(y_test.shape))

      test_loss = map_rollout_over_batch(model, (X_test, y_test), rollout)
      test_loss_unscaled = test_loss * model.p_scaler.std # biases cancel out in pred - truth

      if epoch % print_loss == 0:
        logging.info(
            f"Epoch {epoch:2d}: train={train_loss.item():.5f}, test_loss={test_loss.item():.5f}, test_loss_unscaled={test_loss_unscaled.item():.5f}"
        )
        logging.info("-----------------------------------")
      if use_tensorboard:
        summary_writer.scalar("train_loss", train_loss, epoch)
        summary_writer.scalar("test_loss", test_loss, epoch)
  if use_tensorboard:
    summary_writer.flush()
  logging.info("finished looping over epochs")
  return model, test_loss
