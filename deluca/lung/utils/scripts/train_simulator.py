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

"""functions for training simulator."""
import copy
import functools

from absl import logging
from clu import metric_writers
from deluca.lung.utils.data.breath_dataset import get_shuffled_and_batched_data
from flax.metrics import tensorboard
import jax
import jax.numpy as jnp
import optax

# pylint: disable=pointless-string-statement
# pylint: disable=invalid-name
# pylint: disable=g-long-lambda
# pylint: disable=dangerous-default-value
# pylint: disable=unused-argument
# pylint: disable=logging-format-interpolation


# Case 1: i >= model.transition_threshold use true pressures for p_history
def true_func(state, u_in, pressure, model):
  print("ENTER TRUE FUNC")
  print("pressure:" + str(pressure))
  print("scaled_pressure:" + str(model.p_normalizer(pressure).squeeze()))
  new_p_history = state.p_history.at[-1].set(
      model.p_normalizer(pressure).squeeze())
  state = state.replace(p_history=new_p_history)
  next_state, _ = model(state=state, action=(u_in, 0))
  return next_state


# Case 2: i < model.transition_threshold use predicted pressures for p_history
def false_func(state, u_in, model):
  print("ENTER FALSE FUNC")
  next_state, _ = model(state=state, action=(u_in, 0))
  return next_state


def predict_and_update_state(i, state_loss_model_data):
  """predict and update function."""
  state, loss, model, data = state_loss_model_data
  u_in, pressure = data[0, i], data[1, i]  # unnormalized u_in and pressure
  partial_true_func = functools.partial(
      true_func, u_in=u_in, pressure=pressure, model=model)
  partial_false_func = functools.partial(false_func, u_in=u_in, model=model)
  next_state = jax.lax.cond(i >= model.transition_threshold, partial_true_func,
                            partial_false_func, state)
  pred = model.p_normalizer(
      next_state.predicted_pressure)  # normalized gradient step
  return (next_state, loss + jnp.abs(model.p_normalizer(data[1, i + 1]) - pred),
          model, data)


@jax.jit
def rollout(model, data):
  """rollout function."""
  # data.shape == (2, N)
  start_idx = 0
  end_idx = len(data[0]) - 1  # need minus 1 since we predict for i+1 at idx i
  state, _ = model.reset()
  new_u_history = jnp.zeros((model.u_history_len,))
  new_p_history = jnp.zeros((model.p_history_len,))
  state = state.replace(u_history=new_u_history, p_history=new_p_history)

  loss_init = jnp.abs(
      model.p_normalizer(state.predicted_pressure) -
      model.p_normalizer(data[1, 0]))
  state_loss_model_data = (state, loss_init, model, data)

  (state, total_loss, _, _) = jax.lax.fori_loop(start_idx, end_idx,
                                                predict_and_update_state,
                                                state_loss_model_data)
  """for i in range(start_idx, end_idx):

    state_loss_model_data = predict_and_update_state(i, state_loss_model_data)
  (_, total_loss) = state_loss_model_data
  """

  return total_loss / len(data[0])


def loop_over_loader(model_optimState_lrMult_loss, X_Y, optim, rollout_fn,
                     scheduler):
  """loop over data loader.

    X_batch.shape = Y_batch.shape = (num_batches, batch_size, N=29)
    lrMult is the multiplier for the scheduler

  Args:
    model_optimState_lrMult_loss: (model, optimState, lr_mult, loss)
    X_Y: the data
    optim: optimizer
    rollout_fn: has signature (model, data) -> loss where data.shape = (2, N)
    scheduler: lr scheduler

  Returns:
    updated model, optim_state, lr_mult, and loss
  """
  X_batch, y_batch = X_Y
  model, optim_state, lr_mult, loss = model_optimState_lrMult_loss
  loss, grad = jax.value_and_grad(map_rollout_over_batch)(model,
                                                          (X_batch, y_batch),
                                                          rollout_fn)
  updates, optim_state = optim.update(grad, optim_state, model)
  if scheduler == "ReduceLROnPlateau":
    updates = jax.tree_map(lambda g: lr_mult * g, updates)
  model = optax.apply_updates(model, updates)
  return (model, optim_state, lr_mult, loss), None


# @partial(jax.jit, static_argnums=(2,))
def map_rollout_over_batch(model, data, rollout_fn):
  """map rollout over batch dimension.

  Args:
    model: the model
    data: data.shape = ((batch_size, N), (batch_size, N))
    rollout_fn: has signature (model, data) -> loss where data.shape = (2, N)
  Returns:
    loss
  """
  rollout_partial = lambda xs: functools.partial(
      rollout_fn, model=model)(
          data=xs)
  data_zipped = jnp.array(list(zip(data[0], data[1])))  # (batch_size, 2, N)
  losses = jax.vmap(rollout_partial)(data_zipped)
  return jnp.array(losses).mean()


def train_simulator(
    dataset,
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
    print_loss=10,
    use_tensorboard=False,
    mode="train",
    user_name="alexjyu-brain",
    tb_dir=None,
):
  """train simulator."""
  # evaluate on these at end of epoch
  for key in ["train", "test"]:
    dataset.data[key] = (jnp.array(dataset.data[key][0]),
                         jnp.array(dataset.data[key][1]))
  X_train, y_train = dataset.data[train_key]
  X_test, y_test = dataset.data[test_key]

  # set up optimizer and lr scheduler
  lr_mult = 1.0
  if scheduler == "ReduceLROnPlateau":
    optim = optimizer(**optimizer_params)
    patience_cnt = 0
    prev_loss = float("inf")
  elif scheduler == "Cosine":
    steps_per_epoch = float(X_train.shape[0] / batch_size)
    decay_steps = int((epochs + 1) * steps_per_epoch)
    logging.info("steps_per_epoch: %s", str(steps_per_epoch))
    logging.info("decay_steps: %s", str(decay_steps))
    cosine_scheduler_fn = optax.cosine_decay_schedule(
        init_value=optimizer_params["learning_rate"], decay_steps=decay_steps)
    optimizer_params["learning_rate"] = cosine_scheduler_fn
    logging.info("optimizer_params: %s", str(optimizer_params))
    optim = optimizer(**optimizer_params)
  optim_state = optim.init(model)

  loop_over_loader_partial = functools.partial(
      loop_over_loader, optim=optim, rollout_fn=rollout, scheduler=scheduler)

  # Tensorboard writer
  if use_tensorboard:
    config = copy.deepcopy(model.default_model_parameters)
    del config["activation_fn"]
    config["activation_fn_name"] = activation_fn_name

    if mode == "train":
      file_name = str(config)
    write_path = tb_dir + file_name
    summary_writer = metric_writers.create_default_writer(
        logdir=write_path, just_logging=jax.process_index() != 0)
    summary_writer = tensorboard.SummaryWriter(write_path)
    summary_writer.write_hparams(dict(config))

  # Main Training Loop
  prng_key = jax.random.PRNGKey(0)
  for epoch in range(epochs + 1):
    if epoch % 10 == 0:
      logging.info("epoch: %s", str(epoch))
    X, y, prng_key = get_shuffled_and_batched_data(dataset, batch_size,
                                                   train_key, prng_key)
    if epoch == 0:
      logging.info("X.shape: %s", str(X.shape))
      logging.info("y.shape: %s", str(y.shape))

    (model, optim_state, lr_mult,
     loss), _ = jax.lax.scan(loop_over_loader_partial,
                             (model, optim_state, lr_mult, 0.), (X, y))
    """for i in range(X.shape[0]):

      carry = (model, optim_state, lr_mult, 0.)
      carry, _ = loop_over_loader_partial(carry, (X[i], y[i]))
    model, optim_state, lr_mult, loss = carry
    """
    if scheduler == "ReduceLROnPlateau":
      if loss > prev_loss:
        patience_cnt = patience_cnt + 1
      else:
        patience_cnt = 0
      if patience_cnt == patience:
        lr_mult = lr_mult * lr_decay_factor
        patience_cnt = 0
      prev_loss = loss

    if epoch % print_loss == 0:
      if scheduler == "ReduceLROnPlateau":
        logging.info("loss: %s", str(loss))
        logging.info("prev_loss: %s", str(prev_loss))
        logging.info("patience_cnt: %s", str(patience_cnt))
        logging.info("lr_mult: %s", str(lr_mult))
      # expensive end-of-epoch eval, just for intuition
      train_loss = map_rollout_over_batch(model, (X_train, y_train), rollout)
      # cross-validation
      test_loss = map_rollout_over_batch(model, (X_test, y_test), rollout)

      if epoch % print_loss == 0:
        logging.info(
            f"Epoch {epoch:2d}: train={train_loss.item():.5f}, test_loss={test_loss.item():.5f}"
        )
        logging.info("-----------------------------------")
      if use_tensorboard:
        summary_writer.write_scalars(epoch, {"train_loss": train_loss})
        summary_writer.write_scalars(epoch, {"test_loss": test_loss})
  if use_tensorboard:
    summary_writer.flush()
  logging.info("finished looping over epochs")
  return model, test_loss
