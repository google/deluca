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

import jax
import jax.numpy as jnp
import optax
import numpy as np
import itertools
import flax.linen as nn

import deluca.core
import functools
import copy
import os
import time
from absl import logging
from deluca.lung.core import Controller
from deluca.lung.core import ControllerState
from deluca.lung.core import BreathWaveform
from deluca.lung.controllers._expiratory import Expiratory
from deluca.lung.envs._stitched_sim import StitchedSimObservation
from deluca.lung.utils.data.transform import ShiftScaleTransform
from deluca.lung.core import DEFAULT_DT
from deluca.lung.core import proper_time
from flax.metrics import tensorboard
from google3.pyglib import gfile
from collections.abc import Callable



class DeepNetwork(nn.Module):
  H: int = 100
  kernel_size: int = 5

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(
        features=self.H,
        kernel_size=(self.kernel_size,),
        padding='VALID',
        strides=1,
        name=f"deep_conv")(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=1, use_bias=True, name=f"deep_fc")(x)
    return x


class DeepControllerState(deluca.Obj):
  waveform: deluca.Obj  # waveform has to be here because it is subject to change during training
  errs: jnp.array
  time: float = float("inf")
  steps: int = 0
  dt: float = DEFAULT_DT


class Deep_cnn(Controller):
  params: list = deluca.field(jaxed=True)
  model: nn.module = deluca.field(DeepNetwork, jaxed=False)
  featurizer: jnp.array = deluca.field(jaxed=False)
  H: int = deluca.field(100, jaxed=False)
  input_dim: int = deluca.field(1, jaxed=False)
  history_len: int = deluca.field(10, jaxed=False)
  kernel_size: int = deluca.field(5, jaxed=False)
  clip: float = deluca.field(40.0, jaxed=False)
  normalize: bool = deluca.field(False, jaxed=False)
  u_scaler: ShiftScaleTransform = deluca.field(jaxed=False)
  p_scaler: ShiftScaleTransform = deluca.field(jaxed=False)
  model_apply: Callable = deluca.field(jaxed=False)

  def setup(self):
    self.model = DeepNetwork(H=self.H, kernel_size=self.kernel_size)
    if self.params is None:
      self.params = self.model.init(
          jax.random.PRNGKey(0),
          jnp.expand_dims(jnp.ones([self.history_len]), axis=(0, 2)))["params"]

    # linear feature transform:
    # errs -> [average of last h errs, ..., average of last 2 errs, last err]
    # emulates low-pass filter bank

    self.featurizer = jnp.tril(jnp.ones((self.history_len, self.history_len)))
    self.featurizer /= jnp.expand_dims(
        jnp.arange(self.history_len, 0, -1), axis=0)
    self.model_apply = jax.jit(self.model.apply)

    '''if self.normalize:
      self.u_scaler = u_scaler
      self.p_scaler = p_scaler'''

  # TODO: Handle dataclass initialization of jax objects
  def init(self, waveform=None):
    if waveform is None:
      waveform = BreathWaveform.create()
    errs = jnp.array([0.0] * self.history_len)
    state = DeepControllerState(errs=errs, waveform=waveform)
    return state

  @jax.jit
  def __call__(self, controller_state, obs):
    state, t = obs.predicted_pressure, obs.time
    errs, waveform = controller_state.errs, controller_state.waveform
    target = waveform.at(t)

    if self.normalize:
      target_scaled = self.p_scaler(target).squeeze()
      state_scaled = self.p_scaler(state).squeeze()
      next_errs = jnp.roll(errs, shift=-1)
      next_errs = next_errs.at[-1].set(target_scaled - state_scaled)
    else:
      next_errs = jnp.roll(errs, shift=-1)
      next_errs = next_errs.at[-1].set(target - state)
    controller_state = controller_state.replace(errs=next_errs)

    decay = waveform.decay(t)

    def true_func(null_arg):
      trajectory = jnp.expand_dims(next_errs[-self.history_len:], axis=(0, 1))
      input = jnp.reshape((trajectory @ self.featurizer), (1, self.history_len, 1))
      u_in = self.model_apply({"params": self.params}, input)

      return u_in.squeeze().astype(jnp.float32)

    # changed decay compare from None to float(inf) due to cond requirements
    u_in = jax.lax.cond(
        jnp.isinf(decay), true_func, lambda x: jnp.array(decay), None)

    # u_in = jax.lax.clamp(0.0, u_in.astype(jnp.float32), self.clip).squeeze()
    # Implementing "leaky" clamp to solve the zero gradient problem
    u_in = jax.lax.cond(u_in < 0.0, lambda x: x*0.01, lambda x: x, u_in)
    u_in = jax.lax.cond(u_in > self.clip, lambda x: self.clip, lambda x: x, u_in)

    # update controller_state
    new_dt = jnp.max(
        jnp.array([DEFAULT_DT, t - proper_time(controller_state.time)]))
    new_time = t
    new_steps = controller_state.steps + 1
    controller_state = controller_state.replace(
        time=new_time, steps=new_steps, dt=new_dt)
    return controller_state, u_in


def rollout(controller, sim, tt, use_noise, PEEP, PIP, loss_fn, loss):
  waveform = BreathWaveform.create(custom_range=(PEEP, PIP))
  expiratory = Expiratory.create(waveform=waveform)
  controller_state = controller.init(waveform)
  expiratory_state = expiratory.init()
  sim_state, obs = sim.reset()

  def loop_over_tt(ctrlState_expState_simState_obs_loss, t):
    controller_state, expiratory_state, sim_state, obs, loss = ctrlState_expState_simState_obs_loss
    mean = 1.5
    std = 1.0
    noise = mean + std * jax.random.normal(jax.random.PRNGKey(0), shape=())
    pressure = sim_state.predicted_pressure + use_noise * noise
    sim_state = sim_state.replace(
        predicted_pressure=pressure)
    obs = obs.replace(predicted_pressure=pressure)

    controller_state, u_in = controller(controller_state, obs)
    expiratory_state, u_out = expiratory(expiratory_state, obs)

    sim_state, obs = sim(sim_state, (u_in, u_out))
    loss = jax.lax.cond(
        u_out == 0, lambda x: x + loss_fn(jnp.array(waveform.at(t)), pressure),
        lambda x: x, loss)
    return (controller_state, expiratory_state, sim_state, obs, loss), None

  (_, _, _, _, loss), _ = jax.lax.scan(
      loop_over_tt, (controller_state, expiratory_state, sim_state, obs, loss),
      tt)
  return loss

@functools.partial(jax.jit, static_argnums=(6,))
def rollout_parallel(controller, sim, tt, use_noise, PEEP, PIPs, loss_fn):
  '''loss = jnp.array(0.)
  for PIP in PIPs:
    loss = rollout(controller, sim, tt, use_noise, PEEP, PIP, loss_fn, loss)
  return loss'''
  rollout_partial = lambda p: functools.partial(
      rollout,
      controller=controller,
      sim=sim,
      tt=tt,
      use_noise=use_noise,
      PEEP=PEEP,
      loss_fn=loss_fn,
      loss=0.0)(PIP=p)
  losses = jax.vmap(rollout_partial)(jnp.array(PIPs))
  # logging.info('losses:' + str(losses))
  loss = losses.sum() / float(len(PIPs))
  return loss


# Question: Jax analogue of torch.autograd.set_detect_anomaly(True)?
def deep_train_cnn(
    controller,
    sim,
    pip_feed="parallel",
    duration=0.87,
    dt=0.03,
    epochs=100,
    use_noise=False,
    optimizer=optax.adamw,
    optimizer_params={
        "learning_rate": 1e-3,
        "weight_decay": 1e-4
    },
    loss_fn=lambda x, y: (jnp.abs(x - y)).mean(),
    scheduler="Cosine",
    tensorboard_dir=None,
    mode='train',
    print_loss=1,
):
  PIPs = [10, 15, 20, 25, 30, 35]
  PEEP = 5

  optim_params = copy.deepcopy(optimizer_params)
  if scheduler == "Cosine":
    if pip_feed == "parallel":
      steps_per_epoch = 1
    elif pip_feed == "sequential":
      steps_per_epoch = len(PIPs)

    decay_steps = int(epochs * steps_per_epoch)
    logging.info("steps_per_epoch:" + str(steps_per_epoch))
    logging.info("decay_steps:" + str(decay_steps))
    cosine_scheduler_fn = optax.cosine_decay_schedule(
        init_value=optim_params["learning_rate"], decay_steps=decay_steps)
    optim_params["learning_rate"] = cosine_scheduler_fn
    logging.info("optim_params:" + str(optim_params))
    optim = optimizer(**optim_params)
  optim_state = optim.init(controller)

  tt = jnp.linspace(0, duration, int(duration / dt))
  losses = []

  # torch.autograd.set_detect_anomaly(True)
  # TODO: handle device-awareness

  # Tensorboard writer
  if tensorboard_dir is not None:
    model_parameters = {
        "H": controller.H,
        "kernel_size": controller.kernel_size,
        "history_len": controller.history_len,
        "epochs": epochs,
    }

    if mode == "train":
      trial_name = str(model_parameters)
      write_path = tensorboard_dir + trial_name
    gfile.MakeDirs(os.path.dirname(write_path))
    summary_writer = tensorboard.SummaryWriter(write_path)
    summary_writer.hparams(model_parameters)


  for epoch in range(epochs):
    # a = time.time()
    if pip_feed == "parallel":
      value, grad = jax.value_and_grad(rollout_parallel)(controller, sim, tt,
                                                         use_noise, PEEP, jnp.array(PIPs),
                                                         loss_fn)
      # logging.info('time elapsed rollout_parallel:' + str(time.time() - a))
      # a = time.time()
      updates, optim_state = optim.update(grad, optim_state, controller)
      controller = optax.apply_updates(controller, updates)
      per_step_loss = value / len(tt)
      losses.append(per_step_loss)
      logging.info('VALUE:' + str(value))
      # logging.info('time elapsed losses.append:' + str(time.time() - a))
      # a = time.time()

      if epoch % print_loss == 0:
        logging.info(f"Epoch: {epoch}\tLoss: {per_step_loss:.2f}")
        if tensorboard_dir is not None:
          summary_writer.scalar("per_step_loss", per_step_loss, epoch)

    if pip_feed == "sequential":
      for PIP in PIPs:
        value, grad = jax.value_and_grad(rollout)(controller, sim, tt,
                                                  use_noise, PEEP, PIP, loss_fn,
                                                  jnp.array(0.))
        updates, optim_state = optim.update(grad, optim_state, controller)
        controller = optax.apply_updates(controller, updates)
        per_step_loss = value / len(tt)
        losses.append(per_step_loss)

        if epoch % print_loss == 0:
          logging.info(f"Epoch: {epoch}, PIP: {PIP}\tLoss: {per_step_loss:.2f}")
          if tensorboard_dir is not None:
            summary_writer.scalar("per_step_loss", per_step_loss, epoch)
  return controller, per_step_loss
