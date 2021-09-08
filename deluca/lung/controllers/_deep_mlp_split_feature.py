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

from absl import logging
import functools
import os
import deluca.core
import copy
from deluca.lung.controllers._expiratory import Expiratory
from deluca.lung.core import BreathWaveform
from deluca.lung.core import Controller
from deluca.lung.core import DEFAULT_DT
from deluca.lung.core import proper_time
from deluca.lung.utils.data.transform import ShiftScaleTransform
from deluca.lung.utils.nn import MLP
import flax.linen as nn
from flax.metrics import tensorboard
import jax
import jax.numpy as jnp
import optax
from google3.pyglib import gfile
from collections.abc import Callable


class DeepControllerState(deluca.Obj):
  waveform: deluca.Obj  # waveform has to be here because it is subject to change during training
  bwd_targets: jnp.array
  bwd_pressures: jnp.array
  fwd_targets: jnp.array
  time: float = float("inf")
  steps: int = 0
  dt: float = DEFAULT_DT


class Deep_split_feature(Controller):
  params: list = deluca.field(jaxed=True)
  model: nn.module = deluca.field(MLP, jaxed=False)
  hidden_dim: int = deluca.field(100, jaxed=False)
  n_layers: int = deluca.field(2, jaxed=False)
  droprate: float = deluca.field(0.0, jaxed=False)
  activation_fn_name: str = deluca.field("leaky_relu", jaxed=False)
  back_history_len: int = deluca.field(10, jaxed=False)
  fwd_history_len: int = deluca.field(2, jaxed=False)
  horizon: int = deluca.field(29, jaxed=False)
  clip: float = deluca.field(100.0, jaxed=False)
  normalize: bool = deluca.field(False, jaxed=False)
  u_scaler: ShiftScaleTransform = deluca.field(jaxed=False)
  p_scaler: ShiftScaleTransform = deluca.field(jaxed=False)
  model_apply: Callable = deluca.field(jaxed=False)

  def setup(self):
    if self.activation_fn_name == "leaky_relu":
      activation_fn = nn.leaky_relu
    self.model = MLP(
        hidden_dim=self.hidden_dim,
        out_dim=1,
        n_layers=self.n_layers,
        droprate=self.droprate,
        activation_fn=activation_fn)
    if self.params is None:
      self.params = self.model.init(
          jax.random.PRNGKey(0),
          jnp.ones([2*self.back_history_len + self.fwd_history_len]))["params"]
    self.model_apply = jax.jit(self.model.apply)

    # linear feature transform:
    # errs -> [average of last h errs, ..., average of last 2 errs, last err]
    # emulates low-pass filter bank

  # TODO: Handle dataclass initialization of jax objects
  def init(self, waveform=None):
    if waveform is None:
      waveform = BreathWaveform.create()
    bwd_targets = jnp.array([0.0] * self.back_history_len)
    bwd_pressures = jnp.array([0.0] * self.back_history_len)
    fwd_targets = jnp.array(
        [waveform.at(t * DEFAULT_DT) for t in range(self.fwd_history_len)])
    state = DeepControllerState(
        bwd_targets=bwd_targets,
        bwd_pressures=bwd_pressures,
        fwd_targets=fwd_targets,
        waveform=waveform)
    return state

  @jax.jit
  def __call__(self, controller_state, obs):
    state, t = obs.predicted_pressure, obs.time
    bwd_targets, bwd_pressures, waveform = (controller_state.bwd_targets,
                                            controller_state.bwd_pressures,
                                            controller_state.waveform)
    fwd_targets = controller_state.fwd_targets
    target = waveform.at(t)
    fwd_t = t + self.fwd_history_len * DEFAULT_DT
    if self.fwd_history_len > 0:
      fwd_target = jax.lax.cond(fwd_t >= self.horizon * DEFAULT_DT,
                                lambda x: fwd_targets[-1],
                                lambda x: waveform.at(fwd_t),
                                None)

    if self.normalize:
      target_scaled = self.p_scaler(target).squeeze()
      state_scaled = self.p_scaler(state).squeeze()
      next_bwd_targets = jnp.roll(bwd_targets, shift=-1)
      next_bwd_targets = next_bwd_targets.at[-1].set(target_scaled)
      next_bwd_pressures = jnp.roll(bwd_pressures, shift=-1)
      next_bwd_pressures = next_bwd_pressures.at[-1].set(state_scaled)
      if self.fwd_history_len > 0:
        fwd_target_scaled = self.p_scaler(fwd_target).squeeze()
        next_fwd_targets = jnp.roll(fwd_targets, shift=-1)
        next_fwd_targets = next_fwd_targets.at[-1].set(fwd_target_scaled)
      else:
        next_fwd_targets = jnp.array([])
    else:
      next_bwd_targets = jnp.roll(bwd_targets, shift=-1)
      next_bwd_targets = next_bwd_targets.at[-1].set(target)
      next_bwd_pressures = jnp.roll(bwd_pressures, shift=-1)
      next_bwd_pressures = next_bwd_pressures.at[-1].set(state)
      if self.fwd_history_len > 0:
        next_fwd_targets = jnp.roll(fwd_targets, shift=-1)
        next_fwd_targets = next_fwd_targets.at[-1].set(fwd_target)
      else:
        next_fwd_targets = jnp.array([])
    controller_state = controller_state.replace(bwd_targets=next_bwd_targets,
                                                bwd_pressures=next_bwd_pressures,
                                                fwd_targets=next_fwd_targets)
    decay = waveform.decay(t)
    def true_func(null_arg):
      trajectory = jnp.hstack([next_bwd_targets, next_bwd_pressures, next_fwd_targets])
      u_in = self.model_apply({"params": self.params},
                              trajectory)
      return u_in.squeeze().astype(jnp.float64)

    # changed decay compare from None to float(inf) due to cond requirements
    u_in = jax.lax.cond(
        jnp.isinf(decay), true_func, lambda x: jnp.array(decay), None)
    u_in = jax.lax.clamp(0.0, u_in.astype(jnp.float64), self.clip).squeeze()
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

  # for i in range(len(tt)):
  #   (controller_state, expiratory_state, sim_state, obs, loss), _ = loop_over_tt((controller_state, expiratory_state, sim_state, obs, loss), tt[i])
  return loss


@functools.partial(jax.jit, static_argnums=(6,))
def rollout_parallel(controller, sim, tt, use_noise, PEEP, PIPs, loss_fn):
  loss = jnp.array(0.)
  # for PIP in PIPs:
  #   loss = rollout(controller, sim, tt, use_noise, PEEP, PIP, loss_fn, loss)
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
  loss = losses.sum() / float(len(PIPs))
  return loss


# Question: Jax analogue of torch.autograd.set_detect_anomaly(True)?
def deep_train_split_feature(
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
    print("steps_per_epoch:" + str(steps_per_epoch))
    print("decay_steps:" + str(decay_steps))
    cosine_scheduler_fn = optax.cosine_decay_schedule(
        init_value=optim_params["learning_rate"], decay_steps=decay_steps)
    optim_params["learning_rate"] = cosine_scheduler_fn
    print("optim_params:" + str(optim_params))
    optim = optimizer(**optim_params)
  optim_state = optim.init(controller)

  tt = jnp.linspace(0, duration, int(duration / dt))
  losses = []

  # torch.autograd.set_detect_anomaly(True)
  # TODO: handle device-awareness

  # Tensorboard writer
  if tensorboard_dir is not None:
    model_parameters = {
        "hidden_dim": controller.hidden_dim,
        "n_layers": controller.n_layers,
        "droprate": controller.droprate,
        "activation_fn_name": controller.activation_fn_name,
        "back_history_len": controller.back_history_len,
        "fwd_history_len": controller.fwd_history_len,
        "horizon": controller.horizon,
        "epochs": epochs,
    }

    if mode == "train":
      trial_name = str(model_parameters)
      write_path = tensorboard_dir + trial_name
    gfile.MakeDirs(os.path.dirname(write_path))
    summary_writer = tensorboard.SummaryWriter(write_path)
    summary_writer.hparams(model_parameters)

  for epoch in range(epochs):
    if pip_feed == "parallel":
      value, grad = jax.value_and_grad(rollout_parallel)(controller, sim, tt,
                                                         use_noise, PEEP,
                                                         jnp.array(PIPs),
                                                         loss_fn)
      updates, optim_state = optim.update(grad, optim_state, controller)
      controller = optax.apply_updates(controller, updates)
      per_step_loss = value / len(tt)
      losses.append(per_step_loss)

      if epoch % print_loss == 0:
        print(f"Epoch: {epoch}\tLoss: {per_step_loss:.2f}")
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
          print(f"Epoch: {epoch}, PIP: {PIP}\tLoss: {per_step_loss:.2f}")
          if tensorboard_dir is not None:
            summary_writer.scalar("per_step_loss", per_step_loss, epoch)
  return controller, per_step_loss
