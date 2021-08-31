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
from deluca.lung.core import Controller
from deluca.lung.core import ControllerState
from deluca.lung.core import BreathWaveform
from deluca.lung.controllers._expiratory import Expiratory
from deluca.lung.envs._stitched_sim import StitchedSimObservation
from deluca.lung.utils.data.transform import ShiftScaleTransform
from deluca.lung.core import DEFAULT_DT
from deluca.lung.core import proper_time


class DeepNetwork(nn.Module):
  H: int = 100
  kernel_size: int = 5
  out_dim: int = 1

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(
        features=self.H, kernel_size=self.kernel_size, name=f"deep_conv")(
            x)
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


class Deep(Controller):
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

  # bptt: int = deluca.field(1, jaxed=False) not used right now
  # TODO: add analogue of activation=torch.nn.ReLU

  def setup(self):
    self.model = DeepNetwork(H=self.H, kernel_size=self.kernel_size, out_dim=1)
    if self.params is None:
      self.params = self.model.init(
          jax.random.PRNGKey(0),
          jnp.expand_dims(jnp.ones([self.history_len]), axis=(0, 1)))["params"]

    # linear feature transform:
    # errs -> [average of last h errs, ..., average of last 2 errs, last err]
    # emulates low-pass filter bank

    self.featurizer = jnp.tril(jnp.ones((self.history_len, self.history_len)))
    self.featurizer /= jnp.expand_dims(
        jnp.arange(self.history_len, 0, -1), axis=0)

    if self.normalize:
      self.u_scaler = u_scaler
      self.p_scaler = p_scaler

  # TODO: Handle dataclass initialization of jax objects
  def init(self, waveform=None):
    if waveform is None:
      waveform = BreathWaveform.create()
    errs = jnp.array([0.0] * self.history_len)
    state = DeepControllerState(errs=errs, waveform=waveform)
    return state

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
      u_in = self.model.apply({"params": self.params},
                              (trajectory @ self.featurizer))
      return u_in.squeeze().astype(jnp.float32)

    # changed decay compare from None to float(inf) due to cond requirements
    u_in = jax.lax.cond(
        jnp.isinf(decay), true_func, lambda x: jnp.array(decay), None)

    u_in = jax.lax.clamp(0.0, u_in.astype(jnp.float32), self.clip).squeeze()

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
        predicted_pressure=pressure)  # Need to update p_history as well or no?
    obs = obs.replace(predicted_pressure=pressure, time=t)

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


def rollout_parallel(controller, sim, tt, use_noise, PEEP, PIPs, loss_fn):
  loss = jnp.array(0.)
  for PIP in PIPs:
    loss = rollout(controller, sim, tt, use_noise, PEEP, PIP, loss_fn, loss)
  return loss


# TODO: add scheduler and scheduler_params
# Question: Jax analogue of torch.autograd.set_detect_anomaly(True)?
def deep_train(
    controller,
    sim,
    pip_feed="parallel",
    duration=3,
    dt=0.03,
    epochs=100,
    use_noise=False,
    optimizer=optax.adamw,
    optimizer_params={
        "learning_rate": 1e-3,
        "weight_decay": 1e-4
    },
    loss_fn=lambda x, y: (jnp.abs(x - y)).mean(),
    # scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    # scheduler_params={"factor": 0.9, "patience": 10},
    print_loss=1,
):
  optim = optimizer(**optimizer_params)
  optim_state = optim.init(controller)

  tt = jnp.linspace(0, duration, int(duration / dt))
  losses = []

  # torch.autograd.set_detect_anomaly(True)

  PIPs = [10, 15, 20, 25, 30, 35]
  PEEP = 5

  # TODO: handle device-awareness
  for epoch in range(epochs):

    if pip_feed == "parallel":
      value, grad = jax.value_and_grad(rollout_parallel)(controller, sim, tt,
                                                         use_noise, PEEP, PIPs,
                                                         loss_fn)
      updates, optim_state = optim.update(grad, optim_state, controller)
      controller = optax.apply_updates(controller, updates)
      per_step_loss = value / len(tt)
      losses.append(per_step_loss)

      if epoch % print_loss == 0:
        print(f"Epoch: {epoch}\tLoss: {per_step_loss:.2f}")

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
  return controller
