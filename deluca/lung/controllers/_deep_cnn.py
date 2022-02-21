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

"""deep cnn controller."""
import collections.abc
import functools

import deluca.core
from deluca.lung.controllers._expiratory import Expiratory
from deluca.lung.core import BreathWaveform
from deluca.lung.core import Controller
from deluca.lung.core import DEFAULT_DT
from deluca.lung.core import proper_time
from deluca.lung.utils.data.transform import ShiftScaleTransform
import flax.linen as nn
import jax
import jax.numpy as jnp


# pylint: disable=g-bare-generic
# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=pointless-string-statement


class DeepNetwork(nn.Module):
  """deep cnn controller network."""
  H: int = 100
  kernel_size: int = 5

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(
        features=self.H,
        kernel_size=(self.kernel_size,),
        padding="VALID",
        strides=(1,),
        name="deep_conv")(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=1, use_bias=True, name="deep_fc")(x)
    return x


class DeepControllerState(deluca.Obj):
  waveform: deluca.Obj  # waveform can change during training
  errs: jnp.array
  time: float = float("inf")
  steps: int = 0
  dt: float = DEFAULT_DT


class DeepCnn(Controller):
  """deep cnn controller."""
  params: list = deluca.field(jaxed=True)
  model: nn.module = deluca.field(DeepNetwork, jaxed=False)
  featurizer: jnp.array = deluca.field(jaxed=False)
  H: int = deluca.field(100, jaxed=False)
  input_dim: int = deluca.field(1, jaxed=False)
  history_len: int = deluca.field(10, jaxed=False)
  kernel_size: int = deluca.field(5, jaxed=False)
  clip: float = deluca.field(100.0, jaxed=False)
  normalize: bool = deluca.field(False, jaxed=False)
  u_normalizer: ShiftScaleTransform = deluca.field(jaxed=False)
  p_normalizer: ShiftScaleTransform = deluca.field(jaxed=False)
  model_apply: collections.abc.Callable = deluca.field(jaxed=False)
  use_model_apply_jit: bool = deluca.field(False, jaxed=False)
  use_leaky_clamp: bool = deluca.field(True, jaxed=False)

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
    # TODO(dsuo): resolve jit load/unload
    if self.use_model_apply_jit:
      self.model_apply = jax.jit(self.model.apply)
    else:
      self.model_apply = self.model.apply

  # TODO(dsuo): Handle dataclass initialization of jax objects
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
      target_normalized = self.p_normalizer(target).squeeze()
      state_normalized = self.p_normalizer(state).squeeze()
      next_errs = jnp.roll(errs, shift=-1)
      next_errs = next_errs.at[-1].set(target_normalized - state_normalized)
    else:
      next_errs = jnp.roll(errs, shift=-1)
      next_errs = next_errs.at[-1].set(target - state)
    controller_state = controller_state.replace(errs=next_errs)
    decay = self.decay(waveform, t)

    def true_func(null_arg):
      trajectory = jnp.expand_dims(next_errs[-self.history_len:], axis=(0, 1))
      input_val = jnp.reshape((trajectory @ self.featurizer),
                              (1, self.history_len, 1))
      u_in = self.model_apply({"params": self.params}, input_val)
      return u_in.squeeze().astype(jnp.float32)

    # changed decay compare from None to float(inf) due to cond requirements
    u_in = jax.lax.cond(
        jnp.isinf(decay), true_func, lambda x: jnp.array(decay), None)
    # Implementing "leaky" clamp to solve the zero gradient problem
    if self.use_leaky_clamp:
      u_in = jax.lax.cond(u_in < 0.0, lambda x: x * 0.01, lambda x: x, u_in)
      u_in = jax.lax.cond(u_in > self.clip, lambda x: self.clip + x * 0.01,
                          lambda x: x, u_in)
    else:
      u_in = jax.lax.clamp(0.0, u_in.astype(jnp.float32), self.clip).squeeze()
    # update controller_state
    new_dt = jnp.max(
        jnp.array([DEFAULT_DT, t - proper_time(controller_state.time)]))
    new_time = t
    new_steps = controller_state.steps + 1
    controller_state = controller_state.replace(
        time=new_time, steps=new_steps, dt=new_dt)
    return controller_state, u_in


def rollout(controller, sim, tt, use_noise, peep, pip, loss_fn, loss):
  """rollout function."""
  waveform = BreathWaveform.create(custom_range=(peep, pip))
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
    sim_state = sim_state.replace(predicted_pressure=pressure)
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
def rollout_parallel(controller, sim, tt, use_noise, peep, pips, loss_fn):
  """rollout function parallel version."""
  """loss = jnp.array(0.)

  for pip in pips:
    loss = rollout(controller, sim, tt, use_noise, peep, pip, loss_fn, loss)
  return loss
  """
  def rollout_partial(p):
    return functools.partial(
        rollout,
        controller=controller,
        sim=sim,
        tt=tt,
        use_noise=use_noise,
        peep=peep,
        loss_fn=loss_fn,
        loss=0.0)(pip=p)

  losses = jax.vmap(rollout_partial)(jnp.array(pips))
  loss = losses.sum() / float(len(pips))
  return loss
