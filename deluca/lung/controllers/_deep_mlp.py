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

"""deep mlp controller."""
import collections.abc
import functools

import deluca.core
from deluca.lung.controllers._expiratory import Expiratory
from deluca.lung.core import BreathWaveform
from deluca.lung.core import Controller
from deluca.lung.core import DEFAULT_DT
from deluca.lung.core import proper_time
from deluca.lung.utils.data.transform import ShiftScaleTransform
from deluca.lung.utils.nn import MLP
import flax.linen as nn
import jax
import jax.numpy as jnp

# pylint: disable=g-bare-generic
# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=pointless-string-statement


class DeepControllerState(deluca.Obj):
  waveform: deluca.Obj  # waveform can change during training
  errs: jnp.array
  fwd_targets: jnp.array
  time: float = float("inf")
  steps: int = 0
  dt: float = DEFAULT_DT


class DeepMlp(Controller):
  """Deep mlp controller."""
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
  u_normalizer: ShiftScaleTransform = deluca.field(jaxed=False)
  p_normalizer: ShiftScaleTransform = deluca.field(jaxed=False)
  model_apply: collections.abc.Callable = deluca.field(jaxed=False)
  use_model_apply_jit: bool = deluca.field(False, jaxed=False)

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
          jnp.ones([self.back_history_len + self.fwd_history_len]))["params"]
    if self.use_model_apply_jit:
      self.model_apply = jax.jit(self.model.apply)
    else:
      self.model_apply = self.model.apply
    # linear feature transform:
    # errs -> [average of last h errs, ..., average of last 2 errs, last err]
    # emulates low-pass filter bank

  # TODO(dsuo): Handle dataclass initialization of jax objects
  def init(self, waveform=None):
    if waveform is None:
      waveform = BreathWaveform.create()
    errs = jnp.array([0.0] * self.back_history_len)
    fwd_targets = jnp.array(
        [waveform.at(t * DEFAULT_DT) for t in range(self.fwd_history_len)])
    state = DeepControllerState(
        errs=errs, fwd_targets=fwd_targets, waveform=waveform)
    return state

  @jax.jit
  def __call__(self, controller_state, obs):
    state, t = obs.predicted_pressure, obs.time
    errs, waveform = controller_state.errs, controller_state.waveform
    fwd_targets = controller_state.fwd_targets
    target = waveform.at(t)
    fwd_t = t + self.fwd_history_len * DEFAULT_DT
    if self.fwd_history_len > 0:
      fwd_target = jax.lax.cond(fwd_t >= self.horizon * DEFAULT_DT,
                                lambda x: fwd_targets[-1],
                                lambda x: waveform.at(fwd_t), None)
    if self.normalize:
      target_normalized = self.p_normalizer(target).squeeze()
      state_normalized = self.p_normalizer(state).squeeze()
      next_errs = jnp.roll(errs, shift=-1)
      next_errs = next_errs.at[-1].set(target_normalized - state_normalized)
      if self.fwd_history_len > 0:
        fwd_target_normalized = self.p_normalizer(fwd_target).squeeze()
        next_fwd_targets = jnp.roll(fwd_targets, shift=-1)
        next_fwd_targets = next_fwd_targets.at[-1].set(fwd_target_normalized)
      else:
        next_fwd_targets = jnp.array([])
    else:
      next_errs = jnp.roll(errs, shift=-1)
      next_errs = next_errs.at[-1].set(target - state)
      if self.fwd_history_len > 0:
        next_fwd_targets = jnp.roll(fwd_targets, shift=-1)
        next_fwd_targets = next_fwd_targets.at[-1].set(fwd_target)
      else:
        next_fwd_targets = jnp.array([])
    controller_state = controller_state.replace(
        errs=next_errs, fwd_targets=next_fwd_targets)
    decay = self.decay(waveform, t)

    def true_func(null_arg):
      trajectory = jnp.hstack([next_errs, next_fwd_targets])
      u_in = self.model_apply({"params": self.params}, trajectory)
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
  """for i in range(len(tt)):

    (controller_state, expiratory_state, sim_state, obs,
     loss), _ = loop_over_tt(
         (controller_state, expiratory_state, sim_state, obs, loss), tt[i])"""
  return loss


@functools.partial(jax.jit, static_argnums=(6,))
def rollout_parallel(controller, sim, tt, use_noise, peep, pips, loss_fn):
  """rollout function parallel version."""
  loss = jnp.array(0.)
  # for pip in pips:
  #   loss = rollout(controller, sim, tt, use_noise, peep, pip, loss_fn, loss)
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
