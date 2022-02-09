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

"""functions to train controller."""
import copy
import functools

from clu import metric_writers
from deluca.lung.controllers._expiratory import Expiratory
from deluca.lung.core import BreathWaveform
from deluca.lung.utils.scripts.test_controller import test_controller
# from flax.metrics import tensorboard
import jax
import jax.numpy as jnp
import optax

# pylint: disable=invalid-name
# pylint: disable=dangerous-default-value


@functools.partial(jax.jit, static_argnums=(6,))
def rollout(controller, sim, tt, use_noise, peep, pip, loss_fn, loss):
  """rollout function."""
  waveform = BreathWaveform.create(peep=peep, pip=pip)
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
  # loss = jnp.array(0.)
  # for pip in pips:
  #   loss = rollout(controller, sim, tt, use_noise, peep, pip, loss_fn, loss)
  # return loss
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
  # print('losses:' + str(losses))
  loss = losses.sum() / float(len(pips))
  return loss


def train_controller(
    controller,
    sim,
    pip_feed="parallel",  # or "sequential"
    mode="multipip",  # or "singular"
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
    model_parameters={},  # used for tensorboard
    print_loss=1,
):
  """train controller."""
  peep = 5
  if mode == "multipip":
    pips = [10, 15, 20, 25, 30, 35]
  elif mode == "singular":
    pips = [35]

  # setup optimizer
  optim_params = copy.deepcopy(optimizer_params)
  if scheduler == "Cosine":
    if pip_feed == "parallel":
      steps_per_epoch = 1
    elif pip_feed == "sequential":
      steps_per_epoch = len(pips)
    decay_steps = int(epochs * steps_per_epoch)
    print("steps_per_epoch:" + str(steps_per_epoch))
    print("decay_steps:" + str(decay_steps))
    cosine_scheduler_fn = optax.cosine_decay_schedule(
        init_value=optim_params["learning_rate"], decay_steps=decay_steps)
    optim_params["learning_rate"] = cosine_scheduler_fn
    print("optim_params:" + str(optim_params))
    optim = optimizer(**optim_params)
  optim_state = optim.init(controller)

  # setup Tensorboard writer
  if tensorboard_dir is not None:
    trial_name = str(model_parameters)
    write_path = tensorboard_dir + trial_name
    summary_writer = metric_writers.create_default_writer(
        logdir=write_path, just_logging=jax.process_index() != 0)
    # summary_writer = tensorboard.SummaryWriter(write_path)
    summary_writer.write_hparams(model_parameters)

  tt = jnp.linspace(0, duration, int(duration / dt))
  losses = []
  for epoch in range(epochs):
    if pip_feed == "parallel":
      value, grad = jax.value_and_grad(rollout_parallel)(controller, sim, tt,
                                                         use_noise, peep,
                                                         jnp.array(pips),
                                                         loss_fn)
      updates, optim_state = optim.update(grad, optim_state, controller)
      controller = optax.apply_updates(controller, updates)
      per_step_loss = value / len(tt)
      losses.append(per_step_loss)
      if epoch % print_loss == 0:
        # make new controller with trained parameters and normal clamp
        score = test_controller(controller, sim, pips, peep)
        print(f"Epoch: {epoch}\tLoss: {score:.2f}")
        if tensorboard_dir is not None:
          summary_writer.write_scalars(epoch, {"score": score})
    if pip_feed == "sequential":
      for pip in pips:
        value, grad = jax.value_and_grad(rollout)(controller, sim, tt,
                                                  use_noise, peep, pip,
                                                  loss_fn, jnp.array(0.))
        updates, optim_state = optim.update(grad, optim_state, controller)
        controller = optax.apply_updates(controller, updates)
        per_step_loss = value / len(tt)
        losses.append(per_step_loss)
        if epoch % print_loss == 0:
          # make new controller with trained parameters and normal clamp
          score = test_controller(controller, sim, pips, peep)
          print(f"Epoch: {epoch}, pip: {pip}\tLoss: {score:.2f}")
          if tensorboard_dir is not None:
            summary_writer.write_scalars(epoch, {"per_step_loss": score})
  return controller, per_step_loss, score
