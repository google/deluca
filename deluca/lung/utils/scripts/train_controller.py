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

from deluca.lung.controllers import Expiratory
from deluca.lung.core import BreathWaveform
import jax
import jax.numpy as jnp
import optax


def rollout(controller, sim, waveforms, tt, use_noise, loss_fn, mean, std,
            loss):
  for waveform in waveforms:
    expiratory = Expiratory.create(waveform=waveform)
    controller_state = controller.init()
    expiratory_state = expiratory.init()
    sim_state, obs = sim.reset()

    def loop_over_tt(ctrlState_expState_simState_obs_loss, t):
      controller_state, expiratory_state, sim_state, obs, loss = ctrlState_expState_simState_obs_loss
      noise = mean + std * jax.random.normal(jax.random.PRNGKey(0), shape=())
      pressure = sim_state.predicted_pressure + use_noise * noise
      sim_state = sim_state.replace(
          predicted_pressure=pressure)  # update p_history as well or no?
      obs = obs.replace(predicted_pressure=pressure, time=t)

      controller_state, u_in = controller(controller_state, obs)
      expiratory_state, u_out = expiratory(expiratory_state, obs)

      sim_state, obs = sim(sim_state, (u_in, u_out))

      loss = jax.lax.cond(
          u_out == 0,
          lambda x: x + loss_fn(jnp.array(waveform.at(t)), pressure),
          lambda x: x, loss)
      return (controller_state, expiratory_state, sim_state, obs, loss), None

    (_, _, _, _, loss), _ = jax.lax.scan(
        loop_over_tt,
        (controller_state, expiratory_state, sim_state, obs, loss), tt)
  return loss


def train(controller, sim, waveforms, tt, use_noise, loss_fn, mean, std, loss,
          optim, optim_state):
  value, grad = jax.value_and_grad(rollout)(controller, sim, waveforms, tt,
                                            use_noise, loss_fn, mean, std, loss)
  updates, optim_state = optim.update(grad, optim_state, controller)
  controller = optax.apply_updates(controller, updates)
  per_step_loss = value / len(tt)
  return per_step_loss, optim, optim_state, controller


def train_controller(
    controller,
    sim,
    waveform=None,
    pip_feed="sequential",
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

  if waveform is not None:
    train_mode = "singular"
    mean, std = 1.0, 1.0
  else:
    train_mode = "multipip"
    mean, std = 1.5, 1.0
    PIPs = [10, 15, 20, 25, 30, 35]
    PEEP = 5
    random_PIPs = random.sample(PIPs, 6)

  # torch.autograd.set_detect_anomaly(True)
  for epoch in range(epochs):
    if train_mode == "singular" or (train_mode == "multipip" and
                                    pip_feed == "parallel"):
      if train_mode == "singular":
        waveforms = [waveform]
      else:
        waveforms = [
            BreathWaveform.create(custom_range=(PEEP, PIP))
            for PIP in random_PIPs
        ]
      loss = jnp.array(0.)
      per_step_loss, optim, optim_state, controller = train(
          controller, sim, waveforms, tt, use_noise, loss_fn, mean, std, loss,
          optim, optim_state)
      losses.append(per_step_loss)
      if epoch % print_loss == 0:
        print(f"Epoch: {epoch}\tLoss: {per_step_loss:.2f}")
    elif train_mode == "multipip" and pip_feed == "sequential":
      for PIP in random_PIPs:
        waveforms = [BreathWaveform.create(custom_range=(PEEP, PIP))]
        loss = jnp.array(0.)
        per_step_loss, optim, optim_state, controller = train(
            controller, sim, waveforms, tt, use_noise, loss_fn, mean, std, loss,
            optim, optim_state)
        losses.append(per_step_loss)
        if epoch % print_loss == 0:
          print(f"Epoch: {epoch}, PIP: {PIP}\tLoss: {per_step_loss:.2f}")
  return controller
