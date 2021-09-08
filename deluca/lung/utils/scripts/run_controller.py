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

import os
import time
from functools import partial
import pickle

import numpy as np
import jax
import jax.numpy as jnp

from deluca.lung.controllers._expiratory import Expiratory
from deluca.lung.envs._balloon_lung import BalloonLung
from deluca.lung.core import BreathWaveform


def run_controller(
    controller,
    R,
    C,
    T=1000,
    dt=0.03,
    abort=60,
    env=None,
    waveform=None,
    use_tqdm=False,
    directory=None,
):
  timers = {
    "init": 0.,
    "in_ctrl": 0.,
    "ex_ctrl": 0.,
    "env": 0.,
    "record": 0.,
    "loop": 0.,
    "total": 0.
  }
  
  start = time.time()
  overall_start = start
  env = env or BalloonLung()
  waveform = waveform or BreathWaveform.create()
  expiratory = Expiratory.create()

  result = locals()

  controller_state = controller.init()
  expiratory_state = expiratory.init()

  tt = range(T)
  if use_tqdm:
    tt = tqdm.tqdm(tt, leave=False)

  timestamps = np.zeros(T)
  pressures = np.zeros(T)
  flows = np.zeros(T)
  u_ins = np.zeros(T)
  u_outs = np.zeros(T)

  state = env.init()

  timers["init"] += time.time() - start

  loop_start = time.time()

  try:
    for i, _ in enumerate(tt):
      pressure = state.pressure
      if env.should_abort():
        break

      timestamps[i] = state.time

      start = time.time()
      controller_state, u_in = controller(controller_state, state)
      timers["in_ctrl"] += time.time() - start

      start = time.time()
      expiratory_state, u_out = expiratory(expiratory_state, state)
      timers["ex_ctrl"] += time.time() - start

      start = time.time()
      state, _ = env(state, (u_in, u_out))
      timers["env"] += time.time() - start

      start = time.time()
      u_ins[i] = u_in.squeeze().item()
      # u_outs[i] = u_out.squeeze().item()
      u_outs[i] = 0
      pressures[i] = state.pressure
      flows[i] = state.flow
      timers["record"] += time.time() - start

      state = env.wait(state, max(dt - state.dt, 0))

  finally:
    env.cleanup()

  timers["loop"] += time.time() - loop_start

  timeseries = {
      "timestamp": jnp.array(timestamps),
      "pressure": jnp.array(pressures),
      "flow": jnp.array(flows),
      "target": waveform.at(timestamps),
      "u_in": jnp.array(u_ins),
      "u_out": jnp.array(u_outs),
  }

  for key, val in timeseries.items():
    timeseries[key] = val[:T + 1]

  result["timeseries"] = timeseries

  if directory is not None:
    if not os.path.exists(directory):
      os.makedirs(directory)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    pickle.dump(result, open(f"{directory}/{timestamp}.pkl", "wb"))

  timers["total"] += time.time() - overall_start

  return result, timers

def loop_over_tt(envState_obs_ctrlState_ExpState_i, dummy_data, controller,
                 expiratory, env, dt):
  state, obs, controller_state, expiratory_state, i = envState_obs_ctrlState_ExpState_i
  pressure = obs.predicted_pressure

  # if env.should_abort(): # TODO: how to handle break in scan
  #     break

  controller_state, u_in = controller.__call__(controller_state, obs)
  expiratory_state, u_out = expiratory.__call__(expiratory_state, obs)

  state, obs = env(state, (u_in, u_out))

  timestamps_i = env.time(state) - dt
  u_ins_i = u_in
  u_outs_i = u_out
  pressures_i = pressure
  flows_i = env.flow

  env.wait(max(dt - env.dt, 0))
  return (state, obs, controller_state, expiratory_state,
          i + 1), (timestamps_i, u_ins_i, u_outs_i, pressures_i, flows_i)

def run_controller_scan(
    controller,
    R,
    C,
    T=1000,
    dt=0.03,
    abort=60,
    env=None,
    waveform=None,
    use_tqdm=False,
    directory=None,
    init_controller=False,
):
  env = env or BalloonLung()
  waveform = waveform or BreathWaveform.create()
  expiratory = Expiratory.create(waveform=waveform)

  result = locals()

  if init_controller:
    controller_state = controller.init(waveform)
  else:
    controller_state = controller.init()
  expiratory_state = expiratory.init()

  tt = range(T)
  if use_tqdm:
    tt = tqdm.tqdm(tt, leave=False)

  timestamps = jnp.zeros(T)
  pressures = jnp.zeros(T)
  flows = jnp.zeros(T)
  u_ins = jnp.zeros(T)
  u_outs = jnp.zeros(T)

  state, obs = env.reset()
  # xp = jnp.array(waveform.xp)
  # fp = jnp.array(waveform.fp)
  # period = waveform.period
  # dtype = waveform.dtype

  jit_loop_over_tt = jax.jit(partial(loop_over_tt,
                                     controller=controller,
                                     expiratory=expiratory,
                                     env=env,
                                     dt=dt))
  try:
    _, (timestamps, u_ins, u_outs, pressures, flows) = jax.lax.scan(
        jit_loop_over_tt, (state, obs, controller_state, expiratory_state, 0),
        jnp.arange(T))

  finally:
    env.cleanup()

  timeseries = {
      "timestamp": jnp.array(timestamps),
      "pressure": jnp.array(pressures),
      "flow": jnp.array(flows),
      "target": waveform.at(timestamps),
      "u_in": jnp.array(u_ins),
      "u_out": jnp.array(u_outs),
  }

  for key, val in timeseries.items():
    timeseries[key] = val[:T + 1]

  result["timeseries"] = timeseries

  if directory is not None:
    if not os.path.exists(directory):
      os.makedirs(directory)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    pickle.dump(result, open(f"{directory}/{timestamp}.pkl", "wb"))

  return result
