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

"""Open loop test for simulator benchmarking."""
import functools
import multiprocessing
import pickle

from absl import logging
from deluca.lung.controllers._predestined import Predestined
from deluca.lung.utils.data.analyzer import Analyzer
from deluca.lung.utils.scripts.run_controller import run_controller
from deluca.lung.utils.scripts.run_controller import run_controller_scan
import jax
import jax.numpy as jnp
import tqdm
# pylint: disable=invalid-name


def map_rollout_over_batch(sim, data, rollout_fn):
  """function to map rollout over batch dimension.

  Args:
       sim: Simulator
       data: has shape = ((batch_size, N), (batch_size, N))
       rollout_fn: has signature (model, data) -> loss where data.shape = (2, N)

  Returns:
    loss
  """
  rollout_partial = lambda xs: functools.partial(rollout_fn, sim=sim)(data=xs)
  data_zipped = jnp.array(list(zip(data[0], data[1])))  # (batch_size, 2, N)
  losses = jax.vmap(rollout_partial)(data_zipped)
  return jnp.array(losses).mean()


def rollout(sim, data):
  """rollout over data."""
  test_u_ins, test_pressures = data
  t = len(test_u_ins)
  controller = Predestined.create(u_ins=test_u_ins)
  # lo = jnp.min(test_pressures)
  # hi = jnp.max(test_pressures)
  # custom_range=(lo, hi)
  # print('custom_range:' + str(custom_range))
  # waveform = BreathWaveform.create(custom_range=custom_range)
  run_data = run_controller_scan(
      controller, R=None, C=None, T=t, abort=70, use_tqdm=False, env=sim)
  analyzer = Analyzer(run_data)
  preds = analyzer.pressure
  truth = test_pressures
  loss = jnp.abs(preds - truth).mean()
  return loss


def open_loop_test_vmap(sim, munger, key="test"):
  """open loop test using vmap."""
  # run u_ins, get pressures, check with true trajectory

  if isinstance(munger, str):
    munger = pickle.load(open(munger, "rb"))

  test_summary = {}
  # all_runs = []
  X_test, y_test = munger.get_scaled_data(key, scale_data=False)
  score = map_rollout_over_batch(sim, (X_test, y_test), rollout)
  test_summary["mae"] = score
  return test_summary


def open_loop_test(sim,
                   munger,
                   key="test",
                   abort=70,
                   use_tqdm=False,
                   scan=False,
                   **kwargs):
  """open loop test."""
  # run u_ins, get pressures, check with true trajectory

  if isinstance(munger, str):
    munger = pickle.load(open(munger, "rb"))

  test_summary = {}

  # all_runs = []
  def loop_over_trajectory(munger_trajectories):
    # for test_trajectory in tqdm.tqdm(munger.splits[key]):
    runs = []
    for test_trajectory in tqdm.tqdm(munger_trajectories):
      test_u_ins, test_pressures = test_trajectory
      T = len(test_u_ins)
      controller = Predestined.create(u_ins=test_u_ins)
      if not scan:
        run_data = run_controller(
            controller,
            R=None,
            C=None,
            T=T,
            abort=abort,
            use_tqdm=use_tqdm,
            env=sim,
            **kwargs)
      else:
        run_data = run_controller_scan(
            controller,
            R=None,
            C=None,
            T=T,
            abort=abort,
            use_tqdm=use_tqdm,
            env=sim,
            **kwargs)

      analyzer = Analyzer(run_data)

      # pedantic copying + self-documenting names...
      preds = analyzer.pressure.copy()
      truth = test_pressures.copy()
      runs.append((preds, truth))
    return runs

  logging.info("multiprocessing.cpu_count(): %s",
               str(multiprocessing.cpu_count()))
  cpu_count = multiprocessing.cpu_count()
  pool = multiprocessing.pool.ThreadPool(cpu_count)
  chunk_size = int(len(munger.splits[key]) / cpu_count)
  logging.info("chunk_size: %s", str(chunk_size))
  all_trajectories = munger.splits[key]
  all_trajectories_split = [
      all_trajectories[x:x + chunk_size]
      for x in range(0, len(all_trajectories), chunk_size)
  ]
  sum_of_splits = sum([len(chunk) for chunk in all_trajectories_split])
  assert sum_of_splits == len(all_trajectories)

  def flatten(t):
    flat_list = []
    for sublist in t:
      for item in sublist:
        flat_list.append(item)
    return flat_list

  all_runs = flatten(
      list(pool.map(loop_over_trajectory, all_trajectories_split)))
  pool.close()
  pool.join()
  assert len(all_runs) == len(all_trajectories)

  test_summary["all_runs"] = all_runs
  populate_resids(test_summary)

  return test_summary


def populate_resids(test_summary, key="all_runs", l1_out="mae", l2_out="rmse"):
  """compute loss."""
  all_runs = test_summary[key]

  Tmax = max(map(lambda x: len(x[0]), all_runs))
  l1, l2, counts = jnp.zeros(Tmax), jnp.zeros(Tmax), jnp.zeros(Tmax)
  logging.info("ENTER POPULATE_RESIDS")
  logging.info("Tmax: %s", str(Tmax))
  logging.info("all_runs[:2]: %s", str(all_runs[:2]))

  for preds, truth in all_runs:
    resids = truth - preds
    T = len(resids)
    assert T == Tmax
    l1 = jax.ops.index_update(l1, jax.ops.index[:T], l1[:T] + abs(resids))
    l2 = jax.ops.index_update(l2, jax.ops.index[:T], l2[:T] + resids**2)
    counts = jax.ops.index_update(counts, jax.ops.index[:T], counts[:T] + 1)

  test_summary[l1_out] = l1 / counts
  test_summary[l2_out] = (l2 / counts)**0.5
