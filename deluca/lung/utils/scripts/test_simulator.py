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

"""Open loop test for simulator benchmarking."""
import pickle

from deluca.lung.controllers._predestined import Predestined
from deluca.lung.utils.data.analyzer import Analyzer
from deluca.lung.utils.scripts.run_controller import run_controller_scan
from deluca.lung.utils.scripts.train_simulator import map_rollout_over_batch
import jax.numpy as jnp


def rollout(model, data):
  """rollout over data."""
  test_u_ins, test_pressures = data
  t = len(test_u_ins)
  controller = Predestined.create(u_ins=test_u_ins)
  run_data = run_controller_scan(
      controller, T=t, abort=70, use_tqdm=False, env=model)
  analyzer = Analyzer(run_data)
  preds = analyzer.pressure
  truth = test_pressures
  loss = jnp.abs(preds - truth).mean()
  return loss


def test_simulator(sim, dataset, key="test"):
  """open loop test using vmap."""
  # run u_ins, get pressures, check with true trajectory

  if isinstance(dataset, str):
    dataset = pickle.load(open(dataset, "rb"))

  test_summary = {}
  x_test, y_test = dataset.data[key]
  score = map_rollout_over_batch(sim, (x_test, y_test), rollout)
  test_summary["mae"] = score
  return test_summary
