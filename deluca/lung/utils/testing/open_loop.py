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

from deluca.lung.utils.data.analyzer import Analyzer
from deluca.lung.utils.data.munger import Munger
from deluca.lung.utils.scripts.run_controller import run_controller
from deluca.lung.utils.scripts.run_controller import run_controller_scan

import pickle


def open_loop_test(sim,
                   munger,
                   key="test",
                   abort=70,
                   use_tqdm=False,
                   scan=False,
                   **kwargs):
  # run u_ins, get pressures, check with true trajectory

  if isinstance(munger, str):
    munger = pickle.load(open(munger, "rb"))

  test_summary = {}

  all_runs = []

  for test_trajectory in munger.splits[key]:
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

    all_runs.append((preds, truth))

  test_summary["all_runs"] = all_runs
  populate_resids(test_summary)

  return test_summary
