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

"""Test controller."""
from deluca.lung.core import BreathWaveform
from deluca.lung.utils.data.analyzer import Analyzer
from deluca.lung.utils.scripts.run_controller import run_controller_scan
import jax
import jax.numpy as jnp


@jax.jit
def test_controller(controller, sim, pips, peep):
  """Test controller."""
  # new_controller = controller.replace(use_leaky_clamp=False)
  score = 0.0
  horizon = 29
  for pip in pips:
    waveform = BreathWaveform.create(peep=peep, pip=pip)
    result = run_controller_scan(
        controller,
        T=horizon,
        abort=horizon,
        env=sim,
        waveform=waveform,
        init_controller=True,
    )
    analyzer = Analyzer(result)
    preds = analyzer.pressure  # shape = (29,)
    truth = analyzer.target  # shape = (29,)
    # print('preds.shape: %s', str(preds.shape))
    # print('truth.shape: %s', str(truth.shape))
    score += jnp.abs(preds - truth).mean()
  score = score / len(pips)
  return score
