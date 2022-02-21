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

"""Tests for deluca-lung."""

from absl.testing import absltest
import chex
from deluca.lung.core import BreathWaveform
from deluca.lung.core import Controller
from deluca.lung.core import LungEnv
from deluca.lung.core import LungEnvState
from deluca.lung.core import proper_time
import jax.numpy as jnp

# pylint: disable=invalid-name


class BreathWaveformTest(chex.TestCase):

  def test_pip(self):
    pip = 10
    waveform = BreathWaveform.create(peep=0, pip=10)

    self.assertEqual(pip, waveform.pip)

  def test_peep(self):
    peep = 50
    waveform = BreathWaveform.create(peep=50, pip=10)

    self.assertEqual(peep, waveform.peep)

  def test_is_in(self):
    waveform = BreathWaveform.create()
    self.assertEqual(waveform.is_in(0.5), True)
    self.assertEqual(waveform.is_in(1.0), True)
    self.assertEqual(waveform.is_in(1.5), False)

  def test_is_ex(self):
    waveform = BreathWaveform.create()
    self.assertEqual(waveform.is_ex(0.5), False)
    self.assertEqual(waveform.is_ex(1.0), False)
    self.assertEqual(waveform.is_ex(1.5), True)

  def test_at(self):
    waveform = BreathWaveform.create()
    self.assertEqual(waveform.at(0.), 35.0)
    self.assertEqual(waveform.at(0.5), 35.0)
    self.assertEqual(waveform.at(1.0), 35.0)
    self.assertEqual(waveform.at(1.25), 20.0)
    self.assertEqual(waveform.at(1.5), 5)
    self.assertEqual(waveform.at(3.0), 35.0)

  def test_elapsed(self):
    waveform = BreathWaveform.create()
    assert jnp.allclose(
        waveform.elapsed(1.3), waveform.elapsed(1.3 + waveform.period))


class LungEnvTest(chex.TestCase):

  def test_time(self):
    lung = LungEnv()
    state = LungEnvState(
        t_in=0.03,
        steps=1,
        predicted_pressure=0.0,
    )
    self.assertEqual(lung.time(state), 0.03)

  def test_dt(self):
    lung = LungEnv()
    self.assertEqual(lung.dt, 0.03)

  def test_physical(self):
    lung = LungEnv()
    self.assertEqual(lung.physical, False)

  def test_should_abort(self):
    lung = LungEnv()
    self.assertEqual(lung.should_abort(), False)


class proper_timeTest(chex.TestCase):

  def test_proper_time(self):
    self.assertEqual(proper_time(float('inf')), 0)
    self.assertEqual(proper_time(10.0), 10.0)


class ControllerTest(chex.TestCase):

  def test_init(self):
    controller = Controller()
    controller_state = controller.init()
    self.assertEqual(controller_state.time, float('inf'))
    self.assertEqual(controller_state.steps, 0)
    self.assertEqual(controller_state.dt, 0.03)

  def test_max(self):
    controller = Controller()
    self.assertEqual(controller.max, 100)

  def test_min(self):
    controller = Controller()
    self.assertEqual(controller.min, 0)

  def test_decay(self):
    controller = Controller()
    waveform = BreathWaveform.create()
    self.assertEqual(controller.decay(waveform, 0.5), float('inf'))
    self.assertEqual(controller.decay(waveform, 1.0), 0.0)
    self.assertEqual(
        controller.decay(waveform, 1.5),
        5 * (1 - jnp.exp(5 * (waveform.xp[2] - waveform.elapsed(1.5)))))


if __name__ == '__main__':
  absltest.main()
