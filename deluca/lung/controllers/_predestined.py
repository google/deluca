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

"""predestined controller."""

import deluca
from deluca.lung.core import Controller
from deluca.lung.core import DEFAULT_DT
from deluca.lung.core import proper_time
import jax
import jax.numpy as jnp


class Predestined(Controller):
  """predestined controller."""
  time: float = deluca.field(jaxed=False)
  steps: int = deluca.field(jaxed=False)
  dt: float = deluca.field(jaxed=False)
  u_ins: jnp.array = deluca.field(jaxed=False)

  def __call__(self, state, obs, *args, **kwargs):
    action = jax.lax.dynamic_slice(self.u_ins, (state.steps.astype(int),), (1,))
    time = obs.time
    new_dt = jnp.max(jnp.array([DEFAULT_DT, time - proper_time(state.time)]))
    new_time = time
    new_steps = state.steps + 1
    state = state.replace(time=new_time, steps=new_steps, dt=new_dt)
    return state, action
