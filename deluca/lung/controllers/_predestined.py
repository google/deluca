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

import jax
import jax.numpy as jnp
import deluca
from functools import partial
from typing import List, Callable
from deluca.lung.core import Controller
from deluca.lung.core import DEFAULT_DT
from deluca.lung.core import proper_time


class Predestined(Controller):
  time: float = deluca.field(trainable=False)
  steps: int = deluca.field(trainable=False)
  dt: float = deluca.field(trainable=False)
  u_ins: jnp.array = deluca.field(trainable=False)
  switch_funcs: List[Callable] = deluca.field(trainable=False)

  def setup(self):
    self.switch_funcs = [
        partial(lambda x, y: self.u_ins[y], y=i)
        for i in range(len(self.u_ins))
    ]

  def __call__(self, state, obs, *args, **kwargs):
    # TODO: We should probably use jax.lax.dynamic_slice
    action = jax.lax.switch(state.steps, self.switch_funcs, None)
    time = obs.time
    new_dt = jnp.max(jnp.array([DEFAULT_DT, time - proper_time(state.time)]))
    new_time = time
    new_steps = state.steps + 1
    state = state.replace(time=new_time, steps=new_steps, dt=new_dt)
    return state, action
