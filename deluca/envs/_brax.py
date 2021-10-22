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

"""Wrapper for Brax systems."""

import brax
from brax import envs as brax_envs
from brax.io import html
from deluca.core import Env
from deluca.core import field
from IPython.display import HTML

# pylint:disable=protected-access

class BraxEnv(Env):
  """Brax."""
  sys: brax.System = field(jaxed=False)
  env: brax_envs.Env = field(jaxed=False)

  def setup(self):
    """setup."""
    if self.env is not None:
      self.sys = self.env.sys
    elif isinstance(self.sys, brax.physics.config_pb2.Config):
      self.sys = brax.System(self.sys)

  def init(self):
    """init.

    Returns:

    """
    state = self.sys.default_qp()
    obs = state if self.env is None else self.env._get_obs(
        state, self.sys.info(state))
    return state, obs

  def reset(self):
    return self.init()

  def __call__(self, state, action):
    """__call__.

    Args:
      state:
      action:

    Returns:

    """
    state, info = self.sys.step(state, action)
    obs = state if self.env is None else self.env._get_obs(
        state, info)
    return state, obs

  def render(self, states):
    """render.

    Args:
      states:

    Returns:

    """
    return HTML(html.render(self.sys, states))
