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

"""Wrapper for Brax systems."""

from typing import List
import brax
from brax.envs import PipelineEnv
from brax.base import State as BraxState
from brax.io import html, mjcf
from deluca.core import Env
from deluca.core import field
from IPython.display import HTML

import jax

# pylint:disable=protected-access


class BraxEnv(Env):
    """Brax."""

    env: PipelineEnv = field(jaxed=False)
    states: List[BraxState] = field(jaxed=False)

    @classmethod
    def from_env(cls, env: PipelineEnv):
        lenv = cls.create(env=env)
        lenv.unfreeze()
        return lenv

    # @classmethod
    # def from_name(cls, name):
    #   return cls.from_env(PipelineEnv.create(env_name=name))

    # NOTE Seems like Config is no longer supported, you should import from XML.
    # @classmethod
    # def from_config(cls, config):
    #   return cls.create(sys=brax.System(config))

    def setup(self):
        """setup."""
        

    def init(self, rng: jax.Array):
        """init.

        Returns:

        """
        return self.reset(rng)

    def reset(self, rng: jax.Array):
        """reset.

        Returns:

        """
        state = self.env.reset(rng=rng)

        if state.pipeline_state is not None:
            self.states = [state.pipeline_state]
        else:
            self.states = []

        return state, state.obs

    def __call__(self, state, action):
        """__call__.

        Args:
          state:
          action:

        Returns:

        """
        state = self.env.step(state, action)

        if state.pipeline_state is not None:
            self.states.append(state.pipeline_state)

        return state, state.obs
    
    @property
    def action_size(self) -> int:
      return self.env.action_size

    def render(self):
        """render.

        Args:
          states:

        Returns:

        """
        return self.env.render(self.states)
