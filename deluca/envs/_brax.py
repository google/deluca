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

import functools
from typing import List
from brax.envs import create, Env as InternalBraxEnv, State, PipelineEnv
from brax.envs.base import Observation, ObservationSize
from deluca.core import Env
from deluca.core import field


import jax
from jax import Array
import jax.numpy as jnp
from brax.io import html

from IPython.display import HTML

class BraxEnv(Env):
    """Brax."""

    env: PipelineEnv

    def __init__(self, rng: jax.Array, env_name: str, **kwargs):
        self.env = create(env_name, **kwargs) # type: ignore

    def reset(self, rng: jax.Array):
        state = self.env.reset(rng=rng)

        obs = _obs_to_array(state.obs)
        obs = jnp.expand_dims(obs, -1) # Add singleton dimension

        return 0, state, obs

    def __call__(self, t, state, action, rng):
        # Flatten action
        action = jnp.squeeze(action, -1)
        state = self.env.step(state, action)

        obs = _obs_to_array(state.obs)
        obs = jnp.expand_dims(obs, -1) # Add singleton dimension

        cond = jnp.logical_or(state.info['episode_done'], state.done)
        t = jax.lax.cond(cond, lambda _: 0, lambda _: t + 1, operand=None)

        return t, state, obs

    @property
    def action_size(self) -> int:
        return self.env.action_size
    
    @property
    def observation_size(self) -> int:
        return _obs_size_to_int(self.env.observation_size)
      
    def render(self, states: List[State]):
        """
        Returns an HTML string of the rendered environment.
        """
        return html.render(self.env.sys, [s.pipeline_state for s in states]) # type: ignore


def _obs_size_to_int(obs_size: ObservationSize) -> int:
    if isinstance(obs_size, int):
        return obs_size
    else:
        return sum([sum(s) if isinstance(s, tuple) else s for s in obs_size.values()])

def _obs_to_array(obs: Observation) -> Array:
    if isinstance(obs, jax.Array):
        return obs
    else:
        return jnp.concatenate([obs[k] for k in sorted(obs.keys())])