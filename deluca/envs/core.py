# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import logging

import gym
import jax
import jax.numpy as jnp

from deluca.core import JaxObject

EnvRegistry = {}

logger = logging.getLogger(__name__)


def default_reward(state, action):
    jnp.square(jnp.linalg.norm(state)) + jnp.square(jnp.linalg.norm(action))


def make_env(cls, *args, **kwargs):
    if cls not in EnvRegistry:
        raise ValueError(f"Env `{cls}` not found.")
    return EnvRegistry[cls](*args, **kwargs)


class Env(JaxObject, gym.Env):
    reward_range = (-float("inf"), float("inf"))
    action_space = None
    observation_space = None

    def __new__(cls, *args, **kwargs):
        """For avoiding super().__init__()"""
        obj = super().__new__(cls, *args, **kwargs)

        obj.__setattr__("reward_fn", default_reward)

        for kw, arg in kwargs.items():
            obj.__setattr__(kw, arg)

        return obj

    @classmethod
    def check_spaces(cls):
        if cls.action_space is None:
            raise AttributeError(f"Env `{cls.__name__}` missing action_space.")
        if cls.observation_space is None:
            raise AttributeError(f"Env `{cls.__name__}` missing observation_space.")

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        """For avoiding a decorator for each subclass"""
        super().__init_subclass__(*args, **kwargs)
        if cls.__name__ not in EnvRegistry and not inspect.isabstract(cls):
            EnvRegistry[cls.__name__] = cls

    @property
    def observation(self):
        """
        NOTE: assume observations are fully observable
        """
        return self.state

    def reset(self):
        logger.warn(f"`reset` for agent `{self.name}` not implemented")

    def dynamics(self, state, action):
        return state

    def check_action(self, action):
        # TODO: Check performance penalty of this check
        if not self.action_space.contains(action):
            raise ValueError(f"Env `{self.name}` does not contain {action} in its action space.")

    def check_observation(self, observation):
        # TODO: Check performance penalty of this check
        if not self.observation_space.contains(observation):
            raise ValueError(
                f"Env `{self.name}` does not contain {self.state} in its observation space."
            )

    def step(self, action):
        reward = self.reward_fn(self.state, action)
        self.state = self.dynamics(self.state, action)
        self.last_u = action
        return self.observation, reward, False, {}

    def jacobian(self, func, state, action):
        return jax.jacrev(func, argnums=(0, 1))(state, action)

    def hessian(self, func, state, action):
        return jax.hessian(func, argnums=(0, 1))(state, action)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
