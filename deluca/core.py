# Copyright 2021 Google LLC
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
import os
import pickle
import logging

import gym


class Entity:
    @property
    def name(self):
        return self.__class__.__name__

    def __str__(self):
        return self.name

    def save(self, path):
        dirname = os.path.abspath(os.path.dirname(path))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        return pickle.load(open(path, "rb"))

    def throw(self, err, msg):
        raise err(f"Class {self.name}: {msg}")


EnvRegistry = {}

logger = logging.getLogger(__name__)


def make_env(cls, *args, **kwargs):
    if cls not in EnvRegistry:
        raise ValueError(f"Env `{cls}` not found.")
    return EnvRegistry[cls](*args, **kwargs)


class Env(Entity, gym.core.Env):
    # Set this in SOME subclasses
    metadata = {"render.modes": []}
    reward_range = (-float("inf"), float("inf"))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def __new__(cls, *args, **kwargs):
        """For avoiding super().__init__()"""
        obj = super().__new__(cls, *args, **kwargs)

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
    def state(self):
        raise NotImplementedError()

    @property
    def observation(self):
        """
        NOTE: assume observations are fully observable
        """
        return self.state

    def reward_fn(self, state, action):
        return 0

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

        return self.observation, reward, False, {}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self):
        return

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False

    def render(self):
        raise NotImplementedError()


AgentRegistry = {}

logger = logging.getLogger(__name__)


def make_agent(cls, *args, **kwargs):
    if cls not in AgentRegistry:
        raise ValueError(f"Agent `{cls}` not found.")
    return AgentRegistry[cls](*args, **kwargs)


class Agent(Entity):
    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        """For avoiding a decorator for each subclass"""
        super().__init_subclass__(*args, **kwargs)
        if cls.__name__ not in AgentRegistry and not inspect.isabstract(cls):
            AgentRegistry[cls.__name__] = cls

    def __call__(self, observation):
        raise NotImplementedError()

    def reset(self):
        logger.warn(f"`reset` for agent `{self.name}` not implemented")

    def update(self, reward):
        logger.warn(f"`update` for agent `{self.name}` not implemented")
