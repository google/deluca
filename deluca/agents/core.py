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

from deluca.core import JaxObject

AgentRegistry = {}

logger = logging.getLogger(__name__)

# TODO
# - Support hierarchical agents


def make_agent(cls, *args, **kwargs):
    if cls not in AgentRegistry:
        raise ValueError(f"Agent `{cls}` not found.")
    return AgentRegistry[cls](*args, **kwargs)


class Agent(JaxObject):
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

    def feed(self, reward):
        logger.warn(f"`feed` for agent `{self.name}` not implemented")
