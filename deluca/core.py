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
"""
TODO:
- attrs and init_subclass don't play well together
    - We do all this mumbo jumbo with interfaces and setting methods in a bizarre way instead of just using inheritance
- Don't be lazy with globals()
-
"""

import os
import pickle
import inspect
import jax
import attr

import deluca
import zope.interface
from zope.interface.verify import verifyClass


class IEnv(zope.interface.Interface):
    """Interface describing deluca environments"""

    def reset():
        """Resets the environment"""

    def dynamics(action):
        """Describes the environment dynamics and returns a new environment"""

    def cost(action):
        """Computes the cost of an action"""


class IAgent(zope.interface.Interface):
    """Interface describing deluca agents"""

    def reset():
        """Resets the agent"""

    def action(observation):
        """Returns the action given an observation"""

    def update(reward):
        """Updates the agent with the reward"""


def evolve(self, **changes):
    return attr.evolve(self, **changes)


@property
def name(self):
    return self.__class__.__name__


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


def flatten(self):
    data_fields = []
    meta_fields = []

    for attribute in attr.fields(self.__class__):
        if not jax.core.valid_jaxtype(attribute.default) or (
            "state" in attribute.metadata and not attribute.metadata["state"]
        ):
            meta_fields.append(attribute.name)
        else:
            data_fields.append(attribute.name)

    return tuple(getattr(self, name) for name in data_fields), {
        "class": self.__class__,
        "meta": tuple(getattr(self, name) for name in meta_fields),
    }


def unflatten(aux, children):
    return aux["class"](*(children + aux["meta"]))


def reset(self):
    return self.__class__()


required_fns = [
    "evolve",
    "name",
    "save",
    "load",
    "throw",
    "flatten",
    "unflatten",
]
overrideable_fns = ["reset"]


def _make_and_check_class(iface, cls):
    for name in required_fns:
        # WARNING: This is bad. We should just not be lazy.
        setattr(cls, name, globals()[name])

    for name in overrideable_fns:
        if not hasattr(cls, name):
            # WARNING: This is bad. We should just not be lazy.
            setattr(cls, name, globals()[name])

    cls = attr.s(frozen=True)(cls)

    cls = zope.interface.implementer(iface)(cls)
    verifyClass(iface, cls)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)

    return cls


def env(cls: type):
    return _make_and_check_class(IEnv, cls)


def agent(cls: type):
    return _make_and_check_class(IAgent, cls)


deluca.env = env
deluca.agent = agent
deluca.IEnv = IEnv
deluca.IAgent = IAgent
