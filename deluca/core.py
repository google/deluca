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
- jax.core.valid_jaxtype checks values, but not types, so we don't have a check
  that state is a jax type
- Explain why dynamics etc are not static methods (why pass self explicitly as
arg?)
- extracting state from the env object is a little messy

QUESTIONS:
- should env/agent be frozen?

"""

import os
import pickle

from abc import abstractmethod

from deluca.utils.dataclass import dataclass
from deluca.utils.dataclass import PyTreeNode

import deluca


def save(obj, path):
    dirname = os.path.abspath(os.path.dirname(path))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load(path):
    return pickle.load(open(path, "rb"))


def field(**kwargs):
    if "default_factory" not in kwargs and "default" not in kwargs:
        kwargs["default"] = None
    return deluca.utils.dataclass.field(**kwargs)


def attr(**kwargs):
    kwargs["pytree_node"] = False
    return field(**kwargs)


class Obj:
    def __new__(cls, *args, **kwargs):
        """A true bastardization of __new__..."""

        def __setattr__(self, name, value):
            if self.__frozen__:
                raise FrozenInstanceError
            object.__setattr__(self, name, value)

        cls.__setattr__ = __setattr__
        obj = object.__new__(cls)
        object.__setattr__(obj, "__frozen__", False)

        return obj

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        dataclass(pytree=False)(cls)

    @classmethod
    def create(cls, *args, **kwargs):
        # TODO: docstrings
        # God this is so janky
        dummy = cls(*args, **kwargs)
        dummy.setup()
        kwargs = {key: getattr(dummy, key) for key in dummy.__dataclass_fields__.keys()}

        obj = cls(**kwargs)
        obj.__frozen__ = True

        return obj

    @abstractmethod
    def __call__(self):
        """Return an updated state"""

    def setup(self):
        """Used in place of __init__"""
        pass


class Env(Obj):
    @abstractmethod
    def init(self):
        """Return an initialized state"""


class Agent(Obj):
    pass


deluca.attr = attr
deluca.field = field
deluca.Env = Env
deluca.Agent = Agent
deluca.PyTreeNode = PyTreeNode
deluca.save = save
deluca.load = load
