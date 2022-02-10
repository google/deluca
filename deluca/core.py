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

"""deluca.core.

TODO(dsuo)
- Enforce init over reset

"""
import dataclasses
import inspect
import os
from abc import abstractmethod

import pickle
import flax
import jax

import deluca


def save(obj, path):
  dirname = os.path.abspath(os.path.dirname(path))
  if not os.path.exists(dirname):
    os.makedirs(dirname)

  with open(path, "wb") as file:
    pickle.dump(obj, file)


def load(path):
  return pickle.load(open(path, "rb"))


def field(default=None, jaxed=True, **kwargs):
  if "default_factory" not in kwargs:
    kwargs["default"] = default
  kwargs["pytree_node"] = jaxed
  return flax.struct.field(**kwargs)


class Obj:

  def freeze(self):
    object.__setattr__(self, "__frozen__", True)

  def unfreeze(self):
    object.__setattr__(self, "__frozen__", False)

  def is_frozen(self):
    return not hasattr(self, "__frozen__") or getattr(self, "__frozen__")

  def __new__(cls, *args, **kwargs):
    """A true bastardization of __new__..."""

    def __setattr__(self, name, value):
      if self.is_frozen():
        raise dataclasses.FrozenInstanceError
      object.__setattr__(self, name, value)

    def replace(self, **updates):
      obj = dataclasses.replace(self, **updates)
      obj.freeze()
      return obj

    cls.__setattr__ = __setattr__
    cls.replace = replace

    obj = object.__new__(cls)
    obj.unfreeze()

    return obj

  @classmethod
  def __init_subclass__(cls, *args, **kwargs):
    flax.struct.dataclass(cls)

  @classmethod
  def create(cls, *args, **kwargs):
    # NOTE: Oh boy, this is so janky
    obj = cls(*args, **kwargs)
    obj.setup()
    obj.freeze()

    return obj

  @classmethod
  def unflatten(cls, treedef, leaves):
    """Expost a default unflatten method"""
    return jax.tree_util.tree_unflatten(treedef, leaves)

  def setup(self):
    """Used in place of __init__"""

  def flatten(self):
    """Expose a default flatten method"""
    return jax.tree_util.tree_flatten(self)[0]


class Env(Obj):

  @abstractmethod
  def init(self):
    """Return an initialized state"""

  @abstractmethod
  def __call__(self, state, action, *args, **kwargs):
    """Return an updated state"""


class AgentState(Obj):
  time: float = float("inf")
  steps: int = 0


class Agent(Obj):

  @abstractmethod
  def __call__(self, state, obs, *args, **kwargs):
    """Return an updated state"""

  def init(self):
    return AgentState()


deluca.field = field
deluca.Obj = Obj
deluca.Env = Env
deluca.Agent = Agent
deluca.save = save
deluca.load = load
