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
import inspect
import jax
from flax.struct import dataclass

import deluca

from abc import abstractmethod


def attr_list_to_dict(obj, attrs):
    return {name: getattr(obj, name) for name in attrs}


def flatten(obj):
    meta = [name for name in dir(obj) if not name.startswith("__")]
    if hasattr(obj, "jax_attrs"):
        data = obj.jax_attrs()
    else:
        data = [name for name in meta if not callable(getattr(obj, name))]

    meta = list(set(meta) - set(data))

    children, treedef = jax.tree_util.tree_flatten(attr_list_to_dict(obj, data))

    aux = {
        "class": obj.__class__,
        "treedef": treedef,
        "args": obj.__args__,
        "meta": attr_list_to_dict(obj, meta),
    }

    return children, aux


def unflatten(aux, children):
    obj = aux["class"](*aux["args"].args, **aux["args"].kwargs)
    data = jax.tree_util.tree_unflatten(aux["treedef"], children)
    for key, val in data.items():
        object.__setattr__(obj, key, val)
    for key, val in aux["meta"]:
        object.__setattr__(obj, key, val)
    return obj


def register_pytree_node(cls):
    jax.tree_util.register_pytree_node(cls, flatten, unflatten)

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        object.__setattr__(obj, "__args__", inspect.signature(obj.__init__).bind(*args, **kwargs))
        object.__getattribute__(obj, "__args__").apply_defaults()
        return obj

    cls.__new__ = __new__

    return cls


def dataclass(cls):
    cls = register_pytree_node(cls)
    cls = dataclasses.dataclass(frozen=True)(cls)
    return cls


def evolve(obj, **changes):
    return attr.evolve(obj, **changes)


def save(obj, path):
    dirname = os.path.abspath(os.path.dirname(path))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load(path):
    return pickle.load(open(path, "rb"))


class Obj():
    def __attrs_post_init__(self):
        pass


class Env(Obj):
    def __call__(self, state, action):
        raise NotImplementedError()

    def f_x(self, state, action):
        return jax.jacfwd(state.dynamics, argnums=0)(state, action)

    def f_u(self, state, action):
        return jax.jacfwd(state.dynamics, argnums=1)(state, action)

    def c_x(self, state, action):
        return jax.grad(state.cost, argnums=0)(state, action)

    def c_u(self, state, action):
        return jax.grad(state.cost, argnums=1)(state, action)

    def c_xx(self, state, action):
        return jax.hessian(state.cost, argnums=0)(state, action)

    def c_uu(self, state, action):
        return jax.hessian(state.cost, argnums=1)(state, action)


class Agent(Obj):
    @staticmethod
    @abstractmethod
    def reset(agent):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def action(agent, observation):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def update(agent, observation):
        raise NotImplementedError()


deluca.Obj = Obj
deluca.Env = Env
deluca.Agent = Agent
deluca.evolve = evolve
deluca.save = save
deluca.load = load
