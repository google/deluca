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
"""

import os
import pickle
import typing
import inspect
import jax
import attr

import deluca

from abc import abstractmethod
from collections import OrderedDict
from functools import wraps


def flatten(obj):
    data_fields = []
    meta_fields = []

    for field in attr.fields(obj.__class__):
        if "pytree" in field.metadata and field.metadata["pytree"]:
            data_fields.append(field.name)
        else:
            meta_fields.append(field.name)

    children = tuple(getattr(obj, name) for name in data_fields)
    meta = dict(zip(meta_fields, tuple(getattr(obj, name) for name in meta_fields)))

    return children, {"class": obj.__class__, "meta": meta, "data_fields": data_fields}


def unflatten(aux, children):
    data_fields = aux["data_fields"]
    data = dict(zip(data_fields, children))
    data.update(aux["meta"])
    return aux["class"](**data)


def attrib(default=None, takes_self=False, pytree=False, **kwargs):
    if takes_self:
        default = attr.Factory(default, takes_self=True)

    kwargs["default"] = default

    if pytree:
        if "metadata" not in kwargs:
            kwargs["metadata"] = {}
        kwargs["metadata"]["pytree"] = True

    return attr.ib(**kwargs)


def default(pytree=False):
    def decorator(method):
        setattr(method, "__default_method__", True)
        if pytree:
            setattr(method, "__pytree__", True)
        return method

    if callable(pytree):
        return decorator(pytree)
    else:
        return decorator


def pytree(default=None, **kwargs):
    return attrib(default=default, pytree=True, **kwargs)


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


def isclassmethod(cls, name):
    method = getattr(cls, name)
    bound_to = getattr(method, "__self__", None)
    if not isinstance(bound_to, type):
        # must be bound to a class
        return False
    name = method.__name__
    for cls in bound_to.__mro__:
        descriptor = vars(cls).get(name)
        if descriptor is not None:
            return isinstance(descriptor, classmethod)
    return False


def isstaticmethod(cls, name):
    return isinstance(inspect.getattr_static(cls, name), staticmethod)


def isnotdefaultmethod(cls, name):
    method = getattr(cls, name)
    return callable(method) and not hasattr(getattr(cls, name), "__default_method__")


class OrderedMeta(type):
    @classmethod
    def __prepare__(cls, name, bases):
        return OrderedDict()

    def __new__(cls, name, bases, clsdict):
        c = type.__new__(cls, name, bases, clsdict)
        c.__ordered_keys__ = clsdict.keys()
        for member in c.__ordered_keys__:
            value = getattr(c, member)

            if not hasattr(c, "__annotations__"):
                setattr(cls, "__annotations__", {})

            if (
                member.startswith("_")
                or isstaticmethod(c, member)
                or isclassmethod(c, member)
                or isnotdefaultmethod(c, member)
                or (
                    member in c.__annotations__
                    and hasattr(c.__annotations__[member], "__origin__")
                    and c.__annotations__[member].__origin__ is typing.ClassVar
                )
            ):
                continue

            if not isinstance(value, attr._make._CountingAttr):
                setattr(
                    c,
                    member,
                    attrib(
                        default=value,
                        takes_self=callable(value),
                        pytree=hasattr(value, "__pytree__"),
                    ),
                )

            # All attrs should have annotations
            if member not in c.__annotations__:
                c.__annotations__[member] = type(value)
        c = attr.s(c)
        jax.tree_util.register_pytree_node(c, flatten, unflatten)
        return c


class Obj(metaclass=OrderedMeta):
    def __attrs_post_init__(self):
        pass


class Env(Obj):
    @staticmethod
    @abstractmethod
    def reset(env):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def dynamics(env, action):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def cost(env, action):
        return NotImplementedError()

    @staticmethod
    def f_x(env, action):
        return jax.jacfwd(env.dynamics, argnums=0)(env, action)

    @staticmethod
    def f_u(env, action):
        return jax.jacfwd(env.dynamics, argnums=1)(env, action)

    @staticmethod
    def c_x(env, action):
        return jax.grad(env.cost, argnums=0)(env, action)

    @staticmethod
    def c_u(env, action):
        return jax.grad(env.cost, argnums=1)(env, action)

    @staticmethod
    def c_xx(env, action):
        return jax.hessian(env.cost, argnums=0)(env, action)

    @staticmethod
    def c_uu(env, action):
        return jax.hessian(env.cost, argnums=1)(env, action)


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
deluca.attrib = attrib
deluca.default = default
deluca.pytree = pytree
deluca.evolve = evolve
deluca.save = save
deluca.load = load
