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
- jax convenience functions are a bit of a kludge
- extracting state from the env object is a little messy
"""

import os
import pickle
import typing
import inspect
import jax
import attr

import deluca

import abc
from abc import abstractmethod


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


class Obj(abc.ABC):
    def __init_subclass__(cls, *args, **kwargs):
        parent_cls = cls.__mro__[1]
        for member in dir(parent_cls):
            if isinstance(
                inspect.getattr_static(parent_cls, member), staticmethod
            ) and not isinstance(inspect.getattr_static(cls, member), staticmethod):
                raise TypeError(
                    f"Class {cls.__name__} should have staticmethod "
                    "{member}, but {member} is not static."
                )

            if hasattr(parent_cls, "__abstractmethods__") and member in getattr(
                parent_cls, "__abstractmethods__"
            ):
                if member not in cls.__dict__:
                    raise TypeError(
                        f"Abstract method {member} not defined in class {cls.__name__}."
                    )

            if (
                member in cls.__dict__
                and callable(getattr(cls, member))
                and not (
                    inspect.signature(getattr(cls, member))
                    == inspect.signature(getattr(parent_cls, member))
                )
            ):
                raise TypeError(
                    f"Method `{member}` should have the same signature in class `{cls.__name__}` as in its parent class `{parent_cls.__name__}`"
                )

        if not hasattr(cls, "__annotations__"):
            setattr(cls, "__annotations__", {})

        for member in set(dir(cls)) - set(dir(parent_cls)):
            if (
                member.startswith("_")
                or callable(getattr(cls, member))
                or (
                    member in cls.__annotations__
                    and hasattr(cls.__annotations__[member], "__origin__")
                    and cls.__annotations__[member].__origin__ is typing.ClassVar
                )
            ):
                continue

            value = getattr(cls, member)

            if not isinstance(getattr(cls, member), attr._make._CountingAttr):
                setattr(cls, member, attr.ib(default=value))

            if member not in cls.__annotations__:
                cls.__annotations__[member] = type(value)

        attr.s(cls, frozen=True, kw_only=True)
        jax.tree_util.register_pytree_node(cls, flatten, unflatten)

    @staticmethod
    def evolve(obj, **changes):
        return attr.evolve(obj, **changes)

    def save(self, path):
        dirname = os.path.abspath(os.path.dirname(path))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        return pickle.load(open(path, "rb"))


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


def attrib(default=None, pytree=False, **kwargs):
    if default is not None:
        kwargs["default"] = default

    if pytree:
        if "metadata" not in kwargs:
            kwargs["metadata"] = {}
        kwargs["metadata"]["pytree"] = True

    return attr.ib(**kwargs)


def pytree(default=None, **kwargs):
    return attrib(default=default, pytree=True, **kwargs)


deluca.Obj = Obj
deluca.Env = Env
deluca.Agent = Agent
deluca.attrib = attrib
deluca.pytree = pytree
