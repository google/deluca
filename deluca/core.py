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
import jax
import attr

import deluca
import zope.interface
from zope.interface.verify import verifyClass


class IObj(zope.interface.Interface):
    """Interface describing deluca objects"""


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


def evolve(obj, **changes):
    return attr.evolve(obj, **changes)


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


def save(obj, path):
    dirname = os.path.abspath(os.path.dirname(path))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load(path):
    return pickle.load(open(path, "rb"))


def flatten(self):
    data_fields = []
    meta_fields = []

    for field in attr.fields(self.__class__):
        if "pytree" in field.metadata and field.metadata["pytree"]:
            data_fields.append(field.name)
        else:
            meta_fields.append(field.name)

    children = tuple(getattr(self, name) for name in data_fields)
    meta = dict(zip(meta_fields, tuple(getattr(self, name) for name in meta_fields)))

    return children, {"class": self.__class__, "meta": meta, "data_fields": data_fields}


def unflatten(aux, children):
    data_fields = aux["data_fields"]
    data = dict(zip(data_fields, children))
    data.update(aux["meta"])
    return aux["class"](**data)


def _make_and_check_class(iface, cls, statics=None):
    statics = statics or []

    for member in set(dir(cls)) - set(statics):
        if member.startswith("_") or (
            member in cls.__annotations__
            and hasattr(cls.__annotations__[member], "__origin__")
            and cls.__annotations__[member].__origin__ is typing.ClassVar
        ):
            continue

        value = getattr(cls, member)
        if not isinstance(getattr(cls, member), attr._make._CountingAttr):
            setattr(cls, member, attr.ib(default=value))
        cls.__annotations__[member] = type(value)

    # NOTE: We avoid setting functions without inheritance, but make an
    # exception here
    for member in statics:
        if not isinstance(cls.__dict__[member], staticmethod):
            raise TypeError(
                f"All public methods must be static. Method "
                f"`{member}` in `{cls.__name__}` is not. Either make it "
                f"static or prefix with `_` to indicate it is "
                f"private."
            )

    cls = attr.s(auto_attribs=True, kw_only=True)(cls)

    cls = zope.interface.implementer(iface)(cls)
    verifyClass(iface, cls)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)

    return cls


def obj(cls: type):
    return _make_and_check_class(IObj, cls, [])


def env(cls: type):
    # NOTE: And adding a bunch of functions here...
    def f_x(env, action):
        return jax.jacfwd(cls.dynamics, argnums=0)(env, action)

    def f_u(env, action):
        return jax.jacfwd(cls.dynamics, argnums=1)(env, action)

    def c_x(env, action):
        return jax.grad(cls.cost, argnums=0)(env, action)

    def c_u(env, action):
        return jax.grad(cls.cost, argnums=1)(env, action)

    def c_xx(env, action):
        return jax.hessian(cls.cost, argnums=0)(env, action)

    def c_uu(env, action):
        return jax.hessian(cls.cost, argnums=1)(env, action)

    for func in [f_x, f_u, c_x, c_u, c_xx, c_uu]:
        if not hasattr(cls, func.__name__):
            setattr(cls, func.__name__, staticmethod(func))

    cls = _make_and_check_class(
        IEnv, cls, ["reset", "dynamics", "cost", "f_x", "f_u", "c_x", "c_u", "c_xx", "c_uu"]
    )

    return cls


def agent(cls: type):
    return _make_and_check_class(IAgent, cls, ["reset", "action", "update"])


deluca.obj = obj
deluca.env = env
deluca.agent = agent
deluca.evolve = evolve
deluca.attrib = attrib
deluca.pytree = pytree
deluca.save = save
deluca.load = load
deluca.IEnv = IEnv
deluca.IAgent = IAgent
