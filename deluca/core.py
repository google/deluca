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

import jax


def tree_flatten(obj):
    """Flatten module parameters for Jax"""
    leaves, aux = jax.tree_util.tree_flatten(obj.attrs)
    aux = {
        "treedef": aux,
        "arguments_": obj.arguments_,
        "class": obj.__class__,
    }
    return leaves, aux


def tree_unflatten(aux, leaves):
    """Unflatten obj parameters for Jax"""
    obj = aux["class"](*aux["arguments_"].args, **aux["arguments_"].kwargs)
    attrs = jax.tree_util.tree_unflatten(aux["treedef"], leaves)
    for key, val in attrs.items():
        obj.__setattr__(key, val)
    return obj


class JaxObject:
    def __new__(cls, *args, **kwargs):
        """For avoiding super().__init__()"""
        obj = object.__new__(cls)
        obj.__setattr__("attrs_", {})
        obj.__setattr__("arguments_", inspect.signature(obj.__init__).bind(*args, **kwargs))
        obj.arguments_.apply_defaults()

        for key, val in obj.arguments_.arguments.items():
            obj.__setattr__(key, val)

        return obj

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        """For avoiding a decorator for each subclass"""
        super().__init_subclass__(*args, **kwargs)
        jax.tree_util.register_pytree_node(cls, tree_flatten, tree_unflatten)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def attrs(self):
        return self.attrs_

    def __str__(self):
        return self.name

    def __setattr__(self, key, val):
        if (
            key[-1] != "_"
            and not callable(val)
            and key != "observation_space"
            and key != "action_space"
            and not isinstance(val, str)
            and val is not None
        ):
            self.attrs[key] = val
        self.__dict__[key] = val

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
