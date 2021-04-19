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

class Entity:
    def __new__(cls, *args, **kwargs):
        """For avoiding super().__init__()"""
        obj = object.__new__(cls)
        obj.__setattr__("attrs_", {})
        obj.__setattr__("arguments_", inspect.signature(obj.__init__).bind(*args, **kwargs))
        obj.arguments_.apply_defaults()

        for key, val in obj.arguments_.arguments.items():
            obj.__setattr__(key, val)

        return obj

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def attrs(self):
        return self.attrs_

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
