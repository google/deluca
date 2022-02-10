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

"""Shift Scale Transform."""
import jax.numpy as jnp


class ShiftScaleTransform:
  """Data normalizer."""

  # vectors is an array of vectors
  def __init__(self, vectors):
    vectors_concat = jnp.concatenate(vectors)
    self.mean = jnp.mean(vectors_concat)
    self.std = jnp.std(vectors_concat)
    print(self.mean, self.std)

  def _transform(self, x, mean, std):
    return (x - mean) / std

  def _inverse_transform(self, x, mean, std):
    return (x * std) + mean

  def __call__(self, vector):
    return self._transform(vector, self.mean, self.std)

  def inverse(self, vector):
    return self._inverse_transform(vector, self.mean, self.std)
