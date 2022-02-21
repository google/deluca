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

"""Operate on collection of Analyzers.

This does all the data
pre-processing including shuffling and splitting data, computing
statistics mean/std, and forming jax dataloader.
"""
from deluca.lung.utils.data.analyzer import Analyzer
from deluca.lung.utils.data.transform import ShiftScaleTransform
import jax
import jax.numpy as jnp
import numpy as np


# pylint: disable=dangerous-default-value


class BreathDataset:
  """dataset for breaths."""

  @classmethod
  def from_paths(
      cls,
      paths,
      seed=0,
      clip=(0, None),  # used to be (2, -1)
      breath_length=29,
      keys=["train", "test"],
      splits=[0.9, 0.1],
      fit_scaler_key="train",
  ):
    """process data from paths."""
    obj = object.__new__(cls)
    raw_data = []

    for path in paths:
      analyzer = Analyzer(path)
      raw_data += analyzer.get_breaths(clip=clip, breath_length=breath_length)

    # Shuffle and split raw_data
    rng = np.random.default_rng(seed)
    splits = (np.array(splits) / np.sum(splits) *
              len(raw_data)).astype("int")[:-1]
    rng.shuffle(raw_data)
    obj.splits = {
        key: val for key, val in zip(keys, np.split(raw_data, splits))
    }

    # Compute mean, std for u_in and pressure
    obj.u_normalizer = ShiftScaleTransform(
        jnp.array([u for u, p in obj.splits[fit_scaler_key]]))
    print(f"u_in: mean={obj.u_normalizer.mean}, std={obj.u_normalizer.std}")
    obj.p_normalizer = ShiftScaleTransform(
        jnp.array([p for u, p in obj.splits[fit_scaler_key]]))
    print(f"pressure: mean={obj.p_normalizer.mean}, std={obj.p_normalizer.std}")

    obj.data = {}
    for key in keys:
      obj.data[key] = (jnp.array([u for u, p in obj.splits[key]]),
                       jnp.array([p for u, p in obj.splits[key]]))
    return obj


def get_shuffled_and_batched_data(dataset, batch_size, key, prng_key):
  """function to shuffle and batch data."""
  x, y = dataset.data[key]
  x = jax.random.permutation(prng_key, x)
  y = jax.random.permutation(prng_key, y)
  prng_key, _ = jax.random.split(prng_key)
  num_batches = x.shape[0] // batch_size
  trunc_len = num_batches * batch_size
  trunc_x = x[:trunc_len]
  trunc_y = y[:trunc_len]
  batched_x = jnp.reshape(trunc_x, (num_batches, batch_size, trunc_x.shape[1]))
  batched_y = jnp.reshape(trunc_y, (num_batches, batch_size, trunc_y.shape[1]))
  return batched_x, batched_y, prng_key


def get_initial_pressure(dataset, key):
  return jnp.array([dataset.p_normalizer(p[0]) for u, p in dataset.splits[key]
                   ]).mean().item()
