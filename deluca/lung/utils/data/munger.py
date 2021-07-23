# Copyright 2021 The Deluca Authors.
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

import jax.numpy as jnp
import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt
import pandas as pd

from deluca.lung.utils.data.analyzer import Analyzer
from deluca.lung.utils.data.transform import ShiftScaleTransform
import tensorflow as tf

# NOTES
# - @classmethod factory


def resample(tt, f, dt=0.03, snap_left=False, return_grid=False):
  # resample a continuous-time function interpolated from non-uniform samples
  num_tt = int((tt[-1] - tt[0]) / dt)
  tt_grid = jnp.arange(num_tt) * dt + tt[0]

  if snap_left:
    f_grid = f[jnp.searchsorted(tt, tt_grid)]
  else:
    f_grid = jnp.interp(tt_grid, tt, f)

  if return_grid:
    return tt_grid, f_grid
  else:
    return f_grid


class Munger:
  @classmethod
  def from_kaggle(cls, path, oiasdjfoaisdjf):
    obj = cls(path)

    return obj

  @classmethod
  def from_legacy(cls, paths, oiasdjfoij):
    obj = cls(paths)

    return obj

  def __init__(
      self,
      paths,
      kaggle_path,
      R=20,
      C=10,
      to_round=False,
      dt_resample=False,
      clip=(2, -1),
      inspiratory_only=True,
      splits=[0.9, 0.1],
      fit_scaler_key="train",
      truncate=True,
      **kwargs,
  ):
    self.paths = paths
    self.analyzers = []
    self.data = []
    self.splits = {}
    self.to_round = to_round

    # Use Kaggle Dataset
    if kaggle_path != []:
      self.add_data(kaggle_path, dt_resample, R=R, C=C, **kwargs)
      # Shuffle and split data
      self.split_data(splits=splits)
      # Compute mean, std for u_in and pressure
      self.u_scaler, self.p_scaler = self.fit_scalers(key=fit_scaler_key)
    # Load dataset using legacy paths
    else:
      self.add_data(
          paths,
          dt_resample,
          clip,
          inspiratory_only=inspiratory_only,
          truncate=truncate,
          **kwargs,
      )
      # Shuffle and split data
      self.split_data(splits=splits)
      # Compute mean, std for u_in and pressure
      self.u_scaler, self.p_scaler = self.fit_scalers(key=fit_scaler_key)  # new_training

  # TODO: do we need to skip first 2 and last breath? Right now just reading from kaggle dataset
  def add_data(self, kaggle_path, dt_resample=True, R=20, C=10, **kwargs):
    failed = 0
    df = pd.read_csv(kaggle_path)
    df = df[(df["R"] == R) & (df["C"] == C)]
    breath_ids = df["breath_id"].unique()
    num_breaths = len(breath_ids)
    # print('num_breaths:' + str(num_breaths))
    for breath_id in breath_ids:
      # print('--------------------------')
      # print('breath_id: ' + str(breath_id))
      try:
        df_breath_id = df[df["breath_id"] == breath_id]
        # print(df_breath_id.head())
        if self.to_round:
          u_in = jnp.round(df_breath_id["u_in"].to_numpy())
        else:
          u_in = df_breath_id["u_in"].to_numpy()

        pressure = df_breath_id["pressure"]

        if dt_resample:
          tt = df_breath_id["time_step"]
          u_in = resample(tt, u_in)
          pressure = resample(tt, pressure)

        self.data.append((u_in, pressure))
        # print('u_in:' + str(u_in))
        # print('pressure' + str(pressure))

      except Exception as e:
        print(breath_id, e)
        failed += 1

    print(f"Added {len(self.data)} breaths from {num_breaths - failed} breath ids.")

  # Note: the following methods are meant to be run in order
  def add_data(
      self, paths, dt_resample=True, clip=(2, -1), inspiratory_only=True, truncate=True, **kwargs
  ):
    if isinstance(paths, str):
      paths = [paths]

    failed = 0
    kaggle_truncate_threshold = 29
    for path in paths:
      try:
        analyzer = Analyzer(path)
        self.analyzers.append(analyzer)
        if inspiratory_only:
          clips = analyzer.infer_inspiratory_phases()
        else:
          clips = analyzer.infer_breaths()

        for start, end in clips[clip[0] : clip[1]]:  # skip first 2 & last breaths
          if self.to_round:
            u_in = jnp.round(analyzer.u_in[start:end])
          else:
            u_in = analyzer.u_in[start:end]
          pressure = analyzer.pressure[start:end]
          if dt_resample:
            tt = analyzer.tt[start:end]
            u_in = resample(tt, u_in)
            pressure = resample(tt, pressure)
          if truncate:
            if len(u_in) < kaggle_truncate_threshold:
              continue
            u_in = u_in[:kaggle_truncate_threshold]
            pressure = pressure[:kaggle_truncate_threshold]
          self.data.append((u_in, pressure))

      except Exception as e:
        print(path, e)
        failed += 1

    print(f"Added {len(self.data)} breaths from {len(self.paths) - failed} paths.")

  def split_data(self, seed=0, keys=["train", "test"], splits=[0.9, 0.1], **kwargs):
    rng = np.random.default_rng(seed)

    # Determine split boundaries
    splits = (np.array(splits) / np.sum(splits) * len(self.data)).astype("int")[:-1]

    # Everyday I'm shuffling
    rng.shuffle(self.data)

    # Splitting
    self.splits = {key: val for key, val in zip(keys, np.split(self.data, splits))}

  def fit_scalers(self, key="train"):
    u_scaler = ShiftScaleTransform(jnp.array([u for u, p in self.splits[key]]))
    print(f"u_in: mean={u_scaler.mean}, std={u_scaler.std}")
    p_scaler = ShiftScaleTransform(jnp.array([p for u, p in self.splits[key]]))
    print(f"pressure: mean={p_scaler.mean}, std={p_scaler.std}")

    return u_scaler, p_scaler

  def get_scaled_data(self, key, scale_data=False):
    X, y = [], []
    if scale_data:
      X = jnp.array([self.u_scaler(u) for u, p in self.splits[key]])
      y = jnp.array([self.p_scaler(p) for u, p in self.splits[key]])
    else:
      X = jnp.array([u for u, p in self.splits[key]])
      y = jnp.array([p for u, p in self.splits[key]])
    return X, y

  def get_dataloader_tf(self, key, shuffle=True, scale_data=False, batch_size=20):
    X, y = self.get_scaled_data(key, scale_data)
    # dataset = Dataset_from_XY(X, y)
    # return Dataloader(dataset, batch_size=batch_size, shuffle=shuffle)
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
      dataset = dataset.shuffle(buffer_size=X.shape[0])
    return dataset

  def unscale_pressures(self, p):  # TODO: make the jax version of this
    if type(p) is float:
      p = jnp.array([p])
    return self.p_scaler.inverse(p)[:, 0]

  ###########################################################################
  # Plotting methods
  ###########################################################################

  def plot_boundary_pressures(self):
    plt.rc("figure", figsize=(16, 4))

    for tau in range(1, 6):
      plt.subplot(150 + tau)

      u_init = []
      p_init = []

      for u, p in self.splits["train"]:
        u_init.append(u[:tau].mean())
        p_init.append(p[tau])

      plt.xlim([0, 105])
      plt.ylim([2, 33])
      plt.xlabel(f"u_in[0:{tau}].mean")
      plt.title(f"pressure[{tau}]")

      plt.scatter(u_init, p_init, s=1)

  def scale_and_window_boundary(self, key, boundary_index):
    X, y = [], []

    if boundary_index == 0:  # special case: no features, predict p[0]
      for u, p in self.splits[key]:
        p_scaled = self.p_scaler(p[0])
        target = p_scaled
        y.append(target)
      return None, jnp.array(y)

    # otherwise, collate [first B inputs, first B pressures] -> (next pressure) pairs
    for u, p in self.splits[key]:
      T = len(u)
      if T < boundary_index + 1:  # if trajectory is too short, abort
        continue

      u_scaled = self.u_scaler(u[:boundary_index].reshape(-1, 1)).flat
      p_scaled = self.p_scaler(p[: boundary_index + 1].reshape(-1, 1)).flat

      features = jnp.concatenate([u_scaled, p_scaled[:-1]])
      target = p_scaled[-1]

      X.append(features)
      y.append(target)

    # return torch.tensor(X, dtype=self.dtype), torch.tensor(y, dtype=self.dtype)
    return jnp.array(X), jnp.array(y)

  def get_scaled_data(self, key, scale_data=False):
    X, y = [], []
    if scale_data:
      X = jnp.array([self.u_scaler(u) for u, p in self.splits[key]])
      y = jnp.array([self.p_scaler(p) for u, p in self.splits[key]])
    else:
      X = jnp.array([u for u, p in self.splits[key]])
      y = jnp.array([p for u, p in self.splits[key]])
    return X, y
