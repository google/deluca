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

import os
import numpy as np
import matplotlib
import pickle
import matplotlib.pyplot as plt

from deluca.lung.core import BreathWaveform


class Analyzer:

  def __init__(self, path):
    self.data = path if isinstance(path, dict) else pickle.load(
        open(path, "rb"))
    timeseries = self.data["timeseries"]
    self.tt = timeseries["timestamp"]
    self.u_in = timeseries["u_in"]
    self.u_out = timeseries["u_out"]
    self.pressure = timeseries["pressure"]
    self.target = timeseries["target"]
    self.flow = timeseries["flow"]

    # TODO: remove legacy logic
    if "waveform" in self.data:
      self.waveform = self.data["waveform"]
    elif "controller" in self.data:
      self.controller = self.data["controller"]
      if hasattr(self.controller, "waveform"):
        self.waveform = self.controller.waveform
        self.waveform.dtype = np.float32

    if "env" in self.data:
      self.env = self.data["env"]

  ###########################################################################
  # Plotting methods
  ###########################################################################

  def plot(self,
           title=None,
           axes=None,
           figsize=None,
           xlim=None,
           ylim=[0, 60],
           legend=False,
           control=True,
           path=None,
           **kwargs):
    matplotlib.pyplot.figure(figsize=(figsize or (12, 6)))
    # trash
    if axes is None:
      axes = matplotlib.pyplot.axes()

    if xlim is not None:
      axes.set_xlim(xlim)

    axes.set_xlabel("Time (s)")
    plts = []
    (pressure,) = axes.plot(
        self.tt, self.pressure, color="blue", label="actual pressure", **kwargs)
    (target,) = axes.plot(
        self.tt, self.target, color="orange", label="target pressure", **kwargs)
    axes.set_ylabel("Pressure (cmH2O)")
    axes.set_ylim(ylim)
    expiratory = axes.fill_between(
        self.tt,
        axes.get_ylim()[0],
        axes.get_ylim()[1],
        where=self.u_out.astype(bool),
        color="dimgray",
        alpha=0.3,
        label="expiratory phase",
    )
    inspiratory = axes.fill_between(
        self.tt,
        axes.get_ylim()[0],
        axes.get_ylim()[1],
        where=np.logical_not(self.u_out.astype(bool)),
        color="lightgray",
        alpha=0.3,
        label="inspiratory phase",
    )
    plts.extend([pressure, target, inspiratory, expiratory])

    if control:
      twin_ax = axes.twinx()
      twin_ax.set_ylim([-2, 102])
      (u_in,) = twin_ax.plot(
          self.tt,
          np.clip(self.u_in, 0, 100),
          c="gray",
          label="control",
          **kwargs)
      twin_ax.set_ylabel("Inspiratory valve control (% open)")
      plts.append(u_in)

    if title is not None:
      plt.title(title)

    if legend:
      labels = [p.get_label() for p in plts]
      plt.legend(
          plts,
          labels,
          # bbox_to_anchor=(-0.05, 1.02, 1.1, 0.05),
          # mode="expand",
          # ncol=4,
          loc="upper right",
      )

    if path is not None:
      plt.savefig(path)

  def plot_inspiratory_clips(self, **kwargs):
    inspiratory_clips = self.infer_inspiratory_phases()

    plt.subplot(121)
    plt.title("u_in")
    for start, end in inspiratory_clips:
      u_in = self.u_in[start:end]
      plt.plot(self.tt[start:end] - self.tt[start], u_in, "k", alpha=0.1)

    plt.subplot(122)
    plt.title("pressure")
    for start, end in inspiratory_clips:
      pressure = self.pressure[start:end]
      plt.plot(self.tt[start:end] - self.tt[start], pressure, "b", alpha=0.1)

  ###########################################################################
  # Utility methods
  ###########################################################################
  def get_clips(self, inspiratory_only=True, clip=(2, -1)):

    if inspiratory_only:
      clip_indices = self.infer_inspiratory_phases()
    else:
      clip_indices = self.infer_breaths()

    clips = []
    for start, end in clip_indices[clip[0]:clip[1]]:
      u_in = self.u_in[start:end]
      pressure = self.pressure[start:end]
      clips.append((u_in, pressure))

    return clips

  def get_clips_kaggle(self, R, C, path_id, inspiratory_only=True):

    if inspiratory_only:
      clip_indices = self.infer_inspiratory_phases()
    else:
      clip_indices = self.infer_breaths()

    clips = []
    for i, (start, end) in enumerate(clip_indices):
      untruncated_length = end - start
      if (end - start) < 29:
        continue
      if end - start > 29:
        end = start + 29
      u_in = self.u_in[start:end]
      u_out = self.u_out[start:end]
      pressure = self.pressure[start:end]
      time_step = self.tt[start:end]
      initial_time_step = time_step[0]
      time_step = [t - initial_time_step for t in time_step]
      row = {}
      row["time_step"] = time_step
      row["R"] = R
      row["C"] = C
      row["u_in"] = u_in
      row["u_out"] = u_out
      row["pressure"] = pressure
      row["untruncated_length"] = untruncated_length
      clips.append(row)

    return clips

  def get_min_breath_len_kaggle(self, inspiratory_only=True):
    if inspiratory_only:
      clip_indices = self.infer_inspiratory_phases()
    else:
      clip_indices = self.infer_breaths()
    min_breath_len = 10000000
    for start, end in clip_indices:
      breath_len = end - start
      min_breath_len = min(min_breath_len, breath_len)
    return min_breath_len

  def get_all_breath_lens_kaggle(self, inspiratory_only=True):
    if inspiratory_only:
      clip_indices = self.infer_inspiratory_phases()
    else:
      clip_indices = self.infer_breaths()
    breath_lens = []
    for start, end in clip_indices:
      breath_lens.append(end - start)
    return breath_lens

  def infer_inspiratory_phases(self, use_cached=True):
    # finds inspiratory phase intervals from expiratory valve controls
    # returns list of endpoints so that u_out[lo:hi] == 1

    if not use_cached or not hasattr(self, "cached_inspiratory_phases"):
      d_u_out = np.diff(self.u_out, prepend=1)

      starts = np.where(d_u_out == -1)[0]
      ends = np.where(d_u_out == 1)[0]

      self.cached_inspiratory_phases = list(zip(starts, ends))

    return self.cached_inspiratory_phases

  def infer_breaths(self):
    d_u_out = np.diff(self.u_out, prepend=1)
    starts = np.where(d_u_out == -1)[0]
    ends = np.roll(starts - 1, shift=-1)
    ends[-1] = len(self.u_out) - 1

    return list(zip(starts, ends))

  ###########################################################################
  # Metric methods
  ###########################################################################

  def losses_per_breath(self, target, loss_fn=None):
    # computes trapezoidally integrated loss per inferred breath

    loss_fn = loss_fn or np.square

    # handle polymorphic targets
    if isinstance(target, int):
      target_fn = lambda _: target
    elif isinstance(target, BreathWaveform):
      target_fn = lambda t: target.at(t)
    else:
      raise ValueError("unrecognized type for target")

    breaths = self.infer_inspiratory_phases()
    losses = []

    # integrate loss for each detected inspiratory phase
    for start, end in breaths:
      errs = loss_fn(target_fn(self.tt[start:end]) - self.pressure[start:end])
      loss = np.trapz(errs, self.tt[start:end])
      losses.append(loss)

    return np.array(losses)

  def default_metric(self, target=None, loss_fn=np.abs):
    # I suggest keeping a separate function for default settings
    # so nobody has to change code if we change the benchmark
    # as it stands: average loss across breaths, discounting first breath
    target = target or self.waveform

    return self.losses_per_breath(target, loss_fn)[1:].mean()
