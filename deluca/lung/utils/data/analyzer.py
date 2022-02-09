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

"""Class for handling data processing on individual runs.

Separates runs into breaths, computes default metric,
and supports plotting runs.
"""
import pickle

from deluca.lung.core import BreathWaveform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# pylint: disable=unnecessary-lambda
# pylint: disable=dangerous-default-value


class Analyzer:
  """Class for processing individual runs."""

  def __init__(self, path, test_pressure=None):
    self.data = path if isinstance(path, dict) else pickle.load(
        open(path, "rb"))
    timeseries = self.data["timeseries"]
    self.tt = timeseries["timestamp"]
    self.u_in = timeseries["u_in"]
    self.u_out = timeseries["u_out"]
    self.pressure = timeseries["pressure"]
    self.target = timeseries["target"]
    self.flow = timeseries["flow"]
    self.test_pressure = test_pressure
    if "global_id" in timeseries:
      self.global_id = timeseries["global_id"]
    if "waveform" in self.data:
      self.waveform = self.data["waveform"]

  ###########################################################################
  # Utility methods
  ###########################################################################
  def get_breaths(self, clip=(0, None), breath_length=29):
    """Return list of breaths for inspiratory phases only."""
    breath_indices = self.get_breath_indices(breath_length=breath_length)
    breaths = []

    # NOTE: remove first two breaths and last breath because they often have
    # initialization or timing-related issues
    for start, end in breath_indices[clip[0]:clip[1]]:

      # Ignore breaths shorter than breath_length
      if end - start < breath_length:
        continue

      # Truncate clips longer than breath_length to breath_length
      if end - start > breath_length:
        end = start + breath_length

      u_in = self.u_in[start:end]
      pressure = self.pressure[start:end]
      breaths.append((u_in, pressure))

    return breaths

  def get_breath_indices(self, use_cached=True, breath_length=29):
    """finds inspiratory phase intervals from expiratory valve controls.

    Args:
      use_cached: whether or not to use self.cached_breaths
      breath_length: length of a breath
    Returns:
      list of endpoints so that u_out[lo:hi] == 1
    """

    if not use_cached or not hasattr(self, "cached_breaths"):
      d_u_out = np.diff(self.u_out, prepend=1)

      starts = np.where(d_u_out == -1)[0]
      ends = np.where(d_u_out == 1)[0]

      self.cached_breaths = list(zip(starts, ends))

    # Return a single breath of the entire length
    return self.cached_breaths or [(0, breath_length)]

  ###########################################################################
  # Metric methods
  ###########################################################################
  def losses_per_breath(self,
                        target,
                        clip=(0, None),
                        loss_fn=None,
                        breath_length=29):
    """compute losses per breath."""
    # computes trapezoidally integrated loss per inferred breath
    loss_fn = loss_fn or np.square

    # handle polymorphic targets
    if isinstance(target, BreathWaveform):
      target_fn = lambda t: target.at(t)
    else:
      target_fn = lambda _: target

    breaths_indices = self.get_breath_indices(breath_length=breath_length)
    losses = []

    # integrate loss for each detected inspiratory phase
    for start, end in breaths_indices[clip[0]:clip[1]]:
      errs = loss_fn(target_fn(self.tt[start:end]) - self.pressure[start:end])
      loss = np.trapz(errs, self.tt[start:end])
      losses.append(loss)

    return np.array(losses)

  def default_metric(self,
                     clip=(0, None),
                     target=None,
                     loss_fn=np.abs,
                     breath_length=29):
    """default loss function."""
    # I suggest keeping a separate function for default settings
    # so nobody has to change code if we change the benchmark
    # as it stands: average loss across breaths, discounting first breath
    if target is None:
      target = self.waveform

    return self.losses_per_breath(
        clip=clip, target=target, loss_fn=loss_fn,
        breath_length=breath_length).mean()

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
    """plot function."""
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
    if self.test_pressure is not None:
      (test_pressure,) = axes.plot(
          self.tt,
          self.test_pressure,
          color="red",
          label="test pressure",
          **kwargs)
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
    if self.test_pressure is not None:
      plts.extend([pressure, target, test_pressure, inspiratory, expiratory])
    else:
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
