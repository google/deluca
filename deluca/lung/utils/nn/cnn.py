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

"""CNN Simulator."""
from typing import Callable
import flax.linen as nn
import jax.numpy as jnp


class CNN(nn.Module):
  """CNN neural network."""
  n_layers: int = 2
  out_channels: int = 10
  kernel_size: int = 3
  # strides: Tuple[int, int] = (1, 1)
  strides: int = 1
  out_dim: int = 1
  activation_fn: Callable[[jnp.array], jnp.array] = nn.relu

  @nn.compact
  def __call__(self, x):
    for i in range(self.n_layers - 1):
      x = nn.Conv(
          features=self.out_channels,
          kernel_size=(self.kernel_size,),
          strides=(self.strides,),
          name=f"conv{i}")(
              x)
      x = self.activation_fn(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=self.out_dim, use_bias=True, name=f"CNN_fc{i + 1}")(x)
    return x.squeeze()  # squeeze for consistent shape w/ boundary model output
