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

"""MLP module."""
from typing import Callable
import flax.linen as nn
import jax

# pylint: disable=g-bare-generic


class MLP(nn.Module):
  """multilayered perceptron."""
  hidden_dim: int = 10
  out_dim: int = 1
  n_layers: int = 2
  droprate: float = 0.0
  activation_fn: Callable = nn.relu

  @nn.compact
  def __call__(self, x):
    for i in range(self.n_layers - 1):
      x = nn.Dense(
          features=self.hidden_dim, use_bias=True, name=f"MLP_fc{i}")(
              x)
      x = nn.Dropout(
          rate=self.droprate, deterministic=False)(
              x, rng=jax.random.PRNGKey(0))
      x = self.activation_fn(x)
    x = nn.Dense(features=self.out_dim, use_bias=True, name=f"MLP_fc{i + 1}")(x)
    return x.squeeze()  # squeeze for consistent shape w/ boundary model output
