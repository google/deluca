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

"""SNN Simulator."""
from deluca.lung.utils.nn.alpha_dropout import AlphaDropout
import flax.linen as nn
import jax


class SNN(nn.Module):
  """SNN neural network."""
  out_dim: int = 1
  hidden_dim: int = 100
  n_layers: int = 4
  droprate: float = 0.0
  scale: float = 1.0507009873554804934193349852946
  alpha: float = 1.6732632423543772848170429916717

  @nn.compact
  def __call__(self, x):
    for i in range(self.n_layers - 1):
      x = nn.Dense(
          features=self.hidden_dim, use_bias=False, name=f"SNN_fc{i}")(
              x)
      x = self.scale * nn.elu(x, alpha=self.alpha)
      x = AlphaDropout(
          rate=self.droprate, deterministic=False)(
              x, rng=jax.random.PRNGKey(0))
    x = nn.Dense(features=self.out_dim, use_bias=True, name=f"SNN_fc{i + 1}")(x)
    return x
