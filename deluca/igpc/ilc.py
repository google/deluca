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

import os
import time
import jax.numpy as jnp
from deluca.igpc.rollout import rollout, compute_ders
from deluca.igpc.lqr_solver import LQR
import jax


def iLC_closed(
    env_true,
    env_sim,
    cost_func,
    U,
    T,
    k=None,
    K=None,
    X=None,
    ref_alpha=1.0,
    verbose=True,
    backtracking=True,
):
  alpha, r = ref_alpha, 1
  X, U, c = rollout(env_true, cost_func, U, k, K, X)
  print(f"iLC: t = {-1}, r = {r}, c = {c}")
  prev_loop_fail = False
  for t in range(T):
    _, F, C = compute_ders(env_sim, cost_func, X, U)
    k, K = LQR(F, C)
    if backtracking:
      if prev_loop_fail:
        if verbose:
          print("Backtracking failed - Quitting")
          break

      prev_loop_fail = True
      for alphaC in alpha * 1.1 * 1.1**(-jnp.arange(10)**2):
        r += 1
        XC, UC, cC = rollout(env_true, cost_func, U, k, K, X, alphaC)
        if cC <= c:
          X, U, c, alpha = XC, UC, cC, alphaC
          if verbose:
            print(
                f"iLC (closed+alpha={ref_alpha}): t = {t}, r = {r}, c = {c}, alpha = {alpha}"
            )
            prev_loop_fail = False
          break
    else:
      r += 1
      XC, UC, cC = rollout(env_true, cost_func, U, k, K, X, alpha)
      if verbose:
        print(
            f"iLC (closed+alpha={ref_alpha}): t = {t}, r = {r}, c = {c}, alpha = {alpha}"
        )
      break
  return X, U, k, K, c
