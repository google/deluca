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


def hessian(f, arg):
  return jax.jacfwd(jax.jacrev(f, argnums=arg), argnums=arg)


def iLQR(
    env,
    cost,
    U,
    T,
    k=None,
    K=None,
    X=None,
    info="true",
    start_state=None,
    H=None,
    verbose=True,
    alpha=0.5,
    backtracking=True,
):
  r = 1
  if H == None:
    H = env.H
  X, U, c = rollout(env, cost, U, k, K, X, start_state=start_state, H=H)
  if verbose:
    print(f"iLQR ({info}): t = -1, r = 1, c = {c}")
  for t in range(T):
    # s = time.time()
    _, F, C = compute_ders(env, cost, X, U)
    # t = time.time()
    # print('der time ', t-s)
    k, K = LQR(F, C)
    if backtracking:
      for alphaC in alpha * 1.1 * 1.1**(-jnp.arange(10)**2):
        r += 1
        # s = time.time()
        XC, UC, cC = rollout(
            env,
            cost,
            U,
            k,
            K,
            X,
            alpha,
            start_state=start_state,
            H=H,
        )
        if verbose:
          print(
              f"iLQR ({info}): t = {t}, r = {r}, alphac = {alphaC}, cost = {cC}"
          )
        if cC <= c:
          X, U, c, alpha = XC, UC, cC, alphaC
          if verbose:
            print(f"iLQR ({info}): t = {t}, r = {r}, c = {c}")
          break
    else:
      r += 1
      X, U, c = rollout(
          env,
          cost,
          U,
          k,
          K,
          X,
          alpha,
          start_state=start_state,
          H=H,
      )
      if verbose:
        print(f"iLQR ({info}): t = {t}, r = {r}, cost = {c}")
  return X, U, k, K, c


def iLQR_open(env_true, env_sim, cost_func, U_initial, T):
  X, U, k, K, c = iLQR(env_sim, cost_func, U_initial, T, info="sim")
  X, U, c = rollout(env_true, cost_func, U)
  print(f"iLQR (open): t = 1, r = 1, c = {c}")
  return X, U, None, None, c


def iLQR_closed(env_true, env_sim, cost_func, U_initial, T):
  X, U, k, K, c = iLQR(env_sim, cost_func, U_initial, T, info="sim")
  X, U, c = rollout(env_true, cost_func, U, k, K, X)
  print(f"iLQR (closed): t = 1, r = 1, c = {c}")
  return X, U, k, K, c


def iLQR_oracle(env_true, env_sim, U_initial, T):
  X, U, k, K, c = iLQR(env_sim, U_initial, T, info="sim")
  X, U, k, K, c = iLQR(env_true, U, T, k, K, X, "oracle")
  return X, U, k, K, c
