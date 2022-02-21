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
import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(1,))
def rollout(env,
            cost_func,
            U_old,
            k=None,
            K=None,
            X_old=None,
            alpha=1.0,
            H=None,
            start_state=None):
  """
    Arg List
    env: The environment to do the rollout on. This is treated as a
    derstructible copy.
    U_old: A base open loop control sequence
    k: open loop gain (iLQR)
    K: closed loop gain (iLQR)
    X_old: Previous trajectory to compute gain (iLQR)
    alpha: Optional multplier to the open loop gain (iLQR)
    D: Optional noise vectors for the rollout (GPC)
    F: Linearization shift ??? (GPC)
    H: The horizon length to perform a rollout.
    """
  ## problematic
  # if H == None:
  #     H = env.H
  H = env.H
  X, U = [None] * (H + 1), [None] * (H)
  if start_state is None:
    start_state = env.init()
  X[0], cost = start_state, 0.0
  for h in range(H):
    if k is None:
      U[h] = U_old[h]
    else:
      X_flat = X[h].flatten()
      X_old_flat = X_old[h].flatten()
      U[h] = U_old[h] + alpha * k[h] + K[h] @ (X_flat - X_old_flat)

    X[h + 1], _ = env(X[h], U[h])
    cost += cost_func(X[h], U[h], env)
  return X, U, cost


def hessian(f, argnums=0):
  return jax.jacfwd(jax.jacrev(f, argnums=argnums), argnums=argnums)


@partial(jax.jit, static_argnums=(1,))
def compute_ders_inner(env, cost, X, U, H=None):

  def func(x0, x1, u):
    x0 = env.init().unflatten(x0)
    new_state, _ = env(x0, u)
    d = x1 - new_state.flatten()
    g, _ = jax.jacfwd(env, argnums=(0, 1))(x0, u)
    f = (g.arr[0].arr, g.arr[1])
    c_der = jax.grad(cost, argnums=(0, 1))(x0, u, env)
    c_der_der = hessian(cost, (0, 1))(x0, u, env)
    c = (c_der[0].arr, c_der[1], c_der_der[0].arr[0].arr, c_der_der[1][1])

    return d, f, c

  X = [x.flatten() for x in X]
  X0 = jnp.array(X[:-1])
  X1 = jnp.array(X[1:])
  U = jnp.array(U)
  D, F, C = jax.vmap(func)(X0, X1, U)
  return D, F, C


def compute_ders(env, cost, X, U, H=None):
  D, F, C = compute_ders_inner(env, cost, X, U, H=None)
  F = list(zip(*F))
  C = list(zip(*C))
  return D, F, C
