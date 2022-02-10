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

import jax.numpy as np


def f_at_x(env, X, U, H=None):
    if H is None:
        H = env.H
    D, F, C = (
        [None for _ in range(H)],
        [None for _ in range(H)],
        [None for _ in range(H)],
    )
    for h in range(H):
        D[h] = X[h + 1] - env.f(X[h], U[h])
        F[h] = env.f_x(X[h], U[h]), env.f_u(X[h], U[h])
        C[h] = (
            env.c_x(X[h], U[h]),
            env.c_u(X[h], U[h]),
            env.c_xx(X[h], U[h]),
            env.c_uu(X[h], U[h]),
        )
    return D, F, C


def rollout(
    env, U_old, k=None, K=None, X_old=None, alpha=1.0, D=None, F=None, H=None, start_state=None,
):
    """
    Arg List
    env: The environment to do the rollout on. This is treated as a derstructible copy.
    U_old: A base open loop control sequence
    k: open loop gain (iLQR)
    K: closed loop gain (iLQR)
    X_old: Previous trajectory to compute gain (iLQR)
    alpha: Optional multplier to the open loop gain (iLQR)
    D: Optional noise vectors for the rollout (GPC)
    F: Linearization shift ??? (GPC)
    H: The horizon length to perform a rollout.
    """
    if H is None:
        H = env.H
    X, U = [None for _ in range(H + 1)], [None for _ in range(H)]
    if start_state is not None:
        X[0], cost = env.reset_model(start_state), 0.0
    else:
        X[0], cost = env.reset(), 0.0
    for h in range(H):
        if k is None:
            U[h] = U_old[h]
        else:
            U[h] = U_old[h] + alpha * k[h] + K[h] @ (X[h] - X_old[h])

        if D is None:
            X[h + 1], instant_cost, _, _ = env.step(U[h])
        elif F is None:
            X[h + 1], instant_cost, _, _ = env.step(U[h])
            X[h + 1] += D[h]
        else:
            X[h + 1] = X_old[h + 1] + F[h][0] @ (X[h] - X_old[h]) + F[h][1] @ (U[h] - U_old[h])
            instant_cost = env.c(X[h], U[h], h)

        cost += instant_cost
    return X, U, cost


def LQR(F, C):
    H, d = len(C), C[0][0].shape[0]
    K, k = (
        [None for _ in range(H)],
        [None for _ in range(H)],
    )
    V_x, V_xx = np.zeros(d), np.zeros((d, d))
    for h in range(H - 1, -1, -1):
        f_x, f_u = F[h]
        c_x, c_u, c_xx, c_uu = C[h]
        Q_x, Q_u = c_x + f_x.T @ V_x, c_u + f_u.T @ V_x
        Q_xx, Q_ux, Q_uu = (
            c_xx + f_x.T @ V_xx @ f_x,
            f_u.T @ V_xx @ f_x,
            c_uu + f_u.T @ V_xx @ f_u,
        )

        K[h], k[h] = -np.linalg.inv(Q_uu) @ Q_ux, -np.linalg.inv(Q_uu) @ Q_u
        V_x = Q_x - K[h].T @ Q_uu @ k[h]
        V_xx = Q_xx - K[h].T @ Q_uu @ K[h]
        V_xx = (V_xx + V_xx.T) / 2
    return k, K
