# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""deluca.agents._ilqr

Todos:
    - Leverage LQR agent
    - Remove ILQRInner
"""
import time

import jax.numpy as jnp

from deluca.agents.core import Agent


def iLQR_loop(env, U_initial, T, log=None):
    U, X, k, K, alpha, t = U_initial, None, None, None, 1.0, 1
    X, U, c = rollout(env, U)
    while t <= T:
        F, C = f_at_x(env, X, U)
        k, K = LQR(F, C)
        for alphaC in alpha * 1.1 * 1.1 ** (-jnp.arange(10) ** 2):
            t += 1
            XC, UC, cC = rollout(env, U, k, K, X, alphaC)
            # print('UC = ' + str(UC))
            print('c = ' + str(c))
            print('cC = ' + str(cC))

            if log is not None:
                log.append((t, float(min(c, cC)), env.nsamples))
            print("iLQR : t = %d, c = %f" % (t, min(c, cC)))

            if cC <= c:
                X, U, c, alpha = XC, UC, cC, alphaC
                break
            if t == T:
                return X, U, k, K, c
    return X, U, k, K, c


def LQR(F, C):
    H, d = len(C), C[0][0].shape[0]
    K, k = (
        [None for _ in range(H)],
        [None for _ in range(H)],
    )
    V_x, V_xx = jnp.zeros(d), jnp.zeros((d, d))
    for h in range(H - 1, -1, -1):
        f_x, f_u = F[h]
        c_x, c_u, c_xx, c_uu = C[h]
        Q_x, Q_u = c_x + f_x.T @ V_x, c_u + f_u.T @ V_x
        Q_xx, Q_ux, Q_uu = (
            c_xx + f_x.T @ V_xx @ f_x,
            f_u.T @ V_xx @ f_x,
            c_uu + f_u.T @ V_xx @ f_u,
        )

        # line may be numerically unstable if Q_uu is the 0 matrix
        K[h], k[h] = -jnp.linalg.inv(Q_uu) @ Q_ux, -jnp.linalg.inv(Q_uu) @ Q_u 
        V_x = Q_x - K[h].T @ Q_uu @ k[h]
        V_xx = Q_xx - K[h].T @ Q_uu @ K[h]
        V_xx = (V_xx + V_xx.T) / 2
    return k, K


def rollout(env, U_old, k=None, K=None, X_old=None, alpha=1.0, render=False):
    X, U = [None for _ in range(env.H + 1)], [None for _ in range(env.H)]
    X[0], cost = env.reset(), 0.0
    for h in range(env.H):
        if k is None:
            U[h] = U_old[h]
        else:
            U[h] = U_old[h] + alpha * k[h] + K[h] @ (X[h] - X_old[h])
        X[h + 1], instant_cost, _, _ = env.step(U[h])
        # print('instant_cost = ' + str(instant_cost))
        # print('cost = ' + str(cost))
        cost += instant_cost
        if render:
            env.render()
            time.sleep(0.01)
    return X, U, cost


def f_at_x(env, X, U):
    F, C = [None for _ in range(env.H)], [None for _ in range(env.H)]
    for h in range(env.H):
        F[h] = env.f_x(X[h], U[h]), env.f_u(X[h], U[h])
        C[h] = (
            env.c_x(X[h], U[h]),
            env.c_u(X[h], U[h]),
            env.c_xx(X[h], U[h]),
            env.c_uu(X[h], U[h]),
        )
    return F, C


class ILQRInner:
    def __init__(self, env):
        self.env = env

    def train(self, T):
        self.log = []

        U = [jnp.zeros(self.env.action_dim) for _ in range(self.env.H)]
        _, U, _, _, _ = iLQR_loop(self.env, U, T, self.log)
        return U


class ILQR(Agent):
    def __init__(self):
        self.reset()

    def train(self, sim, train_steps):
        self.sim = sim
        ilqr_agent = ILQRInner(self.sim)
        self.U = jnp.array(ilqr_agent.train(train_steps))

    def reset(self):
        self.t = 0

    def __call__(self, state):
        action = self.U[self.t]
        self.t += 1

        return action
