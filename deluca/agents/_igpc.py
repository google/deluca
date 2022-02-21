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

import jax
import jax.numpy as jnp

from deluca.agents.core import Agent
from deluca.utils.planning import f_at_x
from deluca.utils.planning import LQR
from deluca.utils.planning import rollout

# from deluca.agents._ilqr import LQR


def counterfact_loss(E, off, W, H, M, x, t, env_sim, U_old, k, K, X_old, D, F, w_is, alpha, C):
    y, cost = x, 0
    for h in range(H):
        u_delta = jnp.tensordot(E, W[h : h + M], axes=([0, 2], [0, 1])) + off
        u = U_old[t + h] + alpha * k[t + h] + K[t + h] @ (y - X_old[t + h]) + C * u_delta
        cost = env_sim.c(y, u)

        if w_is == "de":
            y = env_sim.f(y, u) + W[h + M]
        elif w_is == "dede":
            y = env_sim.f(y, u) + D[h + M] + W[h + M]
        else:
            y = (
                X_old[h + M + 1]
                + F[h + M][0] @ (y - X_old[h + M])
                + F[h + M][1] @ (u - U_old[h + M])
                + W[h + M]
            )
    return cost


grad_loss = jax.grad(counterfact_loss, (0, 1))


def rolls_by_igpc(env_true, env_sim, U_old, k, K, X_old, D, F, alpha, w_is, ref_lr=0.1):
    # print(ref_lr)
    H, M, lr, C = 3, 3, ref_lr, 1.0
    X, U, U_delta = (
        [None for _ in range(env_sim.H + 1)],
        [None for _ in range(env_sim.H)],
        [None for _ in range(env_sim.H)],
    )
    W, E, off = (
        jnp.zeros((H + M, env_sim.state_dim)),
        jnp.zeros((H, env_sim.action_dim, env_sim.state_dim)),
        jnp.zeros(env_sim.action_dim),
    )
    X[0], cost = env_true.reset(), 0
    for h in range(env_true.H):
        U_delta[h] = jnp.tensordot(E, W[-M:], axes=([0, 2], [0, 1])) + off
        U[h] = U_old[h] + alpha * k[h] + K[h] @ (X[h] - X_old[h]) + C * U_delta[h]
        X[h + 1], instant_cost, _, _ = env_true.step(U[h])
        cost += instant_cost

        # 3 choices for w are f-g, (f-g)-(f-g last time), (f- LIN(g) with OFF).
        if w_is == "de":
            w = X[h + 1] - env_sim.f(X[h], U[h])
        elif w_is == "dede":
            w = (X[h + 1] - env_sim.f(X[h], U[h])) - D[h]
        else:
            w = X[h + 1] - (
                X_old[h + 1] + F[h][0] @ (X[h] - X_old[h]) + F[h][1] @ (U[h] - U_old[h])
            )

        W = W.at[0].set(w)
        W = jnp.roll(W, -1, axis=0)
        if h >= H:
            delta_E, delta_off = grad_loss(
                E, off, W, H, M, X[h - H], h - H, env_sim, U_old, k, K, X_old, D, F, w_is, alpha, C,
            )
            E -= lr / h * delta_E
            off -= lr / h * delta_off
    # print(norm_U_old, norm_other, norm_off, norm_U_delta, norm_U_vals)
    return X, U, cost


def igpc_closed(
    env_true, env_sim, U, T, k=None, K=None, X=None, w_is="de", lr=None, ref_alpha=1.0, log=None
):
    alpha, r = ref_alpha, 1
    X, U, c = rollout(env_true, U, k, K, X)
    print("initial cost:" + str(c))
    assert w_is in ["de", "dede", "delin"]
    for t in range(T):
        D, F, C = f_at_x(env_sim, X, U)
        k, K = LQR(F, C)
        for alphaC in alpha * 1.1 * 1.1 ** (-jnp.arange(10) ** 2):
            r += 1
            XC, UC, cC = rolls_by_igpc(env_true, env_sim, U, k, K, X, D, F, alphaC, w_is, lr)
            if cC <= c:
                X, U, c, alpha = XC, UC, cC, alphaC
                print(f"(iGPC): t = {t}, r = {r}, c = {c}, alpha = {alpha}")
                if log is not None:
                    log.append(f"(iGPC): t = {t}, r = {r}, c = {c}")
                break
        else:
            return X, U, k, K, c
    return X, U, k, K, c


class IGPCInner:
    def __init__(self, env_true, env_sim):
        self.env_true = env_true
        self.env_sim = env_sim

    def train(self, T, U_init=None, k=None, K=None, X=None, w_is="de", lr=None, ref_alpha=1.0):
        self.log = []
        if U_init is None:
            U_init = [jnp.zeros(self.env_true.action_dim) for _ in range(self.env_true.H)]
        X, U, k, K, c = igpc_closed(
            self.env_true, self.env_sim, U_init, T, k, K, X, w_is, lr, ref_alpha, self.log
        )
        return X, U, k, K, c, self.log


class IGPC(Agent):
    def __init__(self):
        self.reset()

    def train(
        self,
        env_true,
        env_sim,
        train_steps,
        U_init=None,
        k=None,
        K=None,
        X=None,
        w_is="de",
        lr=None,
        ref_alpha=1.0,
    ):
        self.env_true = env_true
        self.env_sim = env_sim
        igpc_agent = IGPCInner(self.env_true, self.env_sim)
        X, U, k, K, c, log = igpc_agent.train(train_steps, U_init, k, K, X, w_is, lr, ref_alpha)
        self.U = jnp.array(U)
        self.X = jnp.array(X)
        self.k = jnp.array(k)
        self.K = jnp.array(K)
        return c, log

    def reset(self):
        self.t = 0

    def __call__(self, state):
        action = self.U[self.t]
        self.t += 1

        return action
