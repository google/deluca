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
import jax.scipy.linalg as sla
import jax.numpy as jnp
import numpy as np


def lambda_adjust(reg_lambda, multiplier, consts, increase):
  factor, lambda_min = consts
  if increase:
    multiplier = max(multiplier * factor, factor)
    reg_lambda = max(reg_lambda * multiplier, lambda_min)
  else:
    multiplier = min(1 / factor, multiplier / factor)
    reg_lambda = reg_lambda * multiplier if reg_lambda * multiplier >= lambda_min else 0
  reg_lambda = max(reg_lambda, lambda_min)
  print(
      f"lambda change {increase} - new lambda {reg_lambda} - new multiplier {multiplier}"
  )
  if reg_lambda > 1e10:
    reg_lambda = 1e10
  return reg_lambda, multiplier


def LQR(F, C):
  H, d = len(C), C[0][0].shape[0]
  K, k = (
      [None for _ in range(H)],
      [None for _ in range(H)],
  )

  V_x, V_xx = np.zeros(d), np.zeros((d, d))
  # V_x, V_xx = jnp.zeros(d), jnp.zeros((d, d))

  # F_x, F_u = F
  # C_x, C_u, C_xx, C_uu = C

  # def loop(carry, args):
  # v_x, v_xx = carry
  # f_x, f_u, c_x, c_u, c_xx, c_uu = args

  # q_x, q_u = c_x + f_x.T @ v_x, c_u + f_u.T @ v_x
  # q_xx, q_ux, q_uu = (
  # c_xx + f_x.T @ v_xx @ f_x,
  # f_u.T @ v_xx @ f_x,
  # c_uu + f_u.T @ v_xx @ f_u,
  # )

  # K, k = -jnp.linalg.inv(q_uu) @ q_ux, -jnp.linalg.inv(q_uu) @ q_u
  # v_x = q_x - K.T @ q_uu @ k
  # v_xx = q_xx - K.T @ q_uu @ K
  # v_xx = (v_xx + v_xx.T) / 2

  # return (v_x, v_xx), (K, k)

  # _, (K, k) = jax.lax.scan(loop, (V_x, V_xx), (F_x, F_u, C_x, C_u, C_xx, C_uu))
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


def LQG(F,
        C,
        reg_lambda=0.0,
        multiplier=1.0,
        damping_consts=[None, None],
        reg_type="V"):
  ### Main changes
  ### C must have c_ux also. Otherwise error.
  ### lambda is being updated in case pass fails due to PD. The next lambda and multiplier are returned
  ### gain estimation is also returned.
  H, d, du = len(C), C[0][0].shape[0], C[0][1].shape[0]
  counter = 0
  while True:
    counter += 1
    if counter >= 10:
      print("No good lambda found. Exiting as no solution.")
      return None, None, [reg_lambda, multiplier, None, None]
      break
    K, k = (
        [None for _ in range(H)],
        [None for _ in range(H)],
    )
    V_x, V_xx = np.zeros(d), np.zeros((d, d))
    gains_lin, gains_quad = 0.0, 0.0
    success = True
    for h in range(H - 1, -1, -1):
      # print(h)
      # print(reg_lambda)
      f_x, f_u = F[h]
      c_x, c_u, c_xx, c_uu, c_ux = C[h]
      Q_x, Q_u = c_x + f_x.T @ V_x, c_u + f_u.T @ V_x
      Q_xx, Q_ux, Q_uu = (
          c_xx + f_x.T @ V_xx @ f_x,
          c_ux + f_u.T @ V_xx @ f_x,
          c_uu + f_u.T @ V_xx @ f_u,
      )
      if reg_type == "Q":
        #### Type Q - Dampen Q_uu += lambda I ####
        Q_uu_reg = Q_uu + reg_lambda * np.eye(du)
        Q_ux_reg = Q_ux
      elif reg_type == "V":
        #### Type V - Dampen Q_uu += Q_uu + lambda*f_u f_u.T -- Also Dampen Q_ux += lambda*f_u @ f_x^{T}
        Q_uu_reg = Q_uu + reg_lambda * f_u.T @ f_u
        Q_ux_reg = Q_ux + reg_lambda * f_u.T @ f_x
      QisPD, Q_uu_reg_inv = isPD_and_invert(Q_uu_reg)
      if QisPD:
        #### Check for inversion - if failure break loop and inform #####
        K[h], k[h] = -Q_uu_reg_inv @ Q_ux_reg, -Q_uu_reg_inv @ Q_u
        #### Improved value update
        V_x = Q_x + K[h].T @ Q_uu @ k[h] + K[h].T @ Q_u + Q_ux.T @ k[h]
        V_xx = Q_xx + K[h].T @ Q_uu @ K[h] + K[h].T @ Q_ux + Q_ux.T @ K[h]
        ##### Should not be needed
        V_xx = (V_xx + V_xx.T) / 2.0
        #### Update the gain computation
        # print(k[h].shape)
        # print(Q_u.shape)
        gains_lin += k[h].T @ Q_u
        gains_quad += k[h].T @ Q_uu @ k[h]
      else:
        success = False
        break
    if success:
      break
    else:
      #### lambda will be increased now
      print("Increasing lambda")
      reg_lambda, multiplier = lambda_adjust(reg_lambda, multiplier,
                                             damping_consts, True)
  return k, K, [reg_lambda, multiplier, gains_lin, gains_quad]


def isPD_and_invert(M):
  L = np.linalg.cholesky(M)
  if np.isnan(np.sum(L)):
    return False, None
  L_inverse = sla.solve_triangular(
      L, np.eye(len(L)), lower=True, check_finite=False)
  return True, L_inverse.T.dot(L_inverse)
