import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import hilbert
import sys 
import functools
sys.path.append('../../')
from deluca.experimental.enviornments.sim_and_output.lds import lds_sim, lds_output
from deluca.experimental.enviornments.sim_and_output.pendulum import pendulum_sim, pendulum_output
from deluca.experimental.agents.sfc import get_sfc_features, compute_filter_matrix

from scipy.linalg import eigh as largest_eigh

def loss_fn(M_specrtal_parameter, state, u_vector, filter_matrix):
    # print(filter_matrix.shape, u_vector.shape, M_specrtal_parameter.shape)
    # temp = jnp.einsum('mh,md1->hd1', filter_matrix, u_vector) # k, d_in, 1
    # temp = filter_matrix.T @ u_vector
    temp = jnp.tensordot(filter_matrix, u_vector, axes=[0,0])
    # M_spectral_parameter: (k, d_in, d_out)
    # temp: (k, d_in, 1)
    # Contract along k and d_in dimensions to get (d_out, 1)
    y_hat = jnp.tensordot(M_specrtal_parameter, temp, axes=[[0,1], [0,1]])
    #   y_hat = jnp.einsum('kdm,kd1->m1', M_specrtal_parameter, temp) # d_out, 1

    return jnp.sum((state - y_hat)**2)

grad_fn = jax.grad(loss_fn, argnums=0)

def get_filters(L: int, k: int = 24):
    k = min(k, L)
    i = np.arange(1, L + 1)
    i_plus_j = i[None] + i[:, None]
    Z = 2 / (i_plus_j ** 3 - i_plus_j)
    Z = np.float64(Z)
    evals, evecs = largest_eigh(Z)
    return jnp.array(evecs[::-1, -k:] * (evals[-k:] ** 0.25))

# Create a Hilbert matrix of order T
def spectral_sysid(key, sim , d_in, d_out, T = 1000, k = 24):
    key, subkey = jax.random.split(key)
    filter_matrix = get_filters(T, k)
    print(filter_matrix.shape)
    state = jax.random.normal(subkey,(d_out,1))
    u_vector = jnp.zeros((T,d_in,1))
    M_specrtal_parameter = jnp.zeros((k, d_in, d_out))
    for t in range(10000):
        key, subkey = jax.random.split(key)
        u = jax.random.normal(subkey,(d_in,1))
        # Update disturbance history
        u_vector = u_vector.at[0,:,:].set(u)
        u_vector = jnp.roll(u_vector, shift=-1, axis=0)
        state = sim(state, u)
        # temp = jnp.einsum('mh,md1->hd1', filter_matrix, u_vector) # k, d_in, 1
        # y_hat = jnp.einsum('kdm,kd1->m1', M_specrtal_parameter, temp) # d_out, 1
        loss = loss_fn(M_specrtal_parameter, state, u_vector, filter_matrix)
        grads = grad_fn(M_specrtal_parameter, state, u_vector, filter_matrix)
        M_specrtal_parameter = M_specrtal_parameter - 0.0001 * grads
        # M_specrtal_parameter = M_specrtal_parameter - 0.001 * temp @ (state - y_hat).T
    
        print(f"Loss: {loss}")
        # Store loss values in a list
        if t == 0:
            loss_history = []
        loss_history.append(float(loss))
        
        # At the end of training, create and save the plot
        if t == T-1:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,6))
            plt.plot(loss_history)
            plt.xlabel('Training Steps')
            plt.ylabel('Loss') 
            plt.yscale('log')
            plt.title('Training Loss Over Time')
            plt.grid(True)
            plt.savefig('spectral_sysid_loss.png')
            plt.close()
    return M_specrtal_parameter 

# sim = functools.partial(pendulum_sim, max_torque=10, m=1, l=1, g=9.81, dt=0.01)
sim = functools.partial(lds_sim, A=jnp.array([[0.9, 0.1], [0.1, 0.9]]), B=jnp.array([[1], [0]]))
spectral_sysid(jax.random.PRNGKey(0), sim, 1, 2)