import numpy as np
import jax
import jax.numpy as jnp
import sys
import functools
import matplotlib.pyplot as plt
import optax

# Add project root to path
sys.path.append('../../')
from deluca.experimental.enviornments.sim_and_output.lds import lds_sim
from deluca.experimental.agents.sfc import compute_filter_matrix

def loss_fn(M_spectral, state, u_vector, filter_matrix):
    """
    Computes the loss for the spectral system identification model.
    The gradient of this function is taken with respect to M_spectral.
    """
    # 1. Reconstruct the full model M from the learned spectral representation
    M_full = jnp.tensordot(filter_matrix, M_spectral, axes=[[1], [0]])

    # 2. Predict the next state by applying the learned M_full to the action history
    y_hat = jnp.einsum('moi,mi1->o1', M_full, jax.lax.stop_gradient(u_vector))

    # 3. The loss is the squared difference between the prediction and the actual state
    return jnp.sum((jax.lax.stop_gradient(state) - y_hat)**2)

def loss_fn_offline(M_spectral, state, u_vector, state_vector,filter_matrix):
    """
    Computes the loss for the spectral system identification model.
    The gradient of this function is taken with respect to M_spectral.
    """
    # 1. Reconstruct the full model M from the learned spectral representation
    M_spectral_state, M_spectral_action = M_spectral
    M_full_state = jnp.tensordot(filter_matrix, M_spectral_state, axes=[[1], [0]])
    M_full_action = jnp.tensordot(filter_matrix, M_spectral_action, axes=[[1], [0]])

    # 2. Predict the next state by applying the learned M_full to the action history
    y_hat_state = jnp.einsum('moi,mi1->o1', M_full_state, jax.lax.stop_gradient(state_vector))
    y_hat_action = jnp.einsum('moi,mi1->o1', M_full_action, jax.lax.stop_gradient(u_vector))

    y_hat = y_hat_state + y_hat_action
    # 3. The loss is the squared difference between the prediction and the actual state
    return jnp.sum((jax.lax.stop_gradient(state) - y_hat)**2)

# JIT-compile a single training step for performance
@functools.partial(jax.jit, static_argnames=['optimizer'])
def update_step(M_spectral, opt_state, optimizer, state, u_vector, filter_matrix):
    """Performs one step of gradient descent."""
    loss, grads = jax.value_and_grad(loss_fn)(M_spectral, state, u_vector, filter_matrix)
    updates, opt_state = optimizer.update(grads, opt_state, M_spectral)
    M_spectral = optax.apply_updates(M_spectral, updates)
    return M_spectral, opt_state, loss

@functools.partial(jax.jit, static_argnames=['optimizer'])
def update_step_offline(M_spectral_state, M_spectral_action, opt_state, optimizer, state, u_vector, state_vector, filter_matrix):
    """Performs one step of gradient descent."""
    loss, grads = jax.value_and_grad(loss_fn_offline)((M_spectral_state, M_spectral_action), state, u_vector, state_vector, filter_matrix)
    updates, opt_state = optimizer.update(grads, opt_state, (M_spectral_state, M_spectral_action))
    M_spectral_state, M_spectral_action = optax.apply_updates((M_spectral_state, M_spectral_action), updates)
    return M_spectral_state, M_spectral_action, opt_state, loss

def run_offline_experiment(
    key,
    sim_fn,
    experiment_name: str,
    d_in: int,
    d_out: int,
    T_history: int = 100,
    k_features: int = 24,
    gamma: float = 0,
    learning_rate: float = 1e-3,
    training_steps: int = 20000,
):
    # 1. Initialize optimizer and parameters for the model we are training
    optimizer = optax.adam(learning_rate=learning_rate)
    filter_matrix = compute_filter_matrix(m=T_history, h=k_features, gamma=gamma)
    
    key, subkey1, subkey2 = jax.random.split(key, 3)
    M_spectral_state = jax.random.normal(subkey1, (k_features, d_out, d_out)) * 0.1
    M_spectral_action = jax.random.normal(subkey2, (k_features, d_in, d_out)) * 0.1
    opt_state = optimizer.init((M_spectral_state, M_spectral_action))

    # Initialize state and history
    key, subkey = jax.random.split(key)
    
    state = jax.random.normal(subkey, (d_out, 1))
    loss_history = []


    print(f"\n--- Starting Experiment: {experiment_name} ---")
    print(f"Training for {training_steps} steps...")
    
    def evolve_state(state, u):
        return sim_fn(state, u), state
    
    for t in range(training_steps):
        key, subkey = jax.random.split(key)
        u_vector = jax.random.normal(subkey, (T_history, d_in, 1))
        final_state, state_vector = jax.lax.scan(evolve_state, state, u_vector)
         # Perform one update step
        M_spectral_state, M_spectral_action, opt_state, loss = update_step_offline(
            M_spectral_state, M_spectral_action, opt_state, optimizer, final_state, u_vector, state_vector, filter_matrix
        )
        loss_history.append(float(loss))
        if t % 1000 == 0:
            print(f"Step {t:5d}, Loss: {loss:.6f}")


    print("Training finished.")

    # 3. Plot and save the results
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Training Steps (x100)')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(f'Training Loss Over Time ({experiment_name})')
    plt.grid(True)
    filename = f"spectral_sysid_{experiment_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Loss plot saved to {filename}")

    return M_spectral_state, M_spectral_action


    
    
    
def run_sysid_experiment(
    key,
    sim_fn,
    experiment_name: str,
    d_in: int,
    d_out: int,
    T_history: int = 100,
    k_features: int = 24,
    gamma: float = 0,
    learning_rate: float = 1e-3,
    training_steps: int = 1000000,
):
    """
    Runs a system identification experiment for a given simulation function.
    """
    # 1. Initialize optimizer and parameters for the model we are training
    optimizer = optax.adam(learning_rate=learning_rate)
    filter_matrix = compute_filter_matrix(m=T_history, h=k_features, gamma=gamma)
    
    key, subkey = jax.random.split(key)
    M_spectral = jax.random.normal(subkey, (k_features, d_in, d_out)) * 0.1
    opt_state = optimizer.init(M_spectral)

    # Initialize state and history
    key, subkey = jax.random.split(key)
    state = jax.random.normal(subkey, (d_out, 1))
    u_vector = jnp.zeros((T_history, d_in, 1))
    loss_history = []

    print(f"\n--- Starting Experiment: {experiment_name} ---")
    print(f"Training for {training_steps} steps...")
    # 2. Run the training loop
    for t in range(training_steps):
        key, subkey = jax.random.split(key)
        u = jax.random.normal(subkey, (d_in, 1))
        

        u_vector = u_vector.at[0].set(u)
        u_vector = jnp.roll(u_vector, shift=-1, axis=0)
        
        # Generate the "true" state from the provided simulation function
        state = sim_fn(state, u)

        # Perform one update step
        M_spectral, opt_state, loss = update_step(
            M_spectral, opt_state, optimizer, state, u_vector, filter_matrix
        )

        # Log the loss
        if t % 100 == 0:
            loss_history.append(float(loss))
            if t % 200 == 0:
                print(f"Step {t:5d}, Loss: {loss:.6f}")

    print("Training finished.")

    # 3. Plot and save the results
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Training Steps (x100)')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(f'Training Loss Over Time ({experiment_name})')
    plt.grid(True)
    filename = f"spectral_sysid_{experiment_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Loss plot saved to {filename}")

    return M_spectral

if __name__ == '__main__':
    # --- Configuration ---
    M = 100
    K = 30
    D_IN = 1
    D_OUT = 2
    main_key = jax.random.PRNGKey(0)
    key, subkey1 = jax.random.split(main_key)
    
    # --- Experiment 3: Original Problem ---
    # Goal: Approximate a stable IIR system with our FIR model.
    # Expectation: The loss will plateau at the best possible non-zero approximation.
    key, subkey3 = jax.random.split(key)
    A = jnp.array([[0.9, 0.1], [0.1, 0.8]]) # Stable A matrix
    B = jnp.array([[1.0], [0.0]])
    sim = lambda state, u: lds_sim(state, u, A=A, B=B)

    # run_sysid_experiment(
    #     key=subkey3,
    #     sim_fn=sim,
    #     experiment_name="FIR vs IIR",
    #     d_in=D_IN,
    #     d_out=D_OUT,
    #     T_history=M,
    #     k_features=K
    # ) 
    
    
    run_offline_experiment(
        key=subkey3,
        sim_fn=sim,
        experiment_name="offline",
        d_in=D_IN,
        d_out=D_OUT,
        T_history=M,
        k_features=K
    )