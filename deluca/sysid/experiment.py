import numpy as np
import jax
import jax.numpy as jnp
import sys
import functools
import matplotlib.pyplot as plt
import optax
from flax import linen as nn

from typing import Sequence, Optional

# Add project root to path
sys.path.append('../../')
from deluca.experimental.enviornments.sim_and_output.lds import lds_sim
from deluca.experimental.enviornments.sim_and_output.pendulum import pendulum_sim

from deluca.experimental.agents.sfc import compute_filter_matrix
from deluca.experimental.agents.mpc import MPCModel
from deluca.experimental.enviornments.disturbances.sinusoidal import sinusoidal_disturbance
from deluca.experimental.enviornments.disturbances.gaussian import gaussian_disturbance
from deluca.experimental.enviornments.disturbances.zero import zero_disturbance
from deluca.experimental.agents.model import FullyConnectedModel, SequentialModel, GridModel


class SysIDModel(nn.Module):
    """
    Fully connected model that takes input of shape (h, d_in, 1) and outputs shape (d_out, 1).
    If hidden_dims is None, it's a linear model.
    """
    h: int  # Input sequence length
    d_in: int  # Input dimension
    d_out: int  # Output dimension
    hidden_dims: Optional[Sequence[int]] = None
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is of shape (h, d_in, 1)
        # Flatten input to (h*d_in, 1)
        x_flat = x.reshape(-1)
        if self.hidden_dims is None:
            # Linear layer
            y = nn.Dense(self.d_out, use_bias=self.use_bias)(x_flat)  # output (1, d_out)
        else:
            # Neural network with hidden layers
            y = x_flat
            for hidden_dim in self.hidden_dims:
                y = nn.Dense(hidden_dim, use_bias=self.use_bias)(y)
                y = nn.relu(y)
            y = nn.Dense(self.d_out, use_bias=self.use_bias)(y)  # output (1, d_out)
        # Reshape to (d_out, 1)
        return y.reshape((self.d_out, 1))
    
    
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

def loss_fn_nn_offline(model_state_params, model_action_params, state, u_vector, state_vector, filter_matrix, model_state_apply, model_action_apply):
    """
    Computes the loss for the neural network system identification model.
    The gradient of this function is taken with respect to model_state_params and model_action_params.
    """
    # 1. Predict the next state by applying the learned models to the filtered history
    filtered_state_vector = jnp.tensordot(filter_matrix, state_vector, axes=[[0], [0]])
    filtered_u_vector = jnp.tensordot(filter_matrix, u_vector, axes=[[0], [0]])
    
    y_hat_state = model_state_apply(model_state_params, filtered_state_vector)
    y_hat_action = model_action_apply(model_action_params, filtered_u_vector)
    y_hat = y_hat_state + y_hat_action
    # 2. The loss is the squared difference between the prediction and the actual state
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

def create_update_step_nn_offline(model_state_apply, model_action_apply):
    """Creates a JIT-compiled update step with the model apply functions captured in closure."""
    
    def loss_fn_with_models(model_state_params, model_action_params, state, u_vector, state_vector, filter_matrix):
        return loss_fn_nn_offline(model_state_params, model_action_params, state, u_vector, state_vector, filter_matrix, model_state_apply, model_action_apply)
    
    @functools.partial(jax.jit, static_argnames=['optimizer'])
    def update_step_nn_offline(model_state_params, model_action_params, opt_state, optimizer, state, u_vector, state_vector, filter_matrix):
        """Performs one step of gradient descent."""
        loss, grads = jax.value_and_grad(loss_fn_with_models, argnums=(0, 1))(model_state_params, model_action_params, state, u_vector, state_vector, filter_matrix)
        updates, opt_state = optimizer.update(grads, opt_state, (model_state_params, model_action_params))
        model_state_params, model_action_params = optax.apply_updates((model_state_params, model_action_params), updates)
        return model_state_params, model_action_params, opt_state, loss
    
    return update_step_nn_offline

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
    
    # key, subkey1, subkey2 = jax.random.split(key, 3)
    # M_spectral_state = jax.random.normal(subkey1, (k_features, d_out, d_out)) * 0.1
    # M_spectral_action = jax.random.normal(subkey2, (k_features, d_in, d_out)) * 0.1
    
    model_state = SysIDModel(h=k_features, d_in=d_out, d_out=d_out, hidden_dims=[16])
    model_action = SysIDModel(h=k_features, d_in=d_in, d_out=d_out, hidden_dims=[16])
    model_state_params = model_state.init(key, jnp.zeros((k_features, d_out, 1)))
    model_action_params = model_action.init(key, jnp.zeros((k_features, d_in, 1)))
    opt_state = optimizer.init((model_state_params, model_action_params))

    # Create the update step function with the model apply functions
    update_step_nn_offline = create_update_step_nn_offline(model_state.apply, model_action.apply)

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
        model_state_params, model_action_params, opt_state, loss = update_step_nn_offline(
            model_state_params, model_action_params, opt_state, optimizer, final_state, u_vector, state_vector, filter_matrix
        )
        loss_history.append(float(loss))
        if t % 1000 == 0:
            print(f"Step {t:5d}, Loss: {loss:.6f}")


    print("Training finished.")

    # 3. Plot and save the results
    window_size = 100
    windowed_loss = [sum(loss_history[i:i+window_size])/window_size 
                    for i in range(0, len(loss_history)-window_size+1)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(windowed_loss)
    plt.xlabel('Training Steps (x100)')
    plt.ylabel('Loss (100-step moving average)')
    plt.yscale('log')
    plt.title(f'Training Loss Over Time ({experiment_name})')
    plt.grid(True)
    filename = f"spectral_sysid_{experiment_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Loss plot saved to {filename}")

    return (model_state.apply, model_action.apply), (model_state_params, model_action_params)


def run_offline_eval(
    key,
    sim_fn,
    M_spectral_state,
    M_spectral_action,
    experiment_name: str,
    d_in: int,
    d_out: int,
    T_history: int = 100,
    k_features: int = 24,
    gamma: float = 0,
    eval_steps: int = 1000,
):
    """
    Evaluates the learned spectral model on a fresh trajectory without any training.
    
    Args:
        key: Random key for generating test data
        sim_fn: True system dynamics function
        M_spectral_state: Learned spectral matrix for state dynamics
        M_spectral_action: Learned spectral matrix for action dynamics
        experiment_name: Name for saving plots
        d_in: Input dimension
        d_out: Output dimension
        T_history: History length
        k_features: Number of spectral features
        gamma: Decay parameter for filter matrix
        eval_steps: Number of evaluation steps
    
    Returns:
        List of prediction losses over the evaluation trajectory
    """
    print(f"\n--- Starting Evaluation: {experiment_name} ---")
    print(f"Evaluating for {eval_steps} steps...")
    
    # Compute filter matrix (same as used in training)
    filter_matrix = compute_filter_matrix(m=T_history, h=k_features, gamma=gamma)
    
    # Initialize random starting state
    key, subkey = jax.random.split(key)
    state = jax.random.normal(subkey, (d_out, 1))
    
    # Initialize action and state history buffers
    u_history = jnp.zeros((T_history, d_in, 1))
    state_history = jnp.zeros((T_history, d_out, 1))
    
    loss_history = []
    
    def evolve_and_predict(carry, t):
        state, u_history, state_history, key = carry
        
        # Generate random action
        key, subkey = jax.random.split(key)
        u = jax.random.normal(subkey, (d_in, 1))
        
        # Update action history
        u_history = u_history.at[0].set(u)
        u_history = jnp.roll(u_history, shift=-1, axis=0)
        
        # Update state history
        state_history = state_history.at[0].set(state)
        state_history = jnp.roll(state_history, shift=-1, axis=0)
        
        # Simulate true next state
        next_state = sim_fn(state, u)
        
        # Make prediction using learned model
        M_full_state = jnp.tensordot(filter_matrix, M_spectral_state, axes=[[1], [0]])
        M_full_action = jnp.tensordot(filter_matrix, M_spectral_action, axes=[[1], [0]])
        
        y_hat_state = jnp.einsum('moi,mi1->o1', M_full_state, state_history)
        y_hat_action = jnp.einsum('moi,mi1->o1', M_full_action, u_history)
        y_hat = y_hat_state + y_hat_action
        
        # Compute prediction loss
        loss = jnp.sum((next_state - y_hat)**2)
        
        new_carry = (next_state, u_history, state_history, key)
        return new_carry, loss
    
    # Run evaluation using scan for efficiency
    initial_carry = (state, u_history, state_history, key)
    _, losses = jax.lax.scan(evolve_and_predict, initial_carry, jnp.arange(eval_steps))
    
    loss_history = losses.tolist()
    
    print(f"Evaluation finished. Mean loss: {jnp.mean(losses):.6f}")
    
    # 3. Plot and save the results
    window_size = 100
    windowed_loss = [sum(loss_history[i:i+window_size])/window_size 
                    for i in range(0, len(loss_history)-window_size+1)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(windowed_loss)
    plt.xlabel('Evaluation Steps')
    plt.ylabel('Prediction Loss (100-step moving average)')
    plt.yscale('log')
    plt.title(f'Evaluation Loss Over Time ({experiment_name})')
    plt.grid(True)
    filename = f"spectral_sysid_{experiment_name.lower().replace(' ', '_')}_eval.png"
    plt.savefig(filename)
    plt.close()
    print(f"Evaluation plot saved to {filename}")
    
    return loss_history


def run_offline_eval_nn(
    key,
    sim_fn,
    model_state_apply,
    model_action_apply,
    model_state_params,
    model_action_params,
    experiment_name: str,
    d_in: int,
    d_out: int,
    T_history: int = 100,
    k_features: int = 24,
    gamma: float = 0,
    eval_steps: int = 1000,
):
    """
    Evaluates the learned neural network model on a fresh trajectory without any training.
    """
    print(f"\n--- Starting NN Evaluation: {experiment_name} ---")
    print(f"Evaluating for {eval_steps} steps...")
    
    # Compute filter matrix (same as used in training)
    filter_matrix = compute_filter_matrix(m=T_history, h=k_features, gamma=gamma)
    
    # Initialize random starting state
    key, subkey = jax.random.split(key)
    state = jax.random.normal(subkey, (d_out, 1))
    
    # Initialize action and state history buffers
    u_history = jnp.zeros((T_history, d_in, 1))
    state_history = jnp.zeros((T_history, d_out, 1))
    
    loss_history = []
    
    def evolve_and_predict(carry, t):
        state, u_history, state_history, key = carry
        
        # Generate random action
        key, subkey = jax.random.split(key)
        u = jax.random.normal(subkey, (d_in, 1))
        
        # Update action history
        u_history = u_history.at[0].set(u)
        u_history = jnp.roll(u_history, shift=-1, axis=0)
        
        # Update state history
        state_history = state_history.at[0].set(state)
        state_history = jnp.roll(state_history, shift=-1, axis=0)
        
        # Simulate true next state
        next_state = sim_fn(state, u)
        
        # Make prediction using learned neural network model
        filtered_state_vector = jnp.tensordot(filter_matrix, state_history, axes=[[0], [0]])
        filtered_u_vector = jnp.tensordot(filter_matrix, u_history, axes=[[0], [0]])
        
        y_hat_state = model_state_apply(model_state_params, filtered_state_vector)
        y_hat_action = model_action_apply(model_action_params, filtered_u_vector)
        y_hat = y_hat_state + y_hat_action
        
        # Compute prediction loss
        loss = jnp.sum((next_state - y_hat)**2)
        
        new_carry = (next_state, u_history, state_history, key)
        return new_carry, loss
    
    # Run evaluation using scan for efficiency
    initial_carry = (state, u_history, state_history, key)
    _, losses = jax.lax.scan(evolve_and_predict, initial_carry, jnp.arange(eval_steps))
    
    loss_history = losses.tolist()
    
    print(f"Evaluation finished. Mean loss: {jnp.mean(losses):.6f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Evaluation Steps')
    plt.ylabel('Prediction Loss')
    plt.yscale('log')
    plt.title(f'NN Evaluation Loss Over Time ({experiment_name})')
    plt.grid(True)
    filename = f"spectral_sysid_{experiment_name.lower().replace(' ', '_')}_nn_eval.png"
    plt.savefig(filename)
    plt.close()
    print(f"NN Evaluation plot saved to {filename}")
    
    return loss_history


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
    # sim = lambda state, u: lds_sim(state, u, A=A, B=B)
    
    # Pendulum environment parameters
    max_torque = 1.0
    m = 1.0  # mass
    l = 1.0  # length
    g = 9.81
    dt = 0.02
    sim = lambda state, u: pendulum_sim(state, u, max_torque=max_torque, m=m, l=l, g=g, dt=dt)

    # run_sysid_experiment(
    #     key=subkey3,
    #     sim_fn=sim,
    #     experiment_name="FIR vs IIR",
    #     d_in=D_IN,
    #     d_out=D_OUT,
    #     T_history=M,
    #     k_features=K
    # ) 
    
    
    (model_state_apply, model_action_apply), (model_state_params, model_action_params) = run_offline_experiment(
        key=subkey3,
        sim_fn=sim,
        experiment_name="offline",
        d_in=D_IN,
        d_out=D_OUT,
        T_history=M,
        k_features=K
    )
    
    # Evaluate the learned model on a fresh trajectory
    key, eval_key = jax.random.split(key)
    eval_losses = run_offline_eval_nn(
        key=eval_key,
        sim_fn=sim,
        model_state_apply=model_state_apply,
        model_action_apply=model_action_apply,
        model_state_params=model_state_params,
        model_action_params=model_action_params,
        experiment_name="offline_nn_evaluation",
        d_in=D_IN,
        d_out=D_OUT,
        T_history=M,
        k_features=K,
        eval_steps=2000
    )
    
    