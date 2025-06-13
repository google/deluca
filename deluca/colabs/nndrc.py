import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze, unfreeze
from flax.training import train_state
import optax
from functools import partial
from typing import Sequence, Callable
import numpy as np

# ---------- Neural Network Definition (Flax) ----------
class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

# ---------- NN-drc Controller State and Initialization ----------
class TrainState(train_state.TrainState):
    pass

def create_train_state(key, model, learning_rate, input_shape):
    """Creates initial `TrainState`."""
    params = model.init(key, jnp.ones(input_shape))['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def nndrc_init_controller_state(key, learning_rate, d, n, p, m, k, nn_features):
    """Initializes the state for the NN-drc controller."""
    # State for the neural network
    model = MLP(features=nn_features)
    input_shape = (m * p,)
    train_state = create_train_state(key, model, learning_rate, input_shape)

    # State for the system dynamics tracking
    z = jnp.zeros((d, 1))
    ynat_history = jnp.zeros((2 * m, p, 1))
    controller_t = 0
    return train_state, z, ynat_history, controller_t

# ---------- NN-drc Policy Loss and Action ----------
def nndrc_policy_loss(
    params, apply_fn, ynat_history_for_loss,
    cost_evaluate_fn, Q_cost, R_cost,
    m_controller, k_future, d, p, sim, output_map
):
    nn_input_shape = (m_controller * p,)
    
    # Phase 1: Evolve for final_delta (as described by user)
    def evolve_for_loss(delta, k_step):
        # For each step, call NN to get one action
        window = jax.lax.dynamic_slice(ynat_history_for_loss, (k_step, 0, 0), (m_controller, p, 1))
        nn_input = window.flatten()
        
        # We need a batch dimension for the model
        u_sequence = apply_fn({'params': params}, nn_input[jnp.newaxis, :])[0]
        u = u_sequence[:1] # Take first action
        u_reshaped = u.reshape(-1, 1)

        next_delta = sim(delta, u_reshaped)
        return next_delta, None

    # Evolve m_controller steps to find the starting state for future predictions
    initial_delta = jnp.zeros((d, 1))
    final_delta, _ = jax.lax.scan(evolve_for_loss, initial_delta, jnp.arange(m_controller))
    
    # Phase 2: Predict k future actions and calculate total cost
    # Get the most recent history window
    window_future = jax.lax.dynamic_slice(ynat_history_for_loss, (m_controller, 0, 0), (m_controller, p, 1))
    nn_input_future = window_future.flatten()
    
    # Call NN to get k_future actions
    future_actions = apply_fn({'params': params}, nn_input_future[jnp.newaxis, :])[0]
    future_actions = future_actions.reshape(k_future, -1, 1) # Shape: (k, n, 1)

    def future_cost_step(carry, u_step):
        current_delta, total_cost = carry
        next_delta = sim(current_delta, u_step)
        y_pred = output_map(next_delta)
        cost = cost_evaluate_fn(y_pred, u_step, Q_cost, R_cost)
        return (next_delta, total_cost + cost), None

    # Sum costs over the k_future horizon
    (_, total_future_cost), _ = jax.lax.scan(future_cost_step, (final_delta, 0.0), future_actions)
    
    return total_future_cost

def nndrc_get_action(params, apply_fn, ynat_history_buffer, m_controller, p):
    """Get the next action from the NN-drc controller."""
    # Use the most recent m steps of ynat_history
    window_for_action = jax.lax.dynamic_slice(ynat_history_buffer, (0, 0, 0), (m_controller, p, 1))
    nn_input = window_for_action.flatten()
    
    # Get sequence of k actions, but only use the first one for the current time step
    u_sequence = apply_fn({'params': params}, nn_input[jnp.newaxis, :])[0]
    u = u_sequence[:1] # Take first action
    return u.reshape(-1, 1)

# ---------- NN-drc Update Step ----------
@partial(jax.jit, static_argnames=('loss_fn', 'apply_lr_decay', 'sim', 'output_map'))
def nndrc_update_controller(
    train_state, z, ynat_history_buffer, controller_t,
    y_meas, u_taken,
    lr_base, apply_lr_decay,
    loss_fn,
    sim, output_map
):
    """Performs a single update step for the NN-drc controller."""
    # 1. Update z state and ynat_history
    z_next = sim(z, u_taken)
    y_nat_current = y_meas - output_map(z_next)
    ynat_history_updated = jnp.roll(ynat_history_buffer, shift=1, axis=0).at[0].set(y_nat_current)

    # 2. Compute gradients and update TrainState
    grad_fn = jax.value_and_grad(loss_fn)
    loss_val, grads = grad_fn(train_state.params, ynat_history_updated)
    
    # Optional: Learning rate decay
    # Note: optax Adam already has its own adaptive LR per parameter. 
    # A manual decay might not be standard but can be added if needed.
    # For now, we rely on Adam's adaptivity. If decay is needed, we'd adjust the optimizer state.
    
    new_train_state = train_state.apply_gradients(grads=grads)
    
    controller_t_next = controller_t + 1
    
    return new_train_state, z_next, ynat_history_updated, controller_t_next, loss_val

# ---------- Simulation for NN-drc ----------
def run_single_nndrc_trial(
    key_trial_run, A, B, C, Q_cost, R_cost, x0, sim, output_map,
    T_sim_steps, 
    nndrc_m, nndrc_k, nndrc_lr, nndrc_lr_decay,
    gaussian_noise_std, lds_transition_type, lds_noise_type, sinusoidal_noise_amp, sinusoidal_noise_freq, # System params
    step_fn, lds_gaussian_step_fn, lds_sinusoidal_step_fn, # Step functions
    disturbance_fn,
    cost_evaluate_fn # Cost function
):
    d, n, p = A.shape[0], B.shape[1], C.shape[0]
    
    # Define NN structure
    nn_features = [4, nndrc_k * n]
    model = MLP(features=nn_features)

    # --- Loss function for gradient computation ---
    def _nndrc_loss_for_grad(params, ynat_hist):
        return nndrc_policy_loss(
            params, model.apply, ynat_hist,
            cost_evaluate_fn, Q_cost, R_cost,
            nndrc_m, nndrc_k, d, p, sim, output_map
        )

    key_nndrc_init, key_sim_loop_main = jax.random.split(key_trial_run)
    
    train_state_init, z_init, ynat_hist_init, t_init = nndrc_init_controller_state(
        key_nndrc_init, nndrc_lr, d, n, p, nndrc_m, nndrc_k, nn_features
    )

    current_x_state = x0
    initial_sim_time_step = 0

    def nndrc_simulation_step(carry, key_step_specific):
        prev_x, prev_train_state, prev_z, prev_ynat_hist, prev_t, prev_sim_t = carry

        # 1. Get action from controller
        u_control_t = nndrc_get_action(
            prev_train_state.params, 
            prev_train_state.apply_fn,
            prev_ynat_hist, nndrc_m, p
        )

        # 2. Step the system dynamics
        # This part is complex because of different system/noise types. 
        # We pass in the required functions to keep this general.
        if lds_transition_type == "special":
             next_x, y_measured_t = step_fn(
                x=prev_x, u=u_control_t, sim=sim, output_map=output_map,
                dist_std=gaussian_noise_std, t_sim_step=prev_sim_t,
                disturbance=disturbance_fn, key=key_step_specific
            )
        elif lds_noise_type == "gaussian":
            next_x, y_measured_t = lds_gaussian_step_fn(
                current_x=prev_x, A=A, B=B, C=C, u_t=u_control_t,
                dist_std=gaussian_noise_std, t_sim_step=prev_sim_t,
                key_disturbance=key_step_specific
            )
        elif lds_noise_type == "sinusoidal":
            next_x, y_measured_t = lds_sinusoidal_step_fn(
                current_x=prev_x, A=A, B=B, C=C, u_t=u_control_t,
                noise_amplitude=sinusoidal_noise_amp, noise_frequency=sinusoidal_noise_freq,
                t_sim_step=prev_sim_t, key_disturbance=key_step_specific
            )
        else:
            # Fallback or error
            # For simplicity, we assume valid combo is checked outside
            next_x, y_measured_t = prev_x, output_map(prev_x)
        
        # 3. Update controller state
        next_train_state, next_z, next_ynat_hist, next_t, loss_val = nndrc_update_controller(
            prev_train_state, prev_z, prev_ynat_hist, prev_t,
            y_measured_t, u_control_t,
            nndrc_lr, nndrc_lr_decay,
            _nndrc_loss_for_grad,
            sim, output_map
        )
        
        # 4. Calculate cost for this step
        cost_at_t = cost_evaluate_fn(y_measured_t, u_control_t, Q_cost, R_cost)
        
        next_sim_t = prev_sim_t + 1
        next_carry = (next_x, next_train_state, next_z, next_ynat_hist, next_t, next_sim_t)
        outputs_to_collect = (cost_at_t, prev_x, u_control_t)
        return next_carry, outputs_to_collect

    initial_carry = (current_x_state, train_state_init, z_init, ynat_hist_init, t_init, initial_sim_time_step)
    keys_for_scan = jax.random.split(key_sim_loop_main, T_sim_steps)
    
    (_, (costs_over_time, trajectory, u_history)) = jax.lax.scan(nndrc_simulation_step, initial_carry, keys_for_scan)
    
    return costs_over_time, trajectory, u_history 