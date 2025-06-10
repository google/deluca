import jax
from jax import numpy as jp
import jax.lax as lax # Added import
import jax.tree_util as jtu # Added import
from functools import partial # Added import
from IPython.display import HTML
import matplotlib.pyplot as plt

import brax
import flax # Required by brax.physics.base.State
from brax import envs
from brax.io import html

# # Enable JAX 64-bit mode
# jax.config.update("jax_enable_x64", True)

env_name = 'inverted_pendulum'
backend = 'generalized'  # @param ['generalized', 'positional', 'spring']

env = envs.get_environment(env_name=env_name,
                           backend=backend)
# Get a template state from a reset, and the JITted step function
template_brax_state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))
jit_env_step = jax.jit(env.step) # JIT env.step once
env_dt_for_plotting = env.dt # Store dt for plotting


def solve_dare_iterative(A, B, Q, R, num_iterations=150, tolerance=1e-7):
    """Solves the discrete-time algebraic Riccati equation (DARE) iteratively."""
    P = Q
    for _ in range(num_iterations):
        P_prev = P
        try:
            R_plus_BTPA_inv = jp.linalg.inv(R + B.T @ P @ B)
        except jp.linalg.LinAlgError:
            # Add regularization if R + B.T @ P @ B is singular
            print("Warning: R + B.T @ P @ B is singular. Adding regularization.")
            reg = 1e-6 * jp.eye(R.shape[0])
            R_plus_BTPA_inv = jp.linalg.inv(R + B.T @ P @ B + reg)

        P = A.T @ P @ A - A.T @ P @ B @ R_plus_BTPA_inv @ B.T @ P @ A + Q
        if jp.max(jp.abs(P - P_prev)) < tolerance:
            break
    try:
        K_inv_term = jp.linalg.inv(R + B.T @ P @ B)
    except jp.linalg.LinAlgError:
        print("Warning: R + B.T @ P @ B is singular for K. Adding regularization.")
        reg = 1e-6 * jp.eye(R.shape[0])
        K_inv_term = jp.linalg.inv(R + B.T @ P @ B + reg)
    K = K_inv_term @ B.T @ P @ A
    return K, P


def get_AB_matrices(current_env, template_brax_state_arg, q_eq_vec, qd_eq_vec, act_eq_vec, local_jit_env_step):
    """Linearizes the system dynamics dx = Ax + Bu around (q_eq, qd_eq, act_eq)."""
    q_size = current_env.sys.q_size()

    @jax.jit
    def f_dynamics_q_qd(q_vec, qd_vec, action_vec):
        # Create a pipeline_state with the given q_vec, qd_vec
        temp_pipeline_s = template_brax_state_arg.pipeline_state.replace(q=q_vec, qd=qd_vec)
        
        # Create a full Brax state using this pipeline_state and other fields from template_brax_state_arg
        input_state_for_env_step = template_brax_state_arg.replace(pipeline_state=temp_pipeline_s)
        
        # Call env.step
        next_brax_state_from_env_step = local_jit_env_step(input_state_for_env_step, action_vec)
        
        # Extract next q and qd
        return next_brax_state_from_env_step.pipeline_state.q, next_brax_state_from_env_step.pipeline_state.qd

    @jax.jit
    def f_flat_dynamics(flat_state_vec, action_vec):
        q_vec = flat_state_vec[:q_size]
        qd_vec = flat_state_vec[q_size:]
        next_q, next_qd = f_dynamics_q_qd(q_vec, qd_vec, action_vec)
        return jp.concatenate([next_q, next_qd])

    x_eq_flat = jp.concatenate([q_eq_vec, qd_eq_vec])
    
    A = jax.jacobian(f_flat_dynamics, argnums=0)(x_eq_flat, act_eq_vec)
    B = jax.jacobian(f_flat_dynamics, argnums=1)(x_eq_flat, act_eq_vec)
    
    return A, B

# LQR Setup (done once)
q_eq = jp.zeros(env.sys.q_size())  # Equilibrium position [cart_x, pole_angle]
qd_eq = jp.zeros(env.sys.qd_size()) # Equilibrium velocity [cart_v, pole_av]
act_eq = jp.zeros(env.action_size)  # Equilibrium action

print("Linearizing system dynamics...")
A_lqr, B_lqr = get_AB_matrices(env, template_brax_state, q_eq, qd_eq, act_eq, jit_env_step)
print("A matrix:\n", A_lqr)
print("B matrix:\n", B_lqr)

# Define LQR cost matrices Q and R
# State: [cart_pos, pole_angle, cart_vel, pole_ang_vel]
Q_cost = jp.diag(jp.array([1.0, 10.0, 0.1, 0.1]))  # Penalize pole angle more
R_cost = jp.diag(jp.array([0.01]))              # Penalize control effort

print("Solving DARE...")
K_lqr, P_lqr = solve_dare_iterative(A_lqr, B_lqr, Q_cost, R_cost)
print("LQR Gain K:\n", K_lqr)

# Simulation parameters
NUM_SIM_STEPS = 1000

# Define initial conditions to test (batched)
# Each row in q_batch corresponds to a row in qd_batch
initial_conditions_q_batch = jp.array([
    [0.0, 0.05],      # IC 0: Small pos deviation, zero vel
    [0.0, 0.3],       # IC 1: Moderate pos deviation, zero vel
    [0.0, 0.7],       # IC 2: Larger pos deviation, zero vel (might fail)
    [0.2, 0.0],       # IC 3: Cart displaced, pole upright, zero vel
    
    # New conditions with non-zero velocities:
    [0.0, 0.0],       # IC 5: Upright, but pole has initial angular velocity
    [0.0, 0.0],       # IC 6: Upright, cart has initial velocity
    [0.1, 0.05],      # IC 7: Cart and pole slightly displaced, pole has initial vel away from eq
    [0.0, -0.1],      # IC 8: Pole slightly off, initial vel towards equilibrium
])

initial_conditions_qd_batch = jp.array([
    [0.0, 0.0],       # IC 0: Zero vel
    [0.0, 0.0],       # IC 1: Zero vel
    [0.0, 0.0],       # IC 2: Zero vel
    [0.0, 0.0],       # IC 3: Zero vel

    # Corresponding velocities for new conditions:
    [0.0, 0.5],       # IC 5: Pole angular velocity of 0.5 rad/s
    [0.3, 0.0],       # IC 6: Cart velocity of 0.3 m/s
    [0.0, -0.2],      # IC 7: Pole angular velocity of -0.2 rad/s (destabilizing initially)
    [0.0, 0.2],       # IC 8: Pole angular velocity of 0.2 rad/s (stabilizing initially)
])

num_simulations_to_run = initial_conditions_q_batch.shape[0]

# Ensure q and qd batches have the same number of initial conditions
assert initial_conditions_q_batch.shape[0] == initial_conditions_qd_batch.shape[0], \
    "Mismatch in number of initial q and qd conditions!"

# --- JAX scan and vmap for simulation ---
@partial(jax.jit, static_argnames=['local_jit_env_step_fn'])
def lqr_control_step_fn(current_brax_state_arg, k_gain, q_eq_lqr, qd_eq_lqr, local_jit_env_step_fn):
    """Performs one step of LQR control and environment interaction."""
    current_q = current_brax_state_arg.pipeline_state.q
    current_qd = current_brax_state_arg.pipeline_state.qd
    x_lqr_dev = jp.concatenate([current_q - q_eq_lqr, current_qd - qd_eq_lqr])
    action_lqr = -k_gain @ x_lqr_dev
    action_for_env = jp.array(action_lqr) # Ensure correct shape for env.step
    next_brax_state = local_jit_env_step_fn(current_brax_state_arg, action_for_env)
    return next_brax_state, action_for_env

def run_single_simulation_scan(initial_q_single, initial_qd_single, 
                               k_matrix_global, q_eq_global, qd_eq_global, 
                               template_brax_state_global, local_jit_env_step_global, 
                               num_steps_sim):
    """Runs a full simulation for a single initial condition using lax.scan."""
    # Create the true initial brax.State for this specific simulation run
    initial_pipeline_state = template_brax_state_global.pipeline_state.replace(q=initial_q_single, qd=initial_qd_single)
    true_initial_brax_state = template_brax_state_global.replace(pipeline_state=initial_pipeline_state, obs=jp.zeros_like(template_brax_state_global.obs), reward=0.0, done=0.0) # Reset obs, reward, done too

    def scan_body(carry_brax_state, _): # _ is for xs, which is None here
        next_b_state, action = lqr_control_step_fn(carry_brax_state, k_matrix_global, q_eq_global, qd_eq_global, local_jit_env_step_global)
        # We want to collect the state *before* the step for rollout, and the action taken
        output_pipeline_state = carry_brax_state.pipeline_state 
        return next_b_state, (output_pipeline_state, action)

    final_state, (collected_pipeline_states, collected_actions) = lax.scan(
        scan_body, true_initial_brax_state, xs=None, length=num_steps_sim
    )
    return collected_pipeline_states, collected_actions

# Vmap the simulation runner
# All args after initial_q and initial_qd are static (None in in_axes)
vmapped_simulation_runner = jax.vmap(
    run_single_simulation_scan, 
    in_axes=(0, 0, None, None, None, None, None, None)
)

print(f"Running {num_simulations_to_run} LQR simulations in parallel (vmap over scan)...")
batched_pipeline_states, batched_actions = vmapped_simulation_runner(
    initial_conditions_q_batch, 
    initial_conditions_qd_batch, 
    K_lqr, 
    q_eq, 
    qd_eq, 
    template_brax_state, 
    jit_env_step,
    NUM_SIM_STEPS
)
print("All simulations finished.")

# --- Plotting and HTML saving loop ---
for sim_idx in range(num_simulations_to_run):
    print(f"Processing results for simulation {sim_idx+1}/{num_simulations_to_run} (Initial q: {initial_conditions_q_batch[sim_idx]})" )

    # Extract the pipeline states for the current simulation
    current_sim_pipeline_states = jtu.tree_map(lambda x: x[sim_idx], batched_pipeline_states)
    # Extract actions for the current simulation
    current_sim_actions = batched_actions[sim_idx] # This was already correct

    # Extract data for plotting for this specific simulation
    times_plot = jp.arange(NUM_SIM_STEPS) * env_dt_for_plotting
    cart_positions_plot = current_sim_pipeline_states.q[:, 0]
    pole_angles_plot = current_sim_pipeline_states.q[:, 1]
    cart_velocities_plot = current_sim_pipeline_states.qd[:, 0]
    pole_angular_velocities_plot = current_sim_pipeline_states.qd[:, 1]
    # Actions are already (NUM_SIM_STEPS, action_dim). For inverted_pendulum, action_dim is 1.
    # If action_dim > 1, may need to adjust .item() or select component.
    actions_plot_single_sim = current_sim_actions.flatten() # Flatten if action_dim is 1

    # Save HTML rollout
    rollout_filename = f'rollout_lqr_ic_{sim_idx}.html'
    
    # Prepare list of pipeline_states for html.render (one Pytree per timestep)
    rollout_for_html = []
    for t_step in range(NUM_SIM_STEPS):
        ps_at_t = jtu.tree_map(lambda leaf_array_over_time: leaf_array_over_time[t_step], current_sim_pipeline_states)
        rollout_for_html.append(ps_at_t)
        
    with open(rollout_filename, 'w') as f:
        f.write(html.render(env.sys.tree_replace({'opt.timestep': env_dt_for_plotting}), rollout_for_html))
    print(f"LQR simulation rollout saved to {rollout_filename}")

    # Generate and save Matplotlib plot
    plot_filename = f'lqr_state_evolution_ic_{sim_idx}.png'
    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

    axs[0].plot(times_plot, cart_positions_plot, label='Cart Position (m)')
    axs[0].set_ylabel('Position (m)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(times_plot, pole_angles_plot, label='Pole Angle (rad)')
    axs[1].set_ylabel('Angle (rad)')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(times_plot, cart_velocities_plot, label='Cart Velocity (m/s)')
    axs[2].set_ylabel('Velocity (m/s)')
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(times_plot, pole_angular_velocities_plot, label='Pole Angular Velocity (rad/s)')
    axs[3].set_ylabel('Angular Velocity (rad/s)')
    axs[3].legend()
    axs[3].grid(True)

    axs[4].plot(times_plot, actions_plot_single_sim, label='Control Action', color='purple')
    axs[4].set_ylabel('Action')
    axs[4].set_xlabel('Time (s)')
    axs[4].legend()
    axs[4].grid(True)

    fig.suptitle(f'LQR Control IC {sim_idx}: q0={initial_conditions_q_batch[sim_idx]}, qd0={initial_conditions_qd_batch[sim_idx]}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(plot_filename)
    plt.close(fig) # Close the figure to free memory
    print(f"LQR state evolution plot saved to {plot_filename}")

print("All processing complete.")