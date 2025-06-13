import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad
from jax import jit
from functools import partial
from typing import Callable
import os # Added for directory creation
import jax.tree_util as jtu
import brax
from brax import envs
from brax import kinematics
from brax.io import html
from collections import namedtuple

# ---------- System Configuration ----------
SYSTEM_TYPE = "pendulum" # Options: "lds", "pendulum", "brax"
BRAX_ENV_NAME = "inverted_pendulum" # Used if SYSTEM_TYPE == "brax"
MAX_TORQUE = 1.0
M = 1.0
L = 1.0
G = 9.81
DT = 0.02

# ---------- LDS Transition Configuration ----------
LDS_TRANSITION_TYPE = "special" # Options: "linear", "relu", "special"

# ---------- Noise Configuration ----------
LDS_NOISE_TYPE = "gaussian"  # Options: "gaussian", "sinusoidal"
GAUSSIAN_NOISE_STD = 0.5
SINUSOIDAL_NOISE_AMPLITUDE = 0.5
SINUSOIDAL_NOISE_FREQUENCY = 0.1 # Example frequency, adjust as needed

# ---------- Brax Environment Setup ----------
brax_env, brax_template_state, jit_brax_env_step, brax_env_dt = (None, None, None, None)
if SYSTEM_TYPE == "brax":
    brax_env = envs.get_environment(env_name=BRAX_ENV_NAME, backend='spring')
    # Get a template state from a reset, and the JITted step function
    brax_template_state = jax.jit(brax_env.reset)(rng=jax.random.PRNGKey(seed=0))
    jit_brax_env_step = jax.jit(brax_env.step)
    brax_env_dt = brax_env.dt


# Hyperparameter Ranges for Tuning (Grid Search)
# INIT_SCALES_TO_TEST = []
# LEARNING_RATES_TO_TEST = []
# R_M_VALUES_TO_TEST = []
# GAMMA_VALUES_TO_TEST = [] # For SFC and DSC

INIT_SCALES_TO_TEST = [2.0]
LEARNING_RATES_TO_TEST = [5e-5, 1e-4, 5e-4]
R_M_VALUES_TO_TEST = [0.005, 0.01, 0.1, 1.0, 5.0, 10.0]
GAMMA_VALUES_TO_TEST = [0.05, 0.1, 0.3] # For SFC and DSC

# ---------- Experiment & Hyperparameter Configuration ----------
# System Parameters
if SYSTEM_TYPE == "lds":
    NUM_STABLE_TRIALS_THRESHOLD = 2 # 50%
    D, N, P = 25, 1, 1
    T = 5000     # Time horizon for simulations
    GPC_M = 10
    SFC_M_HIST = 30
    SFC_H_FILT = 10
    DSC_M_HIST = 30 # Often same as SFC, but kept separate for flexibility
    DSC_H_FILT = 10 # Often same as SFC
    # Adaptive GPC parameters
    ADAPTIVE_GPC_HH = 10
    ADAPTIVE_GPC_ETA_VALUES = [0.1, 0.5, 1.0]
    ADAPTIVE_GPC_EXPERT_DENSITY_VALUES = [50, 100, 200]
    ADAPTIVE_GPC_LIFE_LOWER_BOUND = 100
elif SYSTEM_TYPE == "pendulum":
    NUM_STABLE_TRIALS_THRESHOLD = 10 # 10%
    INIT_SCALES_TO_TEST = [0.01]
    LEARNING_RATES_TO_TEST = [1e-8, 1e-7, 1e-6, 1e-5]
    D, N, P = 2, 1, 2
    T = 500     # Time horizon for simulations
    GPC_M = 3
    SFC_M_HIST = 5
    SFC_H_FILT = 3
    DSC_M_HIST = 5
    DSC_H_FILT = 3
    GAUSSIAN_NOISE_STD = 0.001
    # Adaptive GPC parameters
    ADAPTIVE_GPC_HH = 3
    ADAPTIVE_GPC_ETA_VALUES = [0.1, 0.5, 1.0]
    ADAPTIVE_GPC_EXPERT_DENSITY_VALUES = [10, 20, 40]
    ADAPTIVE_GPC_LIFE_LOWER_BOUND = 20
elif SYSTEM_TYPE == "brax":
    NUM_STABLE_TRIALS_THRESHOLD = 10 # 10%
    D = brax_env.sys.q_size() + brax_env.sys.qd_size()
    N = brax_env.action_size
    P = D  # Full state as output
    T = 50     # Time horizon for simulations
    # Controller Structure Parameters
    GPC_M = 5
    SFC_M_HIST = 10
    SFC_H_FILT = 5
    DSC_M_HIST = 10 # Often same as SFC, but kept separate for flexibility
    DSC_H_FILT = 5 # Often same as SFC
    GAUSSIAN_NOISE_STD = 0
    # Adaptive GPC parameters
    ADAPTIVE_GPC_HH = 5
    ADAPTIVE_GPC_ETA_VALUES = [0.1, 0.5, 1.0]
    ADAPTIVE_GPC_EXPERT_DENSITY_VALUES = [20, 40, 60]
    ADAPTIVE_GPC_LIFE_LOWER_BOUND = 40
else:
    raise ValueError(f"Unknown SYSTEM_TYPE: {SYSTEM_TYPE}")

LR_DECAY = True # Whether to apply learning rate decay

# Master Key and Trial Setup
SEED = 42
NUM_TRIALS = 50 # Number of common environments for all tuning and comparison
TRIAL_KEYS = jax.random.split(jax.random.PRNGKey(SEED), NUM_TRIALS)



# ---------- Plotting Helper Function ----------
def plot_experiment_results(
    time_steps: np.ndarray,
    mean_avg_costs: np.ndarray,
    stderr_avg_costs: np.ndarray,
    controller_name: str,
    experiment_label: str,
    plot_title: str,
    save_filename: str,
    gamma: float | None = None
):
    """Helper function to plot and save cumulative average cost results."""
    plt.figure(figsize=(8, 5))
    
    label_main = f"Mean {controller_name}"
    label_se = f"{controller_name} ±1 SE"
    if gamma is not None:
        label_main = f"Mean {controller_name} (gamma={gamma})"
        
    plt.plot(time_steps, mean_avg_costs, label=label_main)
    plt.fill_between(
        time_steps,
        mean_avg_costs - stderr_avg_costs,
        mean_avg_costs + stderr_avg_costs,
        alpha=0.3,
        label=label_se
    )
    plt.xlabel("Time Step (T)")
    plt.ylabel("Cumulative Average Cost")
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_filename)
    print(f"{controller_name} {experiment_label} plot saved to {save_filename}")

def get_plot_dir(subdirectory: str | None = None) -> str:
    """Constructs the path to the plots directory based on system configuration."""
    base_path = "plots"
    if SYSTEM_TYPE == "brax":
        path = os.path.join(base_path, SYSTEM_TYPE.lower(), BRAX_ENV_NAME.lower())
    elif SYSTEM_TYPE == "lds":
        path = os.path.join(base_path, SYSTEM_TYPE.lower(), LDS_TRANSITION_TYPE.lower(), LDS_NOISE_TYPE.lower())
    else:
        path = os.path.join(base_path, SYSTEM_TYPE.lower())

    if subdirectory:
        path = os.path.join(path, subdirectory)

    os.makedirs(path, exist_ok=True)
    return path

@jit
def quadratic_cost_evaluate(y, u, Q, R):
    """A JIT-compatible version of quadratic cost evaluation."""
    return jnp.sum(y.T @ Q @ y + u.T @ R @ u)


def gaussian_disturbance(d: int, dist_std: float, t: int, key: jax.random.PRNGKey) -> jax.Array:
    return jax.random.normal(key, (d, 1)) * dist_std

def zero_disturbance(d: int, dist_std: float, t: int, key: jax.random.PRNGKey) -> jax.Array:
    return jnp.zeros((d, 1))

@jit
def lds_sim(x: jnp.ndarray, u: jnp.ndarray, A: jnp.ndarray, B:jnp.ndarray) -> jnp.ndarray:
    return A @ x + B @ u

@jit
def lds_output(x: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
    return C @ x

@jit
def pendulum_sim(x: jnp.ndarray, u: jnp.ndarray, max_torque: float, m: float, l: float, g: float, dt: float) -> jnp.ndarray:
    theta, thdot = x
    action = max_torque * jnp.tanh(u[0])
    newthdot = thdot + (
        -3.0 * g /
        (2.0 * l) * jnp.sin(theta + jnp.pi) + 3.0 /
        (m * l**2) * action) * dt
    newth = theta + newthdot * dt
    return jnp.array([newth, newthdot])

@jit
def pendulum_output(x: jnp.ndarray) -> jnp.ndarray:
    return x
    
@partial(jit, static_argnames=['env', 'jit_step_fn'])
def brax_sim(
    x: jnp.ndarray, 
    u: jnp.ndarray,
    env: brax.envs.Env,
    template_state: brax.base.State,
    jit_step_fn: Callable
) -> jnp.ndarray:
    q_size = env.sys.q_size()

    x_flat = x.flatten()
    q = x_flat[:q_size]
    qd = x_flat[q_size:]

    current_pipeline_state = template_state.pipeline_state.replace(q=q, qd=qd)
    input_brax_state = template_state.replace(pipeline_state=current_pipeline_state)

    action = u.flatten()
    next_brax_state = jit_step_fn(input_brax_state, action)

    # --- 4. Pack the resulting state back into the controller's format ---
    next_q = next_brax_state.pipeline_state.q
    next_qd = next_brax_state.pipeline_state.qd
    next_x_flat = jnp.concatenate([next_q, next_qd])

    # Return as a (d, 1) column vector to match the controller's format
    return next_x_flat[:, jnp.newaxis]

@jit
def brax_output(x: jnp.ndarray) -> jnp.ndarray:
    """Output for brax is the full state."""
    return x

def step(
    x: jnp.ndarray,
    u: jnp.ndarray,
    sim: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    output_map: Callable[[jnp.ndarray], jnp.ndarray],
    dist_std: float,
    t_sim_step: int,
    disturbance: Callable[[int, float, int, jax.random.PRNGKey], jax.Array],
    key: jax.random.PRNGKey
) -> tuple[jnp.ndarray, jnp.ndarray]: # Returns (next_x, y_t)
    d = x.shape[0]
    w = disturbance(d, dist_std, t_sim_step, key)
    x_next = sim(x, u) + w
    y = output_map(x_next)
    return x_next, y

@jit
def lds_gaussian_step_dynamic(
    current_x: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    u_t: jnp.ndarray,
    dist_std: float,
    t_sim_step: int,
    key_disturbance: jax.random.PRNGKey
) -> tuple[jnp.ndarray, jnp.ndarray]: # Returns (next_x, y_t)
    d = A.shape[0]

    w_t = jax.random.normal(key_disturbance, (d, 1)) * dist_std

    x_next = A @ current_x + B @ u_t + w_t

    y_t = C @ current_x

    return x_next, y_t

@jit
def lds_sinusoidal_step_dynamic(
    current_x: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    u_t: jnp.ndarray,
    noise_amplitude: float,
    noise_frequency: float,
    t_sim_step: int,
    key_disturbance: jax.random.PRNGKey
) -> tuple[jnp.ndarray, jnp.ndarray]: # Returns (next_x, y_t)
    """Performs one step of the LDS evolution with sinusoidal noise."""
    d = A.shape[0] # Dimension of the system state (and disturbance)

    # Generate sinusoidal noise for each dimension
    # For simplicity, using the same amplitude and frequency for all dimensions.
    # The phase is driven by t_sim_step.
    # Create a (d, 1) noise vector.
    time_scaled = noise_frequency * t_sim_step.astype(jnp.float32)
    
    # For now, all dimensions will be in phase.
    sin_values = jnp.sin(time_scaled)
    w_t = noise_amplitude * jnp.full((d, 1), sin_values)

    x_next = A @ current_x + B @ u_t + w_t

    y_t = C @ current_x

    return x_next, y_t

@jit
def lds_relu_gaussian_step_dynamic(
    current_x: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    u_t: jnp.ndarray,
    dist_std: float,
    t_sim_step: int,
    key_disturbance: jax.random.PRNGKey
) -> tuple[jnp.ndarray, jnp.ndarray]: # Returns (next_x, y_t)
    """Performs one step of the LDS evolution with ReLU transition and Gaussian noise."""
    d = A.shape[0]

    # Gaussian noise
    w_t = jax.random.normal(key_disturbance, (d, 1)) * dist_std

    # State update with ReLU applied to the deterministic part
    deterministic_next_state = A @ current_x + B @ u_t
    x_next = jax.nn.relu(deterministic_next_state) + w_t

    y_t = C @ current_x

    return x_next, y_t

@jit
def lds_relu_sinusoidal_step_dynamic(
    current_x: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    u_t: jnp.ndarray,
    noise_amplitude: float,
    noise_frequency: float,
    t_sim_step: int,
    key_disturbance: jax.random.PRNGKey
) -> tuple[jnp.ndarray, jnp.ndarray]: # Returns (next_x, y_t)
    """Performs one step of the LDS evolution with ReLU transition and sinusoidal noise."""
    d = A.shape[0]

    # Sinusoidal noise
    time_scaled = noise_frequency * t_sim_step.astype(jnp.float32)
    sin_values = jnp.sin(time_scaled)
    w_t = noise_amplitude * jnp.full((d, 1), sin_values)

    # State update with ReLU applied to the deterministic part
    deterministic_next_state = A @ current_x + B @ u_t
    x_next = jax.nn.relu(deterministic_next_state) + w_t
    
    y_t = C @ current_x

    return x_next, y_t


# ---------- System and Cost Generation ----------
def generate_trial_params(key, d, n, p):
    """Generates random system (A, B, C) and cost (Q, R) matrices, and initial state x0."""
    key_A, key_B, key_C, key_Q_M, key_R_M, _ = jax.random.split(key, 6)
    eigvals = jax.random.uniform(key_A, (d,), minval=0.3, maxval=0.7)
    A = jnp.diag(eigvals)
    B = jax.random.normal(key_B, (d, n))
    C = jax.random.normal(key_C, (p, d))

    if SYSTEM_TYPE == "brax" and BRAX_ENV_NAME == "inverted_pendulum":
        # For inverted_pendulum, the state is [cart_pos, pole_angle, cart_vel, pole_ang_vel].
        # We want to keep the pole upright (angle=0) and cart at origin (pos=0).
        # We penalize pole angle deviation most, then cart position, then velocities.
        Q = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
        # We also penalize the control effort.
        R = jnp.diag(jnp.array([0.01]))
    elif SYSTEM_TYPE == "pendulum":
        # For our custom pendulum, state is [theta, thdot]. We want to keep it upright.
        # Penalize angle deviation and angular velocity.
        Q = jnp.diag(jnp.array([1.0, 0.1]))
        # Also penalize control effort.
        R = jnp.diag(jnp.array([0.001]))
    else:
        # For other systems like LDS, or as a fallback for unhandled brax envs, use random matrices.
        # Note: A random cost matrix is unlikely to be meaningful for a specific brax environment.
        M_Q_gen = jax.random.normal(key_Q_M, (p, p))
        Q = M_Q_gen.T @ M_Q_gen
        M_R_gen = jax.random.normal(key_R_M, (n, n))
        R = M_R_gen.T @ M_R_gen
    
    x0 = jnp.zeros((d,1)) 

    return A, B, C, Q, R, x0

def save_brax_rollout(trajectory, controller_name, trial_idx=0):
    """Generates and saves a brax HTML rollout from a state trajectory."""
    if SYSTEM_TYPE != "brax":
        print(f"Skipping rollout for {controller_name} as SYSTEM_TYPE is not 'brax'.")
        return

    # Select the trajectory for the specified trial index
    # The incoming trajectory is batched: (num_trials, T, D, 1)
    trajectory_to_vis = jtu.tree_map(lambda x: x[trial_idx], trajectory)

    rollout_for_html = []
    q_size = brax_env.sys.q_size()
    num_steps = trajectory_to_vis.shape[0]

    for t_step in range(num_steps):
        x_at_t = trajectory_to_vis[t_step]
        q_at_t = x_at_t[:q_size].flatten()
        qd_at_t = x_at_t[q_size:].flatten()
        # Call kinematics.forward to compute the full kinematic state (x, xd, etc.)
        # from the minimal state (q, qd). This is essential for rendering.
        x, xd = kinematics.forward(brax_env.sys, q_at_t, qd_at_t)
        # Re-create the pipeline state with all fields populated for rendering
        complete_ps_at_t = brax_template_state.pipeline_state.replace(q=q_at_t, qd=qd_at_t, x=x, xd=xd)
        rollout_for_html.append(complete_ps_at_t)

    # Define a directory to save the rollouts
    rollout_dir = get_plot_dir("rollouts")
    rollout_filename = os.path.join(rollout_dir, f"rollout_{controller_name}_brax_trial_{trial_idx}.html")
    
    # Render and save the HTML
    with open(rollout_filename, 'w') as f:
        f.write(html.render(brax_env.sys.tree_replace({'opt.timestep': brax_env_dt}), rollout_for_html))
    print(f"{controller_name.upper()} brax rollout saved to {rollout_filename}")


def plot_state_trajectory(trajectory, u_history, controller_name, dt, trial_to_plot_idx=0, plot_title_suffix=""):
    """Generates and saves a plot of state evolution for a selected trial."""

    # Select the trajectory for the specified trial index and clean it up
    trajectory_to_plot = jtu.tree_map(lambda x: x[trial_to_plot_idx], trajectory)
    trajectory_to_plot = trajectory_to_plot.squeeze(axis=-1)
    
    u_history_to_plot = jtu.tree_map(lambda x: x[trial_to_plot_idx], u_history)
    u_history_to_plot = u_history_to_plot.squeeze(axis=-1)


    num_steps, num_states = trajectory_to_plot.shape
    num_actions = u_history_to_plot.shape[-1] if u_history_to_plot.ndim > 1 else 1

    # Create time axis for plotting
    times_plot = np.arange(num_steps) * dt

    # Create figure with subplots for each state variable + action
    fig, axs = plt.subplots(num_states + num_actions, 1, figsize=(10, 2 * (num_states + num_actions)), sharex=True)
    if num_states + num_actions == 1: # Handle case where there is only one subplot
        axs = [axs]

    # Plot each state variable component
    for i in range(num_states):
        # Specific labels for known systems
        label_text = f'State[{i}]'
        if SYSTEM_TYPE == "brax":
            q_size = brax_env.sys.q_size()
            if i < q_size:
                label_text = f'q[{i}] (Position)'
            else:
                label_text = f'qd[{i-q_size}] (Velocity)'
        elif SYSTEM_TYPE == "pendulum":
            if i == 0:
                label_text = 'theta (Angle)'
            elif i == 1:
                label_text = 'thetadot (Angular Velocity)'

        axs[i].plot(times_plot, trajectory_to_plot[:, i], label=label_text)
        axs[i].set_ylabel('Value')
        axs[i].legend()
        axs[i].grid(True)
    
    # Plot each action component
    for i in range(num_actions):
        axs[num_states + i].plot(times_plot, u_history_to_plot, label=f'u[{i}] (Control Input)')
        axs[num_states + i].set_ylabel('Value')
        axs[num_states + i].legend()
        axs[num_states + i].grid(True)


    axs[-1].set_xlabel('Time (s)')
    full_plot_title = f'State Trajectory for {controller_name.upper()} ({SYSTEM_TYPE.upper()}) {plot_title_suffix}'
    fig.suptitle(full_plot_title.strip(), fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the plot to a dedicated directory
    plot_dir = get_plot_dir("trajectories")
    plot_filename = os.path.join(plot_dir, f"trajectory_{controller_name}_{SYSTEM_TYPE}_trial_{trial_to_plot_idx}.png")
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"{controller_name.upper()} {SYSTEM_TYPE.upper()} trajectory plot saved to {plot_filename}")
    
    
# # --- Brax Simulation Sanity Check with Hardcoded Policy for Inverted Pendulum---
# if SYSTEM_TYPE == "brax":
#     print("\n--- Running Brax Simulation Sanity Check ---")

#     # Define a simple policy to stabilize the inverted pendulum
#     @jit
#     def verification_policy(x, gain=-5.0):
#         # State: [cart_pos, pole_angle, cart_vel, pole_ang_vel]
#         pole_angle = x[1]
#         action = gain * pole_angle
#         # Ensure action has shape (N, 1) for brax_sim
#         return jnp.array([action]).reshape((N, 1))

#     # Get a starting state (e.g., pole slightly tilted)
#     # We use a fixed key for reproducibility of the check
#     _, _, _, _, _, x0_verify = generate_trial_params(jax.random.PRNGKey(0), D, N, P)

#     # Partial function for our specific brax environment
#     sim_fn = partial(brax_sim, env=brax_env, template_state=brax_template_state, jit_step_fn=jit_brax_env_step)

#     # Simulation loop using the verification policy
#     @jit
#     def verification_step(x_t, _):
#         u_t = verification_policy(x_t)
#         x_t_plus_1 = sim_fn(x_t, u_t)
#         return x_t_plus_1, x_t

#     # Run the simulation for T steps
#     _, trajectory_verify = jax.lax.scan(verification_step, x0_verify, xs=None, length=T)

#     # The visualization functions expect a batch dimension (num_trials, ...)
#     # Add a batch dimension to our single trajectory.
#     trajectory_verify_batched = trajectory_verify[jnp.newaxis, ...]

#     print("Verification simulation complete. Generating rollout and trajectory plot...")
#     # Generate visualization of the rollout
#     save_brax_rollout(np.array(trajectory_verify_batched), "verification_policy", trial_idx=0)
#     plot_state_trajectory(np.array(trajectory_verify_batched), "verification_policy", brax_env_dt, trial_idx=0)

#     print("--- Brax Sanity Check Complete ---")
    

def gpc_init_controller_state(key, init_scale, m, n, p, d):
    M = init_scale * jax.random.normal(key, shape=(m, n, p))
    z = jnp.zeros((d, 1))
    ynat_history = jnp.zeros((2 * m, p, 1)) # GPC uses 2*m history
    controller_t = 0
    return M, z, ynat_history, controller_t

def _gpc_action_for_policy_loss(k, M_policy, ynat_history_buffer_policy, m_controller_policy, p_policy):
    window = jax.lax.dynamic_slice(ynat_history_buffer_policy, (k, 0, 0), (m_controller_policy, p_policy, 1))
    contribs = jnp.einsum("mnp,mp1->mn1", M_policy, window)
    return jnp.sum(contribs, axis=0)

def gpc_policy_loss(M, ynat_history_for_loss, cost_evaluate_fn, Q_cost, R_cost, m_controller, d, p, sim, output_map):    
    
    def evolve_for_loss(delta, k):
        u = _gpc_action_for_policy_loss(k, M, ynat_history_for_loss, m_controller, p)
        next_delta = sim(delta, u)
        return next_delta, None

    final_delta, _ = jax.lax.scan(evolve_for_loss, jnp.zeros((d, 1)), jnp.arange(m_controller))
    
    y_final_predicted = output_map(final_delta) + ynat_history_for_loss[-1]
    
    u_final_for_cost = _gpc_action_for_policy_loss(m_controller, M, ynat_history_for_loss, m_controller, p)
    return cost_evaluate_fn(y_final_predicted, u_final_for_cost, Q_cost, R_cost)

def gpc_get_action(M, ynat_history_buffer, m_controller, p):
    window_for_action = jax.lax.dynamic_slice(ynat_history_buffer, (m_controller, 0, 0), (m_controller, p, 1))
    contribs = jnp.einsum('mnp,mp1->mn1', M, window_for_action)
    return jnp.sum(contribs, axis=0)

def gpc_update_controller(
    M, z, ynat_history_buffer, controller_t, # current controller state
    y_meas, u_taken, # inputs from current simulation step
    lr_base, apply_lr_decay, R_M_clip, # controller hyperparameters
    gpc_grad_policy_loss_fn, # pre-defined grad function for policy loss
    m_controller, p, sim, output_map # controller structure params
):
    # Update z state for natural response
    z_next = sim(z, u_taken)
    y_nat_current = y_meas - output_map(z_next)

    # Update ynat_history: roll and add current y_nat at index 0
    ynat_history_updated = jnp.roll(ynat_history_buffer, shift=1, axis=0)
    ynat_history_updated = ynat_history_updated.at[0].set(y_nat_current)

    delta_M = gpc_grad_policy_loss_fn(M, ynat_history_updated)
    
    lr_current = lr_base * (1.0 / (controller_t + 1.0) if apply_lr_decay else 1.0)
    M_next_pre_clip = M - lr_current * delta_M

    # Clip M to ensure its norm is bounded by R_M_clip
    norm_M = jnp.linalg.norm(M_next_pre_clip)
    scale = jnp.minimum(1.0, R_M_clip / (norm_M + 1e-8))
    M_next = scale * M_next_pre_clip

    controller_t_next = controller_t + 1
    return M_next, z_next, ynat_history_updated, controller_t_next
    
    
# ----------Simulation for GPC ----------
def run_single_gpc_trial(
    key_trial_run, A, B, C, Q_cost, R_cost, x0, sim, output_map,
    T_sim_steps, 
    gpc_m, gpc_init_scale, gpc_lr, gpc_lr_decay, gpc_R_M
):
    d, n, p = A.shape[0], B.shape[1], C.shape[0]

    # Define a template for the GPC loss function that accepts the cost_eval_func as an argument.
    def _gpc_loss_for_grad_template(cost_eval_func_arg, M, ynat_hist):
        return gpc_policy_loss(
            M, ynat_hist, 
            cost_eval_func_arg, # Use the passed-in cost evaluation function
            Q_cost, R_cost, # Pass Q_cost and R_cost from the outer scope
            gpc_m, d, p, sim, output_map
        )
    
    # Use functools.partial to bake in the quadratic_cost_evaluate function.
    # The gradient will be taken with respect to M (now arg 0 of final_gpc_loss_fn).
    final_gpc_loss_fn = partial(_gpc_loss_for_grad_template, quadratic_cost_evaluate)
    gpc_grad_fn = jit(grad(final_gpc_loss_fn, argnums=0))

    key_gpc_init, key_sim_loop_main = jax.random.split(key_trial_run)
    M_controller, z_controller, ynat_history_controller, t_controller = gpc_init_controller_state(key_gpc_init, gpc_init_scale, gpc_m, n, p, d)

    current_lds_x_state = x0 # Initial LDS state
    initial_sim_time_step = 0

    def gpc_simulation_step(carry, key_step_specific_randomness):
        prev_lds_x, prev_M, prev_z, prev_ynat_hist, prev_t, prev_sim_t_for_disturbance = carry
        key_disturbance_for_step = key_step_specific_randomness

        u_control_t = gpc_get_action(prev_M, prev_ynat_hist, gpc_m, p)
        
        if LDS_TRANSITION_TYPE == "linear":
            if LDS_NOISE_TYPE == "gaussian":
                next_lds_x, y_measured_t = lds_gaussian_step_dynamic(
                    current_x=prev_lds_x,
                    A=A, B=B, C=C,
                    u_t=u_control_t,
                    dist_std=GAUSSIAN_NOISE_STD,
                    t_sim_step=prev_sim_t_for_disturbance, 
                    key_disturbance=key_disturbance_for_step
                )
            elif LDS_NOISE_TYPE == "sinusoidal":
                next_lds_x, y_measured_t = lds_sinusoidal_step_dynamic(
                    current_x=prev_lds_x,
                    A=A, B=B, C=C,
                    u_t=u_control_t,
                    noise_amplitude=SINUSOIDAL_NOISE_AMPLITUDE, 
                    noise_frequency=SINUSOIDAL_NOISE_FREQUENCY, 
                    t_sim_step=prev_sim_t_for_disturbance, 
                    key_disturbance=key_disturbance_for_step
                )
            else:
                raise ValueError(f"Unknown LDS_NOISE_TYPE: {LDS_NOISE_TYPE} for 'linear' transition")
        elif LDS_TRANSITION_TYPE == "relu":
            if LDS_NOISE_TYPE == "gaussian":
                next_lds_x, y_measured_t = lds_relu_gaussian_step_dynamic(
                    current_x=prev_lds_x,
                    A=A, B=B, C=C,
                    u_t=u_control_t,
                    dist_std=GAUSSIAN_NOISE_STD,
                    t_sim_step=prev_sim_t_for_disturbance, 
                    key_disturbance=key_disturbance_for_step
                )
            elif LDS_NOISE_TYPE == "sinusoidal":
                next_lds_x, y_measured_t = lds_relu_sinusoidal_step_dynamic(
                    current_x=prev_lds_x,
                    A=A, B=B, C=C,
                    u_t=u_control_t,
                    noise_amplitude=SINUSOIDAL_NOISE_AMPLITUDE,
                    noise_frequency=SINUSOIDAL_NOISE_FREQUENCY,
                    t_sim_step=prev_sim_t_for_disturbance, 
                    key_disturbance=key_disturbance_for_step
                )
            else:
                raise ValueError(f"Unknown LDS_NOISE_TYPE: {LDS_NOISE_TYPE} for 'relu' transition")
        elif LDS_TRANSITION_TYPE == "special":
            next_lds_x, y_measured_t = step(
                x=prev_lds_x,
                u=u_control_t,
                    sim=sim,
                output_map=output_map,
                dist_std=GAUSSIAN_NOISE_STD,
                t_sim_step=prev_sim_t_for_disturbance,
                disturbance=gaussian_disturbance,
                key=key_disturbance_for_step
            )
        else:
            raise ValueError(f"Unknown LDS_TRANSITION_TYPE: {LDS_TRANSITION_TYPE}")
        
        next_M, next_z, next_ynat_hist, next_t = gpc_update_controller(
                                                                        prev_M, prev_z, prev_ynat_hist, prev_t,
                                                                        y_measured_t, u_control_t,
                                                                        gpc_lr, gpc_lr_decay, gpc_R_M,
                                                                        gpc_grad_fn,
                                                                        gpc_m, p, sim, output_map
                                                                    )

        cost_at_t = quadratic_cost_evaluate(y_measured_t, u_control_t, Q_cost, R_cost)
        
        next_sim_t_for_disturbance = prev_sim_t_for_disturbance + 1
        next_carry = (next_lds_x, next_M, next_z, next_ynat_hist, next_t, next_sim_t_for_disturbance)
        outputs_to_collect = (cost_at_t, prev_lds_x, u_control_t)
        return next_carry, outputs_to_collect

    initial_carry_gpc = (current_lds_x_state, M_controller, z_controller, ynat_history_controller, t_controller, initial_sim_time_step)
    keys_for_T_scan_steps = jax.random.split(key_sim_loop_main, T_sim_steps)
    
    (_, (costs_over_time, trajectory, u_history)) = jax.lax.scan(gpc_simulation_step, initial_carry_gpc, keys_for_T_scan_steps)
    
    return costs_over_time, trajectory, u_history

@partial(jit, static_argnames=("d", "n", "p", "T", "gpc_m", "gpc_init_scale", "gpc_lr", "gpc_decay", "gpc_R_M"))
def gpc_trial_runner_for_vmap(key_trial_specific,d,n,p,T,gpc_m,gpc_init_scale,gpc_lr,gpc_decay,gpc_R_M):
    key_params_gen, key_sim_run = jax.random.split(key_trial_specific)
    A,B,C,Q,R,x0 = generate_trial_params(key_params_gen,d,n,p) 
    if SYSTEM_TYPE == "lds":
        sim = partial(lds_sim, A=A, B=B)
        output_map = partial(lds_output, C=C)
    elif SYSTEM_TYPE == "pendulum":
        sim = partial(pendulum_sim, max_torque=MAX_TORQUE, m=M, l=L, g=G, dt=DT)
        output_map = pendulum_output
    elif SYSTEM_TYPE == "brax":
        sim = partial(brax_sim, env=brax_env, template_state=brax_template_state, jit_step_fn=jit_brax_env_step)
        output_map = brax_output
    else:
        raise ValueError(f"Unknown SYSTEM_TYPE: {SYSTEM_TYPE}")
    
    costs, trajectory, u_history = run_single_gpc_trial(key_sim_run,A,B,C,Q,R,x0,sim,output_map,T,gpc_m,gpc_init_scale,gpc_lr,gpc_decay,gpc_R_M)
    return costs, trajectory, u_history


# --- GPC Hyperparameter Tuning (Grid Search) ---
print("\n--- GPC Hyperparameter Tuning (Grid Search) ---")

# Define common hyperparameter ranges for tuning
common_init_scales = [2.0]
common_learning_rates = [5e-5, 1e-4, 5e-4]
common_R_M_values = [0.005, 0.01, 0.1, 1.0, 5.0, 10.0]
common_gamma_values = [0.05, 0.1, 0.3] # For SFC and DSC


print(f"Tuning GPC with fixed params: d={D}, n={N}, p={P}, m={GPC_M}, T={T}, trials={NUM_TRIALS}, LR decay={LR_DECAY}")


plt.figure(figsize=(15, 10)) # Larger figure for potentially many lines
plt.title(f"GPC Hyperparameter Tuning (d={D}, m={GPC_M}, Transition={LDS_TRANSITION_TYPE}, Noise={LDS_NOISE_TYPE})")
plt.xlabel("Time Step")
plt.ylabel("Mean Cumulative Average Cost")
plt.grid(True)

best_gpc_hyperparams = None
lowest_final_cost = float('inf')

color_idx = 0
num_param_combinations = len(INIT_SCALES_TO_TEST) * len(R_M_VALUES_TO_TEST) * len(LEARNING_RATES_TO_TEST)
colors = plt.cm.viridis(np.linspace(0, 1, num_param_combinations if num_param_combinations > 0 else 1))

# gpc_trial_runner_for_vmap expects: key, d, n, p, T, gpc_m, init_scale, lr, decay, R_M (10 args)
vmap_gpc = jax.vmap(gpc_trial_runner_for_vmap, 
                               in_axes=(0,None,None,None,None,None,None,None,None,None), # 10 entries
                               out_axes=0)

for init_scale in INIT_SCALES_TO_TEST:
    for R_M in R_M_VALUES_TO_TEST:
        for lr in LEARNING_RATES_TO_TEST:
            current_params = {"init_scale": init_scale, "R_M": R_M, "lr": lr}
            print(f"Testing GPC with: init_scale={init_scale}, R_M={R_M}, lr={lr}...")
        
            costs, _, u_history = vmap_gpc(
                TRIAL_KEYS,
                D,
                N,
                P,
                T,
                GPC_M, 
                init_scale,
                lr,
                LR_DECAY,
                R_M
            )
            costs_current_params = np.array(costs)

            cum_costs_current = np.cumsum(costs_current_params, axis=1)

            if np.any(np.isnan(cum_costs_current)) or np.any(np.isinf(cum_costs_current)):
                print(f"  Params {current_params} resulted in NaN/Inf costs. Skipping.")
                color_idx += 1
                continue

            stability_threshold_tune = 1e7 
            stable_mask_current = cum_costs_current[:, -1] < stability_threshold_tune
            num_stable_trials = np.sum(stable_mask_current)

            print(f"  Params {current_params}: {num_stable_trials}/{NUM_TRIALS} stable trials.")

            if num_stable_trials > NUM_TRIALS // NUM_STABLE_TRIALS_THRESHOLD: 
                avg_costs_stable = cum_costs_current[stable_mask_current] / (np.arange(1, T + 1)[None, :])
                mean_avg_cost_current = np.mean(avg_costs_stable, axis=0)
                
                label_str = f"is={init_scale}, rm={R_M}, lr={lr:.0e}"
                # Ensure color_idx is within bounds for colors array
                current_color = colors[color_idx % len(colors)] if len(colors) > 0 else 'blue'
                plt.plot(mean_avg_cost_current, label=label_str, color=current_color, alpha=0.7)
                
                final_avg_cost = mean_avg_cost_current[-1]
                if final_avg_cost < lowest_final_cost:
                    lowest_final_cost = final_avg_cost
                    best_gpc_hyperparams = current_params
                    print(f"    New best params: {best_gpc_hyperparams} with final avg cost: {final_avg_cost:.4f}")
            else:
                print(f"  Params {current_params}: Not enough stable trials. Skipping plot.")
            color_idx += 1

if plt.gca().has_data() and len(plt.gca().lines) > 20 : 
    print("Many lines on the plot, omitting legend for clarity. See console for parameters.")
elif plt.gca().has_data():
    plt.legend(loc='upper right', fontsize='small')
    
plt.ylim(bottom=0, top= (lowest_final_cost * 2 if lowest_final_cost != float('inf') and lowest_final_cost > 0 else (np.max(plt.ylim()) if plt.gca().has_data() else 100) ) ) 

base_filename_gpc = "gpc_tuning_full_grid_search_plot.png"
plots_dir_gpc = get_plot_dir()
full_save_path_gpc = os.path.join(plots_dir_gpc, base_filename_gpc)
plt.savefig(full_save_path_gpc)
print(f"GPC grid search tuning plot saved to {full_save_path_gpc}")

if best_gpc_hyperparams:
    print(f"\nSelected best GPC hyperparameters from grid search: {best_gpc_hyperparams}")
    print(f"Corresponding lowest final mean cumulative average cost: {lowest_final_cost:.4f}")
else:
    print("\nCould not determine a best set of GPC hyperparameters from the tested values. All may have diverged or were unstable.")

print("\n--- GPC Hyperparameter Tuning (Grid Search) Complete ---")


# ---------- Adaptive GPC Controller ----------
from collections import namedtuple

# Pytree for storing the state of all GPC experts
GPCExpertStates = namedtuple('GPCExpertStates', ['M', 'z', 'ynat_history', 't'])

# Pytree for the full state of the Adaptive GPC controller
AdaptiveGPCState = namedtuple('AdaptiveGPCState', ['t', 'weights', 'alive', 'experts'])

def lifetime(x, lower_bound):
    """Computes the lifetime of an expert based on its birth time."""
    l = 4
    # This loop is a bit unusual but matches the reference implementation.
    # It will not JIT compile with a `while` loop on a traced value.
    # However, we call this with a static `i` from a Python loop, so it's fine.
    while x > 0 and x % 2 == 0:
        l *= 2
        x /= 2
    return max(lower_bound, l + 1)

def adaptive_gpc_init_controller_state(key, T_sim, d, n, p, gpc_m, init_scale, hh):
    """Initializes the state of the Adaptive GPC controller."""
    # Expert 0 is born at t=0 and is alive.
    key_expert0, _ = jax.random.split(key)
    M0, z0, ynat_hist0, t0 = gpc_init_controller_state(key_expert0, init_scale, gpc_m, n, p, d)

    # All other experts are dormant.
    experts_M = jnp.zeros((T_sim, gpc_m, n, p)).at[0].set(M0)
    experts_z = jnp.zeros((T_sim, d, 1)).at[0].set(z0)
    experts_ynat_hist = jnp.zeros((T_sim, 2 * gpc_m, p, 1)).at[0].set(ynat_hist0)
    experts_t = jnp.zeros((T_sim,), dtype=jnp.int32).at[0].set(t0)

    expert_states = GPCExpertStates(M=experts_M, z=experts_z, ynat_history=experts_ynat_hist, t=experts_t)

    # Only expert 0 is alive and has weight
    weights = jnp.zeros((T_sim,)).at[0].set(1.0)
    alive = jnp.zeros((T_sim,), dtype=jnp.int32).at[0].set(1)

    return AdaptiveGPCState(t=0, weights=weights, alive=alive, experts=expert_states)


def run_single_adaptive_gpc_trial(
    key_trial_run, A, B, C, Q_cost, R_cost, x0, sim, output_map,
    T_sim_steps,
    gpc_m, gpc_init_scale, gpc_lr, gpc_lr_decay, gpc_R_M, # GPC params
    hh, eta, eps, inf, life_lower_bound, expert_density # Adaptive params
):
    d, n, p = A.shape[0], B.shape[1], C.shape[0]

    # Precompute Time of Death for all potential experts
    tod = np.arange(T_sim_steps)
    for i in range(1, T_sim_steps):
        tod[i] = i + lifetime(i, life_lower_bound)
    tod[0] = life_lower_bound
    tod = jnp.array(tod)

    # --- JIT-compiled GPC loss function (re-defined for local scope) ---
    def _gpc_loss_for_grad_template(cost_eval_func_arg, M, ynat_hist):
        # This is the same GPC loss function used before
        return gpc_policy_loss(M, ynat_hist, cost_eval_func_arg, Q_cost, R_cost, gpc_m, d, p, sim, output_map)
    final_gpc_loss_fn = partial(_gpc_loss_for_grad_template, quadratic_cost_evaluate)
    gpc_grad_fn = jit(grad(final_gpc_loss_fn, argnums=0))

    # --- Vmapped functions for updating all experts in parallel ---
    @jit
    def vmapped_gpc_update(experts, y_meas, u_taken):
        def update_one_expert(M, z, ynat_hist, t):
            # We need to compute the grad inside, or pass it in.
            # Passing it in is better.
            return gpc_update_controller(M, z, ynat_hist, t, y_meas, u_taken, gpc_lr, gpc_lr_decay, gpc_R_M, gpc_grad_fn, gpc_m, p, sim, output_map)
        return jax.vmap(update_one_expert)(experts.M, experts.z, experts.ynat_history, experts.t)

    @jit
    def vmapped_gpc_loss(experts):
        return jax.vmap(final_gpc_loss_fn)(experts.M, experts.ynat_history)

    key_adaptive_init, key_sim_loop_main = jax.random.split(key_trial_run)
    adaptive_controller_state = adaptive_gpc_init_controller_state(key_adaptive_init, T_sim_steps, d, n, p, gpc_m, gpc_init_scale, hh)

    current_x_state = x0
    initial_sim_time_step = 0

    def adaptive_gpc_simulation_step(carry, key_step):
        prev_x, prev_adaptive_state, prev_sim_t = carry
        
        # 1. Select best expert and get action
        play_i = jnp.argmax(prev_adaptive_state.weights)
        
        # Slice out the state of the best expert
        played_M = jtu.tree_map(lambda leaf: leaf[play_i], prev_adaptive_state.experts.M)
        played_ynat_hist = jtu.tree_map(lambda leaf: leaf[play_i], prev_adaptive_state.experts.ynat_history)
        
        u_t = gpc_get_action(played_M, played_ynat_hist, gpc_m, p)

        # 2. Step the true system dynamics
        # (This section is identical to other controllers)
        if LDS_TRANSITION_TYPE == "special":
             next_x, y_measured_t = step(
                x=prev_x, u=u_t, sim=sim, output_map=output_map,
                dist_std=GAUSSIAN_NOISE_STD, t_sim_step=prev_sim_t,
                disturbance=gaussian_disturbance, key=key_step
            )
        # Note: Add other LDS types here if needed, following GPC/SFC structure
        else:
             raise ValueError(f"Unsupported LDS transition type for AdaptiveGPC: {LDS_TRANSITION_TYPE}")


        # 3. Update all GPC experts with the single (y_measured, u_taken) pair
        next_M, next_z, next_ynat_hist, next_t_experts = vmapped_gpc_update(prev_adaptive_state.experts, y_measured_t, u_t)
        updated_expert_states = GPCExpertStates(M=next_M, z=next_z, ynat_history=next_ynat_hist, t=next_t_experts)
        
        # 4. Evaluate loss for all experts to update adaptive weights
        losses = vmapped_gpc_loss(updated_expert_states)
        
        # Update weights based on loss, only for alive experts
        new_weights = prev_adaptive_state.weights * jnp.exp(-eta * losses)
        new_weights = jnp.where(prev_adaptive_state.alive, new_weights, prev_adaptive_state.weights)
        new_weights = jnp.clip(new_weights, a_min=eps, a_max=inf) # Clip weights

        # 5. Birth and Death of experts
        t_current = prev_adaptive_state.t
        t_next = t_current + 1
        
        # Death
        should_die_mask = (tod == t_next)
        next_alive = jnp.where(should_die_mask, 0, prev_adaptive_state.alive)
        new_weights = jnp.where(should_die_mask, 0, new_weights)
        
        # Birth
        is_birth_time = (t_next % expert_density == 0)
        
        # Initialize new expert state if it's birth time
        key_birth, _ = jax.random.split(key_step)
        new_expert_M, new_expert_z, new_expert_ynat, new_expert_t = gpc_init_controller_state(
            key_birth, gpc_init_scale, gpc_m, n, p, d
        )

        # Place the new expert into the Pytree of states
        final_expert_M = jax.lax.cond(
            is_birth_time,
            lambda x: x.at[t_next].set(new_expert_M),
            lambda x: x,
            updated_expert_states.M
        )
        final_expert_z = jax.lax.cond(is_birth_time, lambda x: x.at[t_next].set(new_expert_z), lambda x: x, updated_expert_states.z)
        final_expert_ynat = jax.lax.cond(is_birth_time, lambda x: x.at[t_next].set(new_expert_ynat), lambda x: x, updated_expert_states.ynat_history)
        final_expert_t = jax.lax.cond(is_birth_time, lambda x: x.at[t_next].set(new_expert_t), lambda x: x, updated_expert_states.t)
        
        # Update alive status and weights for the newborn
        next_alive = jax.lax.cond(is_birth_time, lambda x: x.at[t_next].set(1), lambda x: x, next_alive)
        new_weights = jax.lax.cond(is_birth_time, lambda x: x.at[t_next].set(eps), lambda x: x, new_weights)
        
        final_expert_states = GPCExpertStates(M=final_expert_M, z=final_expert_z, ynat_history=final_expert_ynat, t=final_expert_t)

        # 6. Rescale weights
        max_w = jnp.max(new_weights)
        final_weights = jax.lax.cond(max_w < 1, lambda w: w / max_w, lambda w: w, new_weights)

        # 7. Assemble next state
        next_adaptive_state = AdaptiveGPCState(t=t_next, weights=final_weights, alive=next_alive, experts=final_expert_states)
        
        # Calculate cost for this step
        cost_at_t = quadratic_cost_evaluate(y_measured_t, u_t, Q_cost, R_cost)
        
        next_carry = (next_x, next_adaptive_state, prev_sim_t + 1)
        outputs_to_collect = (cost_at_t, prev_x, u_t)
        return next_carry, outputs_to_collect

    initial_carry = (current_x_state, adaptive_controller_state, initial_sim_time_step)
    keys_for_scan = jax.random.split(key_sim_loop_main, T_sim_steps)

    (_, (costs_over_time, trajectory, u_history)) = jax.lax.scan(adaptive_gpc_simulation_step, initial_carry, keys_for_scan)
    return costs_over_time, trajectory, u_history

@partial(jit, static_argnames=("d","n","p","T","gpc_m","gpc_init_scale","gpc_lr","gpc_decay","gpc_R_M", "hh", "eta", "eps", "inf", "life_lower_bound", "expert_density"))
def adaptive_gpc_trial_runner_for_vmap(key_trial_specific,d,n,p,T,gpc_m,gpc_init_scale,gpc_lr,gpc_decay,gpc_R_M, hh, eta, eps, inf, life_lower_bound, expert_density):
    key_params_gen, key_sim_run = jax.random.split(key_trial_specific)
    A,B,C,Q,R,x0 = generate_trial_params(key_params_gen,d,n,p)
    if SYSTEM_TYPE == "lds":
        sim = partial(lds_sim, A=A, B=B)
        output_map = partial(lds_output, C=C)
    elif SYSTEM_TYPE == "pendulum":
        sim = partial(pendulum_sim, max_torque=MAX_TORQUE, m=M, l=L, g=G, dt=DT)
        output_map = pendulum_output
    elif SYSTEM_TYPE == "brax":
        sim = partial(brax_sim, env=brax_env, template_state=brax_template_state, jit_step_fn=jit_brax_env_step)
        output_map = brax_output
    else:
        raise ValueError(f"Unknown SYSTEM_TYPE: {SYSTEM_TYPE}")
    
    costs, trajectory, u_history = run_single_adaptive_gpc_trial(
        key_sim_run,A,B,C,Q,R,x0,sim,output_map,T,
        gpc_m,gpc_init_scale,gpc_lr,gpc_decay,gpc_R_M,
        hh, eta, eps, inf, life_lower_bound, expert_density
    )
    return costs, trajectory, u_history

# --- Adaptive GPC Hyperparameter Tuning (Grid Search) ---
print("\n--- Adaptive GPC Hyperparameter Tuning (Grid Search) ---")

if best_gpc_hyperparams:
    print(f"Using best GPC params for base controller: {best_gpc_hyperparams}")
    agpc_base_init_scale = best_gpc_hyperparams['init_scale']
    agpc_base_lr = best_gpc_hyperparams['lr']
    agpc_base_R_M = best_gpc_hyperparams['R_M']
else:
    print("Warning: No best GPC params found. Using defaults for Adaptive GPC base.")
    agpc_base_init_scale = 0.1
    agpc_base_lr = 1e-4
    agpc_base_R_M = 10.0

print(f"Tuning Adaptive GPC with fixed params: d={D}, n={N}, p={P}, m={GPC_M}, T={T}, trials={NUM_TRIALS}")

plt.figure(figsize=(15, 10))
plt.title(f"Adaptive GPC Hyperparameter Tuning (d={D}, m={GPC_M}, System={SYSTEM_TYPE})")
plt.xlabel("Time Step")
plt.ylabel("Mean Cumulative Average Cost")
plt.grid(True)

best_adaptive_gpc_hyperparams = None
lowest_final_adaptive_gpc_cost = float('inf')

agpc_color_idx = 0
num_agpc_param_combinations = len(ADAPTIVE_GPC_ETA_VALUES) * len(ADAPTIVE_GPC_EXPERT_DENSITY_VALUES)
agpc_colors = plt.cm.magma(np.linspace(0, 1, num_agpc_param_combinations if num_agpc_param_combinations > 0 else 1))

vmap_agpc = jax.vmap(adaptive_gpc_trial_runner_for_vmap, 
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, 0, None, None, None, 0),
    out_axes=0)

for eta in ADAPTIVE_GPC_ETA_VALUES:
    for expert_density in ADAPTIVE_GPC_EXPERT_DENSITY_VALUES:
        current_params = {"eta": eta, "expert_density": expert_density}
        print(f"Testing Adaptive GPC with: {current_params}...")
        
        costs, _, u_history = vmap_agpc(
            TRIAL_KEYS, D, N, P, T, GPC_M, 
            agpc_base_init_scale, agpc_base_lr, LR_DECAY, agpc_base_R_M,
            ADAPTIVE_GPC_HH, eta, 1e-6, 1e6, ADAPTIVE_GPC_LIFE_LOWER_BOUND, expert_density
        )
        costs_current_params = np.array(costs)

        cum_costs_current = np.cumsum(costs_current_params, axis=1)

        if np.any(np.isnan(cum_costs_current)) or np.any(np.isinf(cum_costs_current)):
            print(f"  Params {current_params} resulted in NaN/Inf costs. Skipping.")
            agpc_color_idx += 1
            continue

        stable_mask = cum_costs_current[:, -1] < 1e7
        num_stable_trials = np.sum(stable_mask)

        print(f"  Params {current_params}: {num_stable_trials}/{NUM_TRIALS} stable trials.")

        if num_stable_trials > NUM_TRIALS // NUM_STABLE_TRIALS_THRESHOLD:
            avg_costs_stable = cum_costs_current[stable_mask] / (np.arange(1, T + 1)[None, :])
            mean_avg_cost_current = np.mean(avg_costs_stable, axis=0)
            
            label_str = f"eta={eta}, dens={expert_density}"
            current_color = agpc_colors[agpc_color_idx % len(agpc_colors)]
            plt.plot(mean_avg_cost_current, label=label_str, color=current_color, alpha=0.7)
            
            final_avg_cost = mean_avg_cost_current[-1]
            if final_avg_cost < lowest_final_adaptive_gpc_cost:
                lowest_final_adaptive_gpc_cost = final_avg_cost
                best_adaptive_gpc_hyperparams = current_params
                print(f"    New best params: {best_adaptive_gpc_hyperparams} with final avg cost: {final_avg_cost:.4f}")
        else:
            print(f"  Params {current_params}: Not enough stable trials. Skipping plot.")
        agpc_color_idx += 1

if plt.gca().has_data():
    plt.legend(loc='upper right', fontsize='small')
plt.ylim(bottom=0)
plots_dir_agpc = get_plot_dir()
plt.savefig(os.path.join(plots_dir_agpc, "adaptive_gpc_tuning_plot.png"))
print(f"Adaptive GPC tuning plot saved to {os.path.join(plots_dir_agpc, 'adaptive_gpc_tuning_plot.png')}")

if best_adaptive_gpc_hyperparams:
    print(f"\nSelected best Adaptive GPC hyperparameters: {best_adaptive_gpc_hyperparams}")
else:
    print("\nCould not determine best Adaptive GPC hyperparameters.")

print("\n--- Adaptive GPC Hyperparameter Tuning Complete ---")


# ---------- SFC Controller ----------
def sfc_compute_filters(m_hist, h_filt, gamma):
    indices = np.arange(1, m_hist + 1)
    I, J = np.meshgrid(indices, indices, indexing='ij')
    H_matrix = ((1 - gamma) ** (I + J - 1)) / (I + J - 1)
    eigvals, eigvecs = np.linalg.eigh(H_matrix)
    top_eigvals = eigvals[-h_filt:]
    top_eigvecs = eigvecs[:, -h_filt:]
    scaling = top_eigvals**0.25
    return jnp.array(np.flip((scaling[:, None] * top_eigvecs.T), axis=1))

def sfc_init_controller_state(key, init_scale, h_filt, n, p, d, m_hist):
    key_M0, key_M = jax.random.split(key)
    M0 = init_scale * jax.random.normal(key_M0, shape=(n, p))
    M = init_scale * jax.random.normal(key_M, shape=(h_filt, n, p))
    z = jnp.zeros((d, 1))
    ynat_history = jnp.zeros((2 * m_hist + 1, p, 1))
    controller_t = 0
    return M0, M, z, ynat_history, controller_t

def _sfc_action_for_policy_loss(k, M0_policy, M_policy, ynat_hist_policy, filters_policy, h_filt_policy, m_hist_policy, p_policy):
    window = jax.lax.dynamic_slice(ynat_hist_policy, (k, 0, 0), (m_hist_policy, p_policy, 1))
    last_ynat_for_action = jnp.squeeze(jax.lax.dynamic_slice(ynat_hist_policy, (k + m_hist_policy, 0, 0), (1, p_policy, 1)), axis=0)
    proj = jnp.einsum("hm,mp1->hp1", filters_policy, window)
    spectral_term = jnp.einsum("hnp,hp1->n1", M_policy, proj)
    return M0_policy @ last_ynat_for_action + spectral_term

def sfc_policy_loss(M0, M, ynat_history_for_loss, cost_evaluate_fn, Q_cost, R_cost, filters, h_filt, m_hist, d, p, sim, output_map):
    def evolve_for_loss(delta_state, m_idx_in_scan):
        u_for_evolve = _sfc_action_for_policy_loss(m_idx_in_scan, M0, M, ynat_history_for_loss, filters, h_filt, m_hist, p)
        next_delta_state = sim(delta_state, u_for_evolve)
        return next_delta_state, None
    initial_delta_state = jnp.zeros((d, 1))
    final_delta, _ = jax.lax.scan(evolve_for_loss, initial_delta_state, jnp.arange(m_hist))
    y_final_predicted = output_map(final_delta) + ynat_history_for_loss[-1]
    u_final_for_cost = _sfc_action_for_policy_loss(m_hist, M0, M, ynat_history_for_loss, filters, h_filt, m_hist, p)
    return cost_evaluate_fn(y_final_predicted, u_final_for_cost, Q_cost, R_cost)

def sfc_get_action(M0, M, ynat_history_buffer, filters, h_filt, m_hist, p):
    window_for_action = jax.lax.dynamic_slice(ynat_history_buffer, (m_hist, 0, 0), (m_hist, p, 1))
    last_ynat_for_action = jnp.squeeze(jax.lax.dynamic_slice(ynat_history_buffer, (2 * m_hist, 0, 0), (1, p, 1)), axis=0)
    proj = jnp.einsum("hm,mp1->hp1", filters, window_for_action)
    spectral_term = jnp.einsum("hnp,hp1->n1", M, proj)
    return M0 @ last_ynat_for_action + spectral_term

def sfc_update_controller(
    M0, M, z, ynat_history_buffer, controller_t, 
    y_meas, u_taken, 
    lr_base, apply_lr_decay, R_M_clip, 
    sfc_grad_policy_loss_fn, 
    h_filt, m_hist, p, sim, output_map
):
    z_next = sim(z, u_taken)
    y_nat_current = y_meas - output_map(z_next)
    ynat_history_updated = jnp.roll(ynat_history_buffer, shift=1, axis=0)
    ynat_history_updated = ynat_history_updated.at[0].set(y_nat_current)
    delta_M0, delta_M = sfc_grad_policy_loss_fn(M0, M, ynat_history_updated)
    lr_current = lr_base * (1.0 / (controller_t + 1.0) if apply_lr_decay else 1.0)
    M0_next_pre_clip = M0 - lr_current * delta_M0
    M_next_pre_clip = M - lr_current * delta_M
    norm_params = jnp.sqrt(jnp.sum(M0_next_pre_clip**2) + jnp.sum(M_next_pre_clip**2))
    scale = jnp.minimum(1.0, R_M_clip / (norm_params + 1e-8)) # Add epsilon for stability
    M0_next = scale * M0_next_pre_clip
    M_next = scale * M_next_pre_clip
    controller_t_next = controller_t + 1
    return M0_next, M_next, z_next, ynat_history_updated, controller_t_next


def run_single_sfc_trial(
    key_trial_run, A, B, C, Q_cost, R_cost, x0, sim, output_map,
    T_sim_steps, 
    sfc_m_hist, sfc_h_filt, sfc_init_scale, sfc_lr, sfc_lr_decay, sfc_R_M, 
    sfc_filters
):
    d, n, p = A.shape[0], B.shape[1], C.shape[0]

    def _sfc_loss_for_grad_template(filters_arg, M0, M, ynat_hist):
        return sfc_policy_loss(
            M0, M, ynat_hist, 
            quadratic_cost_evaluate, Q_cost, R_cost, 
            filters_arg, 
            sfc_h_filt, sfc_m_hist, d, p, sim, output_map
        )
    final_sfc_loss_fn = partial(_sfc_loss_for_grad_template, sfc_filters)
    sfc_grad_fn = jit(grad(final_sfc_loss_fn, argnums=(0, 1)))
    
    key_sfc_init, key_sim_loop_main = jax.random.split(key_trial_run)
    M0, M, z, ynat_hist, t = sfc_init_controller_state(
        key_sfc_init, sfc_init_scale, sfc_h_filt, n, p, d, sfc_m_hist
    )
    current_lds_x_state = x0
    initial_sim_time_step = 0

    def sfc_simulation_step(carry, key_step_specific_randomness):
        prev_lds_x, prev_M0, prev_M, prev_z, prev_ynat_hist, prev_t, prev_sim_t_for_disturbance = carry
        key_disturbance_for_step = key_step_specific_randomness

        u_control_t = sfc_get_action(
            prev_M0, prev_M, prev_ynat_hist, 
            sfc_filters, 
            sfc_h_filt, sfc_m_hist, p
        )

        # Determine which LDS step function to use based on global configuration
        if SYSTEM_TYPE == "lds" and LDS_TRANSITION_TYPE != "special":
            if LDS_TRANSITION_TYPE == "linear":
                if LDS_NOISE_TYPE == "gaussian":
                    next_lds_x, y_measured_t = lds_gaussian_step_dynamic(
                        current_x=prev_lds_x,
                        A=A, B=B, C=C,
                        u_t=u_control_t,
                        dist_std=GAUSSIAN_NOISE_STD, 
                        t_sim_step=prev_sim_t_for_disturbance,
                        key_disturbance=key_disturbance_for_step
                    )
                elif LDS_NOISE_TYPE == "sinusoidal":
                    next_lds_x, y_measured_t = lds_sinusoidal_step_dynamic(
                        current_x=prev_lds_x,
                        A=A, B=B, C=C,
                        u_t=u_control_t,
                        noise_amplitude=SINUSOIDAL_NOISE_AMPLITUDE, 
                        noise_frequency=SINUSOIDAL_NOISE_FREQUENCY, 
                        t_sim_step=prev_sim_t_for_disturbance,
                        key_disturbance=key_disturbance_for_step
                    )
                else: 
                    raise ValueError(f"Unknown LDS_NOISE_TYPE: {LDS_NOISE_TYPE} for 'linear' transition")
            elif LDS_TRANSITION_TYPE == "relu":
                if LDS_NOISE_TYPE == "gaussian":
                    next_lds_x, y_measured_t = lds_relu_gaussian_step_dynamic(
                        current_x=prev_lds_x,
                        A=A, B=B, C=C,
                        u_t=u_control_t,
                        dist_std=GAUSSIAN_NOISE_STD, 
                        t_sim_step=prev_sim_t_for_disturbance,
                        key_disturbance=key_disturbance_for_step
                    )
                elif LDS_NOISE_TYPE == "sinusoidal":
                    next_lds_x, y_measured_t = lds_relu_sinusoidal_step_dynamic(
                        current_x=prev_lds_x,
                        A=A, B=B, C=C,
                        u_t=u_control_t,
                        noise_amplitude=SINUSOIDAL_NOISE_AMPLITUDE,
                        noise_frequency=SINUSOIDAL_NOISE_FREQUENCY,
                        t_sim_step=prev_sim_t_for_disturbance,
                        key_disturbance=key_disturbance_for_step
                    )
                else:
                    raise ValueError(f"Unknown LDS_NOISE_TYPE: {LDS_NOISE_TYPE} for 'relu' transition")
        else:
            next_lds_x, y_measured_t = step(
                x=prev_lds_x,
                u=u_control_t,
                sim=sim,
                output_map=output_map,
                dist_std=GAUSSIAN_NOISE_STD,
                t_sim_step=prev_sim_t_for_disturbance,
                disturbance=gaussian_disturbance,
                key=key_disturbance_for_step
            )
        
        next_M0, next_M, next_z, next_ynat_hist, next_t = sfc_update_controller(
            prev_M0, prev_M, prev_z, prev_ynat_hist, prev_t, 
            y_measured_t, u_control_t, 
            sfc_lr, sfc_lr_decay, sfc_R_M, 
            sfc_grad_fn, 
            sfc_h_filt, sfc_m_hist, p, sim, output_map
        )
            
        cost_at_t = quadratic_cost_evaluate(y_measured_t, u_control_t, Q_cost, R_cost)
        
        next_sim_t_for_disturbance = prev_sim_t_for_disturbance + 1 
        next_carry = (next_lds_x, next_M0, next_M, next_z, next_ynat_hist, next_t, next_sim_t_for_disturbance)
        outputs = (cost_at_t, prev_lds_x, u_control_t)
        return next_carry, outputs
    
    initial_carry_sfc = (current_lds_x_state, M0, M, z, ynat_hist, t, initial_sim_time_step)
    keys_for_T_scan_steps = jax.random.split(key_sim_loop_main, T_sim_steps)
    (_, (costs_over_time, trajectory, u_history)) = jax.lax.scan(sfc_simulation_step, initial_carry_sfc, keys_for_T_scan_steps)
    return costs_over_time, trajectory, u_history

@partial(jit, static_argnames=("d","n","p","T","sfc_m_hist","sfc_h_filt","sfc_init_sc","sfc_lr","sfc_decay","sfc_R_M"))
def sfc_trial_runner_for_vmap(key_trial_specific,d,n,p,T,sfc_m_hist,sfc_h_filt,sfc_init_sc,sfc_lr,sfc_decay,sfc_R_M,sfc_filters):
    key_params_gen, key_sim_run = jax.random.split(key_trial_specific)
    A,B,C,Q,R,x0 = generate_trial_params(key_params_gen,d,n,p)
    if SYSTEM_TYPE == "lds":
        sim = partial(lds_sim, A=A, B=B)
        output_map = partial(lds_output, C=C)
    elif SYSTEM_TYPE == "pendulum":
        sim = partial(pendulum_sim, max_torque=MAX_TORQUE, m=M, l=L, g=G, dt=DT)
        output_map = pendulum_output
    elif SYSTEM_TYPE == "brax":
        sim = partial(brax_sim, env=brax_env, template_state=brax_template_state, jit_step_fn=jit_brax_env_step)
        output_map = brax_output
    else:
        raise ValueError(f"Unknown SYSTEM_TYPE: {SYSTEM_TYPE}")
    
    costs, trajectory, u_history = run_single_sfc_trial(key_sim_run,A,B,C,Q,R,x0,sim,output_map,T,sfc_m_hist,sfc_h_filt,sfc_init_sc,sfc_lr,sfc_decay,sfc_R_M,sfc_filters)
    return costs, trajectory, u_history


# --- SFC Hyperparameter Tuning (Grid Search) ---
print("\n--- SFC Hyperparameter Tuning (Grid Search) ---")


print(f"Tuning SFC with fixed params: d={D}, n={N}, p={P}, m_hist={SFC_M_HIST}, h_filt={SFC_H_FILT}, T={T}, trials={NUM_TRIALS}, LR decay={LR_DECAY}")

plt.figure(figsize=(15, 10))
plt.title(f"SFC Hyperparameter Tuning (d={D}, m_hist={SFC_M_HIST}, h_filt={SFC_H_FILT}, Transition={LDS_TRANSITION_TYPE}, Noise={LDS_NOISE_TYPE})")
plt.xlabel("Time Step")
plt.ylabel("Mean Cumulative Average Cost")
plt.grid(True)

best_sfc_hyperparams = None
lowest_sfc_final_cost = float('inf')

sfc_color_idx = 0
sfc_num_param_combinations = len(INIT_SCALES_TO_TEST) * len(R_M_VALUES_TO_TEST) * len(LEARNING_RATES_TO_TEST) * len(GAMMA_VALUES_TO_TEST)
sfc_colors = plt.cm.cool(np.linspace(0, 1, sfc_num_param_combinations if sfc_num_param_combinations > 0 else 1))

# sfc_trial_runner_for_vmap expects: key, d,n,p,T,sfc_m_hist,sfc_h_filt,sfc_init_sc,sfc_lr,sfc_decay,sfc_R_M,sfc_filters (12 args)
vmapped_sfc_runner = jax.vmap(
    sfc_trial_runner_for_vmap, 
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None), # Should be 12 entries
    out_axes=0
)

for sfc_init_scale in INIT_SCALES_TO_TEST:
    for sfc_R_M in R_M_VALUES_TO_TEST:
        for sfc_lr in LEARNING_RATES_TO_TEST:
            for sfc_gamma in GAMMA_VALUES_TO_TEST:
                current_sfc_params = {
                    "init_scale": sfc_init_scale, 
                    "R_M": sfc_R_M, 
                    "lr": sfc_lr,
                    "gamma": sfc_gamma
                }
                print(f"Testing SFC with: {current_sfc_params}...")
                
                current_sfc_filters = sfc_compute_filters(SFC_M_HIST, SFC_H_FILT, sfc_gamma)

                costs, _, u_history = vmapped_sfc_runner(
                    TRIAL_KEYS, # Use common master keys
                    D, N, P, 
                    T,
                    SFC_M_HIST, SFC_H_FILT, 
                    sfc_init_scale, sfc_lr, LR_DECAY, sfc_R_M,
                    current_sfc_filters
                )
                costs_current_sfc_params = np.array(costs)

                cum_costs_current_sfc = np.cumsum(costs_current_sfc_params, axis=1)

                if np.any(np.isnan(cum_costs_current_sfc)) or np.any(np.isinf(cum_costs_current_sfc)):
                    print(f"  SFC Params {current_sfc_params} resulted in NaN/Inf costs. Skipping.")
                    sfc_color_idx += 1
                    continue

                stability_threshold_sfc_tune = 1e7 
                stable_mask_current_sfc = cum_costs_current_sfc[:, -1] < stability_threshold_sfc_tune
                num_stable_sfc_trials = np.sum(stable_mask_current_sfc)

                print(f"  SFC Params {current_sfc_params}: {num_stable_sfc_trials}/{NUM_TRIALS} stable trials.")

                if num_stable_sfc_trials > NUM_TRIALS // NUM_STABLE_TRIALS_THRESHOLD: 
                    avg_costs_stable_sfc = cum_costs_current_sfc[stable_mask_current_sfc] / (np.arange(1, T + 1)[None, :])
                    mean_avg_cost_current_sfc = np.mean(avg_costs_stable_sfc, axis=0)
                    
                    sfc_label_str = f"is={sfc_init_scale}, rm={sfc_R_M}, lr={sfc_lr:.0e}, g={sfc_gamma}"
                    current_sfc_color = sfc_colors[sfc_color_idx % len(sfc_colors)] if len(sfc_colors) > 0 else 'green'
                    plt.plot(mean_avg_cost_current_sfc, label=sfc_label_str, color=current_sfc_color, alpha=0.6)
                    
                    final_avg_sfc_cost = mean_avg_cost_current_sfc[-1]
                    if final_avg_sfc_cost < lowest_sfc_final_cost:
                        lowest_sfc_final_cost = final_avg_sfc_cost
                        best_sfc_hyperparams = current_sfc_params
                        print(f"    New best SFC params: {best_sfc_hyperparams} with final avg cost: {lowest_sfc_final_cost:.4f}")
                else:
                    print(f"  SFC Params {current_sfc_params}: Not enough stable trials. Skipping plot.")
                sfc_color_idx += 1

if plt.gca().has_data() and len(plt.gca().lines) > 20: 
    print("Many lines on the SFC tuning plot, omitting legend for clarity. See console for parameters.")
elif plt.gca().has_data():
    plt.legend(loc='upper right', fontsize='x-small')
    
plt.ylim(bottom=0, top= (lowest_sfc_final_cost * 5 if lowest_sfc_final_cost != float('inf') and lowest_sfc_final_cost > 0 else (np.max(plt.ylim()) if plt.gca().has_data() else 100) ) )

base_filename_sfc = "sfc_tuning_full_grid_search_plot.png"
plots_dir_sfc = get_plot_dir()
full_save_path_sfc = os.path.join(plots_dir_sfc, base_filename_sfc)
plt.savefig(full_save_path_sfc)
print(f"SFC grid search tuning plot saved to {full_save_path_sfc}")

if best_sfc_hyperparams:
    print(f"\nSelected best SFC hyperparameters from grid search: {best_sfc_hyperparams}")
    print(f"Corresponding lowest final mean cumulative average cost for SFC: {lowest_sfc_final_cost:.4f}")
else:
    print("\nCould not determine a best set of SFC hyperparameters from the tested values.")

print("\n--- SFC Hyperparameter Tuning (Grid Search) Complete ---")


# ---------- DSC Controller Functions ----------
def dsc_init_controller_state(key, init_scale, h_filt, m_hist, n, p, d):
    key_M0, key_M1, key_M2, key_M3 = jax.random.split(key, 4)
    M0 = init_scale * jax.random.normal(key_M0, shape=(n, p))
    M1 = init_scale * jax.random.normal(key_M1, shape=(h_filt, n, p))
    M2 = init_scale * jax.random.normal(key_M2, shape=(h_filt, n, p))
    M3 = init_scale * jax.random.normal(key_M3, shape=(h_filt, h_filt, n, p)) # M3 is (h,h,n,p)
    z = jnp.zeros((d, 1))
    ynat_history = jnp.zeros((3 * m_hist + 1, p, 1)) # DSC uses 3*m+1 history, newest at index 0
    controller_t = 0
    return M0, M1, M2, M3, z, ynat_history, controller_t

def _dsc_get_last_ynat_matrix_static(ynat_history_buffer, base_k_offset, m_hist, p):
    def extract_window(i_vmap_offset): # i_vmap_offset from 0 to m_hist-1
        start_idx_for_dynamic_slice = base_k_offset + 1 + i_vmap_offset
        return jax.lax.dynamic_slice(ynat_history_buffer, (start_idx_for_dynamic_slice, 0, 0), (m_hist, p, 1))
    return jax.vmap(extract_window)(jnp.arange(m_hist))


def _dsc_action_terms_static(M0, M1, M2, M3, ynat_hist_buffer, filters_static, h_filt, m_hist, p, k_action_offset):
    idx_current_last_ynat = k_action_offset + 2 * m_hist 
    current_last_ynat = jnp.squeeze(jax.lax.dynamic_slice(ynat_hist_buffer, (idx_current_last_ynat, 0, 0), (1, p, 1)), axis=0)
    term_M0 = M0 @ current_last_ynat

    idx_prev_m_start = k_action_offset + m_hist
    prev_m_ynats = jax.lax.dynamic_slice(ynat_hist_buffer, (idx_prev_m_start, 0, 0), (m_hist, p, 1))
    proj1 = jnp.einsum("hm,mp1->hp1", filters_static, prev_m_ynats)
    term_M1 = jnp.einsum("hnp,hp1->n1", M1, proj1)

    idx_curr_m_start = k_action_offset + m_hist + 1
    current_last_m_ynats = jax.lax.dynamic_slice(ynat_hist_buffer, (idx_curr_m_start, 0, 0), (m_hist, p, 1))
    proj2 = jnp.einsum("hm,mp1->hp1", filters_static, current_last_m_ynats) 
    term_M2 = jnp.einsum("hnp,hp1->n1", M2, proj2)
    
    current_last_ynat_matrix = _dsc_get_last_ynat_matrix_static(ynat_hist_buffer, k_action_offset, m_hist, p)
    temp = jnp.einsum("hi,imp1->hmp1", filters_static, current_last_ynat_matrix)
    filtered = jnp.einsum("jm,hmp1->hjp1", filters_static, temp)
    term_M3_contrib = jnp.einsum("hjnp,hjp1->hjn1", M3, filtered)
    term_M3 = jnp.sum(term_M3_contrib, axis=(0, 1))

    return term_M0 + term_M1 + term_M2 + term_M3

def dsc_policy_loss(M0, M1, M2, M3, ynat_history_for_loss, cost_evaluate_fn, Q_cost, R_cost, filters, h_filt, m_hist, d, p, sim, output_map):
    # ynat_history_for_loss: newest is at index 0. Oldest relevant for prediction base is ynat_history_for_loss[2*m_hist + m_hist] = ynat_history_for_loss[3*m_hist]
    def evolve_for_loss(delta_state, m_idx_in_scan): 
        u_for_evolve = _dsc_action_terms_static(M0, M1, M2, M3, ynat_history_for_loss, filters, h_filt, m_hist, p, k_action_offset=m_idx_in_scan)
        next_delta_state = sim(delta_state, u_for_evolve)
        return next_delta_state, None

    initial_delta_state = jnp.zeros((d, 1))
    final_delta, _ = jax.lax.scan(evolve_for_loss, initial_delta_state, jnp.arange(m_hist)) # Evolve m steps
    
    y_final_predicted = output_map(final_delta) + ynat_history_for_loss[-1]
    u_final_for_cost = _dsc_action_terms_static(M0, M1, M2, M3, ynat_history_for_loss, filters, h_filt, m_hist, p, k_action_offset=m_hist)
    return cost_evaluate_fn(y_final_predicted, u_final_for_cost, Q_cost, R_cost)

def dsc_update_controller(
    M0, M1, M2, M3, z, ynat_history_buffer, controller_t,
    y_meas, u_taken,
    lr_base, apply_lr_decay, R_M_clip,
    dsc_grad_policy_loss_fn,
    h_filt, m_hist, p, sim, output_map
):
    z_next = sim(z, u_taken)
    y_nat_current = y_meas - output_map(z_next)
    ynat_history_updated = jnp.roll(ynat_history_buffer, shift=1, axis=0) # newest at [0]
    ynat_history_updated = ynat_history_updated.at[0].set(y_nat_current)
    
    delta_M0, delta_M1, delta_M2, delta_M3 = dsc_grad_policy_loss_fn(M0, M1, M2, M3, ynat_history_updated)
    
    lr_current = lr_base * (1.0 / (controller_t + 1.0) if apply_lr_decay else 1.0)
    M0_n = M0 - lr_current * delta_M0
    M1_n = M1 - lr_current * delta_M1
    M2_n = M2 - lr_current * delta_M2
    M3_n = M3 - lr_current * delta_M3

    norm_params = jnp.sqrt(jnp.sum(M0_n**2) + jnp.sum(M1_n**2) + jnp.sum(M2_n**2) + jnp.sum(M3_n**2))
    scale = jnp.minimum(1.0, R_M_clip / (norm_params + 1e-8)) # Add epsilon for stability
    M0_next = scale * M0_n
    M1_next = scale * M1_n
    M2_next = scale * M2_n
    M3_next = scale * M3_n
    
    controller_t_next = controller_t + 1
    return M0_next, M1_next, M2_next, M3_next, z_next, ynat_history_updated, controller_t_next

# ---------- DSC Simulation ----------
def run_single_dsc_trial(
    key_trial_run, A, B, C, Q_cost, R_cost, x0, sim, output_map,
    T_sim_steps,
    dsc_m_hist, dsc_h_filt, dsc_init_scale, 
    dsc_lr, dsc_lr_decay, dsc_R_M,
    dsc_filters # Precomputed filters
):
    d, n, p = A.shape[0], B.shape[1], C.shape[0]

    def _dsc_loss_for_grad_template(filters_arg, M0_c, M1_c, M2_c, M3_c, ynat_hist_c):
        return dsc_policy_loss(
            M0_c, M1_c, M2_c, M3_c, ynat_hist_c,
            quadratic_cost_evaluate, Q_cost, R_cost,
            filters_arg, dsc_h_filt, dsc_m_hist, d, p, sim, output_map
        )
    
    final_dsc_loss_fn = partial(_dsc_loss_for_grad_template, dsc_filters)
    dsc_grad_fn = jit(grad(final_dsc_loss_fn, argnums=(0, 1, 2, 3)))

    key_dsc_init, key_sim_loop_main = jax.random.split(key_trial_run)
    M0, M1, M2, M3, z, ynat_hist, t = dsc_init_controller_state(key_dsc_init, dsc_init_scale, dsc_h_filt, dsc_m_hist, n, p, d)

    current_lds_x_state = x0
    initial_sim_time_step = 0

    def dsc_simulation_step(carry, key_step_specific_randomness):
        prev_lds_x, prev_M0, prev_M1, prev_M2, prev_M3, prev_z, prev_ynat_hist, prev_t, prev_sim_t_for_disturbance = carry
        key_disturbance_for_step = key_step_specific_randomness

        u_control_t = _dsc_action_terms_static(
            prev_M0, prev_M1, prev_M2, prev_M3, prev_ynat_hist,
            dsc_filters, dsc_h_filt, dsc_m_hist, p,
            k_action_offset=0
        )
        
        # Determine which LDS step function to use based on global configuration
        if LDS_TRANSITION_TYPE == "linear":
            if LDS_NOISE_TYPE == "gaussian":
                next_lds_x, y_measured_t = lds_gaussian_step_dynamic(
                    current_x=prev_lds_x,
                    A=A, B=B, C=C,
                    u_t=u_control_t,
                    dist_std=GAUSSIAN_NOISE_STD, 
                    t_sim_step=prev_sim_t_for_disturbance,
                    key_disturbance=key_disturbance_for_step
                )
            elif LDS_NOISE_TYPE == "sinusoidal":
                next_lds_x, y_measured_t = lds_sinusoidal_step_dynamic(
                    current_x=prev_lds_x,
                    A=A, B=B, C=C,
                    u_t=u_control_t,
                    noise_amplitude=SINUSOIDAL_NOISE_AMPLITUDE, 
                    noise_frequency=SINUSOIDAL_NOISE_FREQUENCY, 
                    t_sim_step=prev_sim_t_for_disturbance,
                    key_disturbance=key_disturbance_for_step
                )
            else: 
                raise ValueError(f"Unknown LDS_NOISE_TYPE: {LDS_NOISE_TYPE} for 'linear' transition")
        elif LDS_TRANSITION_TYPE == "relu":
            if LDS_NOISE_TYPE == "gaussian":
                next_lds_x, y_measured_t = lds_relu_gaussian_step_dynamic(
                    current_x=prev_lds_x,
                    A=A, B=B, C=C,
                    u_t=u_control_t,
                    dist_std=GAUSSIAN_NOISE_STD, 
                    t_sim_step=prev_sim_t_for_disturbance,
                    key_disturbance=key_disturbance_for_step
                )
            elif LDS_NOISE_TYPE == "sinusoidal":
                next_lds_x, y_measured_t = lds_relu_sinusoidal_step_dynamic(
                    current_x=prev_lds_x,
                    A=A, B=B, C=C,
                    u_t=u_control_t,
                    noise_amplitude=SINUSOIDAL_NOISE_AMPLITUDE,
                    noise_frequency=SINUSOIDAL_NOISE_FREQUENCY,
                    t_sim_step=prev_sim_t_for_disturbance,
                    key_disturbance=key_disturbance_for_step
                )
            else:
                raise ValueError(f"Unknown LDS_NOISE_TYPE: {LDS_NOISE_TYPE} for 'relu' transition")
        elif LDS_TRANSITION_TYPE == "special":
            next_lds_x, y_measured_t = step(
                x=prev_lds_x,
                u=u_control_t,
                sim=sim,
                output_map=output_map,
                dist_std=GAUSSIAN_NOISE_STD,
                t_sim_step=prev_sim_t_for_disturbance,
                disturbance=gaussian_disturbance,
                key=key_disturbance_for_step
            )
        else:
            raise ValueError(f"Unknown LDS_TRANSITION_TYPE: {LDS_TRANSITION_TYPE}")

        next_M0, next_M1, next_M2, next_M3, next_z, next_ynat_hist, next_t = dsc_update_controller(
                                                                                prev_M0, prev_M1, prev_M2, prev_M3, prev_z, prev_ynat_hist, prev_t,
                                                                                y_measured_t, u_control_t,
                                                                                dsc_lr, dsc_lr_decay, dsc_R_M,
                                                                                dsc_grad_fn,
                                                                                dsc_h_filt, dsc_m_hist, p, sim, output_map
                                                                            )
            
        cost_at_t = quadratic_cost_evaluate(y_measured_t, u_control_t, Q_cost, R_cost)
        
        next_sim_t_for_disturbance = prev_sim_t_for_disturbance + 1
        next_carry = (next_lds_x, next_M0, next_M1, next_M2, next_M3, next_z, next_ynat_hist, next_t, next_sim_t_for_disturbance)
        outputs = (cost_at_t, prev_lds_x, u_control_t)
        return next_carry, outputs

    initial_carry_dsc = (current_lds_x_state, M0, M1, M2, M3, z, ynat_hist, t, initial_sim_time_step)
    keys_for_T_scan_steps = jax.random.split(key_sim_loop_main, T_sim_steps)
    
    (_, (costs_over_time, trajectory, u_history)) = jax.lax.scan(dsc_simulation_step, initial_carry_dsc, keys_for_T_scan_steps)
    return costs_over_time, trajectory, u_history

@partial(jit, static_argnames=(
    "d","n","p","T",
    "dsc_m_hist","dsc_h_filt","dsc_init_sc", 
    "dsc_lr","dsc_decay","dsc_R_M"
))
def dsc_trial_runner_for_vmap(
    key_trial_specific,
    d,n,p,T, 
    dsc_m_hist,dsc_h_filt,dsc_init_sc,
    dsc_lr,dsc_decay,dsc_R_M,
    dsc_filters # broadcasted by vmap
):
    key_params_gen, key_sim_run = jax.random.split(key_trial_specific)
    A,B,C,Q,R,x0 = generate_trial_params(key_params_gen,d,n,p)
    if SYSTEM_TYPE == "lds":
        sim = partial(lds_sim, A=A, B=B)
        output_map = partial(lds_output, C=C)
    elif SYSTEM_TYPE == "pendulum":
        sim = partial(pendulum_sim, max_torque=MAX_TORQUE, m=M, l=L, g=G, dt=DT)
        output_map = pendulum_output
    elif SYSTEM_TYPE == "brax":
        sim = partial(brax_sim, env=brax_env, template_state=brax_template_state, jit_step_fn=jit_brax_env_step)
        output_map = brax_output
    else:
        raise ValueError(f"Unknown SYSTEM_TYPE: {SYSTEM_TYPE}")

    costs, trajectory, u_history = run_single_dsc_trial(
        key_sim_run,A,B,C,Q,R,x0,sim,output_map,T,
        dsc_m_hist,dsc_h_filt,dsc_init_sc,dsc_lr,dsc_decay,dsc_R_M,
        dsc_filters
    )
    return costs, trajectory, u_history


# --- DSC Hyperparameter Tuning (Grid Search) ---
print("\n--- DSC Hyperparameter Tuning (Grid Search) ---")
print(f"Tuning DSC with fixed params: d={D}, n={N}, p={P}, m_hist={DSC_M_HIST}, h_filt={DSC_H_FILT}, T={T}, trials={NUM_TRIALS}, LR decay={LR_DECAY}")

plt.figure(figsize=(15, 10))
plt.title(f"DSC Hyperparameter Tuning (d={D}, m_hist={DSC_M_HIST}, h_filt={DSC_H_FILT}, Transition={LDS_TRANSITION_TYPE}, Noise={LDS_NOISE_TYPE})")
plt.xlabel("Time Step")
plt.ylabel("Mean Cumulative Average Cost")
plt.grid(True)

best_dsc_hyperparams = None
lowest_dsc_final_cost = float('inf')

dsc_color_idx = 0
dsc_num_param_combinations = len(INIT_SCALES_TO_TEST) * len(R_M_VALUES_TO_TEST) * len(LEARNING_RATES_TO_TEST) * len(GAMMA_VALUES_TO_TEST)
dsc_colors = plt.cm.plasma(np.linspace(0, 1, dsc_num_param_combinations if dsc_num_param_combinations > 0 else 1))

# dsc_trial_runner_for_vmap expects: key,d,n,p,T,dsc_m_hist,dsc_h_filt,dsc_init_sc,dsc_lr,dsc_decay,dsc_R_M,dsc_filters (12 args)
vmapped_dsc_runner = jax.vmap(
    dsc_trial_runner_for_vmap, 
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None), # 12 entries
    out_axes=0
)

for dsc_init_scale in INIT_SCALES_TO_TEST:
    for dsc_R_M in R_M_VALUES_TO_TEST:
        for dsc_lr in LEARNING_RATES_TO_TEST:
            for dsc_gamma in GAMMA_VALUES_TO_TEST:
                current_dsc_params = {
                    "init_scale": dsc_init_scale, 
                    "R_M": dsc_R_M, 
                    "lr": dsc_lr,
                    "gamma": dsc_gamma
                }
                print(f"Testing DSC with: {current_dsc_params}...")
                
                current_dsc_filters = sfc_compute_filters(DSC_M_HIST, DSC_H_FILT, dsc_gamma)

                costs, _, u_history = vmapped_dsc_runner(
                    TRIAL_KEYS, # Use common master keys
                    D, N, P, 
                    T,
                    DSC_M_HIST, DSC_H_FILT, 
                    dsc_init_scale, dsc_lr, LR_DECAY, dsc_R_M,
                    current_dsc_filters
                )
                costs_current_dsc_params = np.array(costs)

                cum_costs_current_dsc = np.cumsum(costs_current_dsc_params, axis=1)

                if np.any(np.isnan(cum_costs_current_dsc)) or np.any(np.isinf(cum_costs_current_dsc)):
                    print(f"  DSC Params {current_dsc_params} resulted in NaN/Inf costs. Skipping.")
                    dsc_color_idx += 1
                    continue

                stability_threshold_dsc_tune = 1e7 
                stable_mask_current_dsc = cum_costs_current_dsc[:, -1] < stability_threshold_dsc_tune
                num_stable_dsc_trials = np.sum(stable_mask_current_dsc)

                print(f"  DSC Params {current_dsc_params}: {num_stable_dsc_trials}/{NUM_TRIALS} stable trials.")

                if num_stable_dsc_trials > NUM_TRIALS // NUM_STABLE_TRIALS_THRESHOLD: 
                    avg_costs_stable_dsc = cum_costs_current_dsc[stable_mask_current_dsc] / (np.arange(1, T + 1)[None, :])
                    mean_avg_cost_current_dsc = np.mean(avg_costs_stable_dsc, axis=0)
                    
                    dsc_label_str = f"is={dsc_init_scale}, rm={dsc_R_M:.1e}, lr={dsc_lr:.0e}, g={dsc_gamma}"
                    current_dsc_color = dsc_colors[dsc_color_idx % len(dsc_colors)] if len(dsc_colors) > 0 else 'red'
                    plt.plot(mean_avg_cost_current_dsc, label=dsc_label_str, color=current_dsc_color, alpha=0.6)
                    
                    final_avg_dsc_cost = mean_avg_cost_current_dsc[-1]
                    if final_avg_dsc_cost < lowest_dsc_final_cost:
                        lowest_dsc_final_cost = final_avg_dsc_cost
                        best_dsc_hyperparams = current_dsc_params
                        print(f"    New best DSC params: {best_dsc_hyperparams} with final avg cost: {lowest_dsc_final_cost:.4f}")
                else:
                    print(f"  DSC Params {current_dsc_params}: Not enough stable trials. Skipping plot.")
                dsc_color_idx += 1

if plt.gca().has_data() and len(plt.gca().lines) > 20: 
    print("Many lines on the DSC tuning plot, omitting legend for clarity. See console for parameters.")
elif plt.gca().has_data():
    plt.legend(loc='upper right', fontsize='x-small')
    
plt.ylim(bottom=0, top= (lowest_dsc_final_cost * 5 if lowest_dsc_final_cost != float('inf') and lowest_dsc_final_cost > 0 else (np.max(plt.ylim()) if plt.gca().has_data() else 100) ) )

base_filename_dsc = "dsc_tuning_full_grid_search_plot.png"
plots_dir_dsc = get_plot_dir()
full_save_path_dsc = os.path.join(plots_dir_dsc, base_filename_dsc)
plt.savefig(full_save_path_dsc)
print(f"DSC grid search tuning plot saved to {full_save_path_dsc}")

if best_dsc_hyperparams:
    print(f"\nSelected best DSC hyperparameters from grid search: {best_dsc_hyperparams}")
    print(f"Corresponding lowest final mean cumulative average cost for DSC: {lowest_dsc_final_cost:.4f}")
else:
    print("\nCould not determine a best set of DSC hyperparameters from the tested values.")

print("\n--- DSC Hyperparameter Tuning (Grid Search) Complete ---")


# Store best hyperparameters for comparison plots, with fallbacks
# GPC
tuned_gpc_init_scale_comp = best_gpc_hyperparams['init_scale'] if best_gpc_hyperparams else 0.1 
tuned_gpc_lr_comp = best_gpc_hyperparams['lr'] if best_gpc_hyperparams else 1e-4       
tuned_gpc_R_M_comp = best_gpc_hyperparams['R_M'] if best_gpc_hyperparams else 10.0     
gpc_params_source_message = "using tuned params" if best_gpc_hyperparams else "using default GPC params (tuning failed)"

# SFC
default_sfc_init_scale = 0.1
default_sfc_lr = 1e-4
default_sfc_R_M = 10.0
default_sfc_gamma = 0.2

tuned_sfc_init_scale_comp = best_sfc_hyperparams['init_scale'] if best_sfc_hyperparams else default_sfc_init_scale
tuned_sfc_lr_comp = best_sfc_hyperparams['lr'] if best_sfc_hyperparams else default_sfc_lr
tuned_sfc_R_M_comp = best_sfc_hyperparams['R_M'] if best_sfc_hyperparams else default_sfc_R_M
tuned_sfc_gamma_comp = best_sfc_hyperparams['gamma'] if best_sfc_hyperparams else default_sfc_gamma
sfc_params_source_message = "using tuned params" if best_sfc_hyperparams else "using default SFC params (tuning failed)"

# DSC
default_dsc_init_scale = 0.1
default_dsc_lr = 1e-4
default_dsc_R_M = 0.001
default_dsc_gamma = 0.2

tuned_dsc_init_scale_comp = best_dsc_hyperparams['init_scale'] if best_dsc_hyperparams else default_dsc_init_scale
tuned_dsc_lr_comp = best_dsc_hyperparams['lr'] if best_dsc_hyperparams else default_dsc_lr
tuned_dsc_R_M_comp = best_dsc_hyperparams['R_M'] if best_dsc_hyperparams else default_dsc_R_M
tuned_dsc_gamma_comp = best_dsc_hyperparams['gamma'] if best_dsc_hyperparams else default_dsc_gamma
dsc_params_source_message = "using tuned params" if best_dsc_hyperparams else "using default DSC params (tuning failed)"

# Adaptive GPC
tuned_agpc_eta_comp = best_adaptive_gpc_hyperparams['eta'] if best_adaptive_gpc_hyperparams else 0.5
tuned_agpc_exp_dens_comp = best_adaptive_gpc_hyperparams['expert_density'] if best_adaptive_gpc_hyperparams else 50
agpc_params_source_message = "using tuned params" if best_adaptive_gpc_hyperparams else "using default Adaptive GPC params (tuning failed)"


# --- Comparison Plots---
print("\n--- Running Comparison Experiments (Optimized GPC, SFC, DSC) ---")


comp_num_trials = NUM_TRIALS 
comp_threshold = 1e7 

comp_trial_keys = TRIAL_KEYS # Use the master keys for comparison as well

# GPC for Comparison 
gpc_init_scale_for_comp = tuned_gpc_init_scale_comp
gpc_lr_for_comp = tuned_gpc_lr_comp
gpc_R_M_for_comp = tuned_gpc_R_M_comp
gpc_hyperparam_details_str = ""
if best_gpc_hyperparams:
    gpc_hyperparam_details_str = f": is={best_gpc_hyperparams['init_scale']:.3f}, lr={best_gpc_hyperparams['lr']:.1e}, R_M={best_gpc_hyperparams['R_M']:.3f}"
gpc_label_suffix = f" ({gpc_params_source_message}{gpc_hyperparam_details_str})"

print(f"GPC for Comparison ({gpc_params_source_message}): {comp_num_trials} trials, {T} steps, m={GPC_M}, init_scale={gpc_init_scale_for_comp:.3f}, lr={gpc_lr_for_comp:.1e}, R_M={gpc_R_M_for_comp:.3f}")
comp_costs_gpc, comp_trajectories_gpc, gpc_u_history = vmap_gpc(
    comp_trial_keys, 
    D, N, P, 
    T, 
    GPC_M,
    gpc_init_scale_for_comp, 
    gpc_lr_for_comp, 
    LR_DECAY,
    gpc_R_M_for_comp
)
comp_costs_gpc_np = np.array(comp_costs_gpc)
print("GPC for Comparison complete.")
# Determine dt for plotting and save visualizations
dt_for_plot = brax_env_dt if SYSTEM_TYPE == "brax" else DT
cum_gpc_comp = np.cumsum(comp_costs_gpc_np, axis=1)
mask_gpc_comp = cum_gpc_comp[:,-1] < comp_threshold

# Logic to select the best stable trial for plotting
plot_idx_gpc = 0
plot_title_gpc = f"(Trial {plot_idx_gpc} - All Unstable)"
if np.any(mask_gpc_comp):
    final_costs_gpc = cum_gpc_comp[:, -1]
    costs_for_min_finding = np.where(mask_gpc_comp, final_costs_gpc, np.inf)
    best_stable_idx = np.argmin(costs_for_min_finding)
    plot_idx_gpc = best_stable_idx
    plot_title_gpc = f"(Best Stable Trial #{plot_idx_gpc})"

save_brax_rollout(np.array(comp_trajectories_gpc), "gpc", trial_idx=plot_idx_gpc)
plot_state_trajectory(np.array(comp_trajectories_gpc), np.array(gpc_u_history), "gpc", dt_for_plot, trial_to_plot_idx=plot_idx_gpc, plot_title_suffix=plot_title_gpc)

rem_gpc_comp = np.sum(~mask_gpc_comp)
print(f"Removed {rem_gpc_comp} GPC comparison trial(s)")
mean_avg_cost_gpc_comp = np.zeros(T)
stderr_avg_cost_gpc_comp = np.zeros(T)
if mask_gpc_comp.sum() > 0:
    c_st_gpc_comp = cum_gpc_comp[mask_gpc_comp]
    n_st_gpc_comp = c_st_gpc_comp.shape[0]
    avg_gpc_comp = c_st_gpc_comp / (np.arange(1, T + 1)[None, :])
    mean_avg_cost_gpc_comp = np.mean(avg_gpc_comp, axis=0)
    stderr_avg_cost_gpc_comp = np.std(avg_gpc_comp, axis=0) / np.sqrt(n_st_gpc_comp)

# Adaptive GPC for Comparison
agpc_hyperparam_details_str = ""
if best_adaptive_gpc_hyperparams:
    agpc_hyperparam_details_str = f": eta={tuned_agpc_eta_comp:.2f}, dens={tuned_agpc_exp_dens_comp}"
agpc_label_suffix = f" ({agpc_params_source_message}{agpc_hyperparam_details_str})"

print(f"Adaptive GPC for Comparison ({agpc_params_source_message}): {comp_num_trials} trials, {T} steps")
comp_costs_agpc, comp_trajectories_agpc, agpc_u_history = vmap_agpc(
    comp_trial_keys, D, N, P, T, GPC_M,
    agpc_base_init_scale, agpc_base_lr, LR_DECAY, agpc_base_R_M,
    ADAPTIVE_GPC_HH, tuned_agpc_eta_comp, 1e-6, 1e6, ADAPTIVE_GPC_LIFE_LOWER_BOUND, tuned_agpc_exp_dens_comp
)
comp_costs_agpc_np = np.array(comp_costs_agpc)
print("Adaptive GPC for Comparison complete.")
cum_agpc_comp = np.cumsum(comp_costs_agpc_np, axis=1)
mask_agpc_comp = cum_agpc_comp[:, -1] < comp_threshold

plot_idx_agpc = 0
plot_title_agpc = f"(Trial {plot_idx_agpc} - All Unstable)"
if np.any(mask_agpc_comp):
    final_costs_agpc = cum_agpc_comp[:, -1]
    costs_for_min_finding = np.where(mask_agpc_comp, final_costs_agpc, np.inf)
    best_stable_idx = np.argmin(costs_for_min_finding)
    plot_idx_agpc = best_stable_idx
    plot_title_agpc = f"(Best Stable Trial #{plot_idx_agpc})"

save_brax_rollout(np.array(comp_trajectories_agpc), "adaptive_gpc", trial_idx=plot_idx_agpc)
plot_state_trajectory(np.array(comp_trajectories_agpc), np.array(agpc_u_history), "adaptive_gpc", dt_for_plot, trial_to_plot_idx=plot_idx_agpc, plot_title_suffix=plot_title_agpc)

rem_agpc_comp = np.sum(~mask_agpc_comp)
print(f"Removed {rem_agpc_comp} Adaptive GPC comparison trial(s)")
mean_avg_cost_agpc_comp = np.zeros(T)
stderr_avg_cost_agpc_comp = np.zeros(T)
if mask_agpc_comp.sum() > 0:
    c_st_agpc_comp = cum_agpc_comp[mask_agpc_comp]
    n_st_agpc_comp = c_st_agpc_comp.shape[0]
    avg_agpc_comp = c_st_agpc_comp / (np.arange(1, T + 1)[None, :])
    mean_avg_cost_agpc_comp = np.mean(avg_agpc_comp, axis=0)
    stderr_avg_cost_agpc_comp = np.std(avg_agpc_comp, axis=0) / np.sqrt(n_st_agpc_comp)

# SFC for Comparison 
sfc_init_scale_for_comp = tuned_sfc_init_scale_comp
sfc_lr_for_comp = tuned_sfc_lr_comp
sfc_R_M_for_comp = tuned_sfc_R_M_comp
sfc_gamma_for_comp = tuned_sfc_gamma_comp
sfc_hyperparam_details_str = ""
if best_sfc_hyperparams:
    sfc_hyperparam_details_str = f": is={best_sfc_hyperparams['init_scale']:.3f}, lr={best_sfc_hyperparams['lr']:.1e}, R_M={best_sfc_hyperparams['R_M']:.3f}, g={best_sfc_hyperparams['gamma']:.3f}"
sfc_label_suffix = f" ({sfc_params_source_message}{sfc_hyperparam_details_str})"

comp_sfc_filters = sfc_compute_filters(SFC_M_HIST, SFC_H_FILT, sfc_gamma_for_comp)

print(f"SFC for Comparison ({sfc_params_source_message}): {comp_num_trials} trials, {T} steps, m_hist={SFC_M_HIST}, h_filt={SFC_H_FILT}, init_scale={sfc_init_scale_for_comp:.3f}, lr={sfc_lr_for_comp:.1e}, R_M={sfc_R_M_for_comp:.3f}, gamma={sfc_gamma_for_comp:.3f}")
comp_costs_sfc, comp_trajectories_sfc, sfc_u_history = vmapped_sfc_runner(
    comp_trial_keys, D, N, P, 
    T, 
    SFC_M_HIST, SFC_H_FILT, 
    sfc_init_scale_for_comp, sfc_lr_for_comp, LR_DECAY, sfc_R_M_for_comp, # Use global LR_DECAY
    comp_sfc_filters
)
comp_costs_sfc_np = np.array(comp_costs_sfc)
print("SFC for Comparison complete.")
# Determine dt for plotting and save visualizations
dt_for_plot = brax_env_dt if SYSTEM_TYPE == "brax" else DT
cum_sfc_comp = np.cumsum(comp_costs_sfc_np, axis=1)
mask_sfc_comp = cum_sfc_comp[:,-1] < comp_threshold

# Logic to select the best stable trial for plotting
plot_idx_sfc = 0
plot_title_sfc = f"(Trial {plot_idx_sfc} - All Unstable)"
if np.any(mask_sfc_comp):
    final_costs_sfc = cum_sfc_comp[:, -1]
    costs_for_min_finding = np.where(mask_sfc_comp, final_costs_sfc, np.inf)
    best_stable_idx = np.argmin(costs_for_min_finding)
    plot_idx_sfc = best_stable_idx
    plot_title_sfc = f"(Best Stable Trial #{plot_idx_sfc})"

save_brax_rollout(np.array(comp_trajectories_sfc), "sfc", trial_idx=plot_idx_sfc)
plot_state_trajectory(np.array(comp_trajectories_sfc), np.array(sfc_u_history), "sfc", dt_for_plot, trial_to_plot_idx=plot_idx_sfc, plot_title_suffix=plot_title_sfc)

rem_sfc_comp = np.sum(~mask_sfc_comp)
print(f"Removed {rem_sfc_comp} SFC comparison trial(s)")
mean_avg_cost_sfc_comp = np.zeros(T)
stderr_avg_cost_sfc_comp = np.zeros(T)
if mask_sfc_comp.sum() > 0:
    c_st_sfc_comp = cum_sfc_comp[mask_sfc_comp]
    n_st_sfc_comp = c_st_sfc_comp.shape[0]
    avg_sfc_comp = c_st_sfc_comp / (np.arange(1, T + 1)[None, :])
    mean_avg_cost_sfc_comp = np.mean(avg_sfc_comp, axis=0)
    stderr_avg_cost_sfc_comp = np.std(avg_sfc_comp, axis=0) / np.sqrt(n_st_sfc_comp)

# DSC for Comparison 
dsc_init_scale_for_comp = tuned_dsc_init_scale_comp
dsc_lr_for_comp = tuned_dsc_lr_comp
dsc_R_M_for_comp = tuned_dsc_R_M_comp
dsc_gamma_for_comp = tuned_dsc_gamma_comp
dsc_hyperparam_details_str = ""
if best_dsc_hyperparams:
    dsc_hyperparam_details_str = f": is={best_dsc_hyperparams['init_scale']:.3f}, lr={best_dsc_hyperparams['lr']:.1e}, R_M={best_dsc_hyperparams['R_M']:.4f}, g={best_dsc_hyperparams['gamma']:.3f}"
dsc_label_suffix = f" ({dsc_params_source_message}{dsc_hyperparam_details_str})"

comp_dsc_filters = sfc_compute_filters(DSC_M_HIST, DSC_H_FILT, dsc_gamma_for_comp) 

print(f"DSC for Comparison ({dsc_params_source_message}): {comp_num_trials} trials, {T} steps, m_hist={DSC_M_HIST}, h_filt={DSC_H_FILT}, init_scale={dsc_init_scale_for_comp:.3f}, lr={dsc_lr_for_comp:.1e}, R_M={dsc_R_M_for_comp:.4f}, gamma={dsc_gamma_for_comp:.3f}")
comp_costs_dsc, comp_trajectories_dsc, dsc_u_history = vmapped_dsc_runner(
    comp_trial_keys, D, N, P, 
    T,
    DSC_M_HIST, DSC_H_FILT,
    dsc_init_scale_for_comp, dsc_lr_for_comp, LR_DECAY, dsc_R_M_for_comp, # Use global LR_DECAY
    comp_dsc_filters
)
comp_costs_dsc_np = np.array(comp_costs_dsc)
print("DSC for Comparison complete.")
# Determine dt for plotting and save visualizations
dt_for_plot = brax_env_dt if SYSTEM_TYPE == "brax" else DT
cum_dsc_comp = np.cumsum(comp_costs_dsc_np, axis=1)
mask_dsc_comp = cum_dsc_comp[:,-1] < comp_threshold

# Logic to select the best stable trial for plotting
plot_idx_dsc = 0
plot_title_dsc = f"(Trial {plot_idx_dsc} - All Unstable)"
if np.any(mask_dsc_comp):
    final_costs_dsc = cum_dsc_comp[:, -1]
    costs_for_min_finding = np.where(mask_dsc_comp, final_costs_dsc, np.inf)
    best_stable_idx = np.argmin(costs_for_min_finding)
    plot_idx_dsc = best_stable_idx
    plot_title_dsc = f"(Best Stable Trial #{plot_idx_dsc})"

save_brax_rollout(np.array(comp_trajectories_dsc), "dsc", trial_idx=plot_idx_dsc)
plot_state_trajectory(np.array(comp_trajectories_dsc), np.array(dsc_u_history), "dsc", dt_for_plot, trial_to_plot_idx=plot_idx_dsc, plot_title_suffix=plot_title_dsc)

rem_dsc_comp = np.sum(~mask_dsc_comp)
print(f"Removed {rem_dsc_comp} DSC comparison trial(s)")

mean_avg_cost_dsc_comp = np.zeros(T)
stderr_avg_cost_dsc_comp = np.zeros(T)
if mask_dsc_comp.sum() > 0:
    c_st_dsc_comp = cum_dsc_comp[mask_dsc_comp]
    n_st_dsc_comp = c_st_dsc_comp.shape[0]
    avg_dsc_comp = c_st_dsc_comp / (np.arange(1, T + 1)[None, :])
    mean_avg_cost_dsc_comp = np.mean(avg_dsc_comp, axis=0)
    stderr_avg_cost_dsc_comp = np.std(avg_dsc_comp, axis=0) / np.sqrt(n_st_dsc_comp)
else: 
    print("No stable DSC trials for comparison plot.")


# Final Comparison Plot
plt.figure(figsize=(12, 7))
# GPC 
plt.plot(mean_avg_cost_gpc_comp, label=f"GPC {gpc_label_suffix}", color="blue")
plt.fill_between(np.arange(T), 
                mean_avg_cost_gpc_comp - stderr_avg_cost_gpc_comp,
                mean_avg_cost_gpc_comp + stderr_avg_cost_gpc_comp,
                alpha=0.2, color="blue")
# Adaptive GPC
plt.plot(mean_avg_cost_agpc_comp, label=f"Adaptive GPC {agpc_label_suffix}", color="purple")
plt.fill_between(np.arange(T),
                mean_avg_cost_agpc_comp - stderr_avg_cost_agpc_comp,
                mean_avg_cost_agpc_comp + stderr_avg_cost_agpc_comp,
                alpha=0.2, color="purple")
# SFC 
plt.plot(mean_avg_cost_sfc_comp, label=f"SFC {sfc_label_suffix}", color="green")
plt.fill_between(np.arange(T), 
                mean_avg_cost_sfc_comp - stderr_avg_cost_sfc_comp,
                mean_avg_cost_sfc_comp + stderr_avg_cost_sfc_comp,
                alpha=0.2, color="green")
# DSC 
plt.plot(mean_avg_cost_dsc_comp, label=f"DSC {dsc_label_suffix}", color="red")
plt.fill_between(np.arange(T), 
                mean_avg_cost_dsc_comp - stderr_avg_cost_dsc_comp,
                mean_avg_cost_dsc_comp + stderr_avg_cost_dsc_comp,
                alpha=0.2, color="red")

plt.xlabel("Time Step")
plt.ylabel("Cumulative Average Cost")

title_parts = []
if best_gpc_hyperparams: title_parts.append("Tuned GPC")
else: title_parts.append("Default GPC")
if best_adaptive_gpc_hyperparams: title_parts.append("Tuned AdaptiveGPC")
if best_sfc_hyperparams: title_parts.append("Tuned SFC")
else: title_parts.append("Default SFC")
if best_dsc_hyperparams: title_parts.append("Tuned DSC")
else: title_parts.append("Default DSC")
comparison_plot_title = f"Comparison: {', '.join(title_parts)} (Transition={LDS_TRANSITION_TYPE}, Noise={LDS_NOISE_TYPE})"

plt.title(comparison_plot_title)
plt.legend(fontsize='small')
plt.grid(True)

# Determine a reasonable y-limit based on all plotted data
all_means_for_ylim = [mean_avg_cost_gpc_comp, mean_avg_cost_agpc_comp, mean_avg_cost_sfc_comp, mean_avg_cost_dsc_comp]
all_stderr_for_ylim = [stderr_avg_cost_gpc_comp, stderr_avg_cost_agpc_comp, stderr_avg_cost_sfc_comp, stderr_avg_cost_dsc_comp]
max_y = 0
for mean_data, stderr_data in zip(all_means_for_ylim, all_stderr_for_ylim):
    if mean_data.size > 0 and stderr_data.size > 0: # Check if array is not empty
        current_max = np.max(mean_data + stderr_data)
        if not np.isnan(current_max) and not np.isinf(current_max) : # Check for valid numbers
            max_y = max(max_y, current_max)
if max_y == 0: max_y = 100 # Default if no valid data

plt.ylim(bottom=0, top=max_y * 1.1 + 1) 
plt.tight_layout()

base_filename_comp = "comparison_cumulative_average_costs_tuned.png"
plots_dir_comp = get_plot_dir()
full_save_path_comp = os.path.join(plots_dir_comp, base_filename_comp)
plt.savefig(full_save_path_comp)
print(f"Comparison plot saved to {full_save_path_comp}")
