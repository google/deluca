# brax_config.py

# --- Brax Environment ---
BRAX_ENV_NAME = "inverted_pendulum"  # e.g., "ant", "humanoid", "reacher"

# --- Experiment ---
seed = 0
num_trials = 5
num_steps = 200
learning_rate = 1e-3
R_M = 1.0  # For optimizer clipping

# --- GPC Model ---
m_gpc = 5
k_gpc = 5

# --- SFC Model ---
m_sfc = 5
h_sfc = 10
k_sfc = 5
gamma = 0.99

# --- Model Architecture ---
# Set to a tuple like (64, 64) for a neural network model,
# or None for a linear model.
hidden_dims = None

# --- Disturbances ---
# Parameters for different disturbance types
disturbance_params = {
    "zero": {},
    "gaussian": {
        "dist_std": 0.1,
    },
    "sinusoidal": {
        "noise_amplitude": 0.5,
        "noise_frequency": 0.1,
    }
} 