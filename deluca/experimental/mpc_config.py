# Pendulum environment parameters
max_torque = 1.0
m = 1.0  # mass
l = 1.0  # length
g = 9.81
dt = 0.02

# System dimensions
d = 2  # state dimension (theta, thdot)
n = 1  # action dimension (torque)
p = 2  # output dimension (same as state)
min_eig = 0.5
max_eig = 0.9

# Model parameters
k = 5  # action horizon for MPC
hidden_dims = [64, 64]

# Disturbance parameters
disturbance_type = 'gaussian'  # 'sinusoidal', 'gaussian', or 'zero'
disturbance_params = {
    'sinusoidal': {
        'noise_amplitude': 0.1,
        'noise_frequency': 0.1,
    },
    'gaussian': {
        'mean': 0.0,
        'std': 0.1,
    },
    'zero': {}
}

# Training parameters
learning_rate = 1e-3
R_M = 1.0 # For optimizer clipping
num_steps = 5000
gradient_updates_per_step = 100

# Experiment parameters
num_trials = 10
seed = 42 