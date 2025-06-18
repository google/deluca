# System parameters
d = 10
n = 3
min_eig = 0.5
max_eig = 1.5

# Model parameters
m = 5
k = 5 # Action horizon, must be <= m for GPC, and == h for SFC
# For SFC
h = 4
gamma = 0.9
hidden_dims = None
hidden_dims_nn = [8]

# Disturbance parameters
disturbance_params = {
    'sinusoidal': {
        'noise_amplitude': 1.0,
        'noise_frequency': 0.1,
    },
    'gaussian': {
        'dist_std': 1.0,
    },
    'zero': {}
}

# Training parameters
learning_rate = 1e-3
R_M = 1.0
num_steps = 1000
window_length = 100

# Experiment parameters
num_trials = 10
seed = 42 