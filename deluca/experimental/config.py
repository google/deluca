# System parameters
d = 10
n = 3
min_eig = 0.5
max_eig = 1.5

# Model parameters
model_type = 'fully_connected'  # 'fully_connected' or 'sequential' or 'grid'
algorithm_type = 'gpc' # 'gpc' or 'sfc'
m = 5
k = 5 # Action horizon, must be <= m for GPC, and == h for SFC
# For SFC
h = 4
gamma = 0.9
hidden_dims = [64, 64]

# Disturbance parameters
disturbance_type = 'gaussian'  # 'sinusoidal', 'gaussian', or 'zero'
disturbance_params = {
    'sinusoidal': {
        'amplitude': 1.0,
        'frequency': 0.1,
        'phase': 0.0,
    },
    'gaussian': {
        'mean': 0.0,
        'std': 1.0,
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
