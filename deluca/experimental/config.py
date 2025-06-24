# System parameters
sim_type = 'lds' # 'lds' or 'pendulum' or 'brax'

#Specify if sim_type is 'lds'
d = 5
n = 2
min_eig = 0.5
max_eig = 0.9

#Specify if sim_type is 'pendulum'
max_torque = 1.0
mass = 1.0
length = 1.0
g = 9.81
dt = 0.02

# Model parameters
model_type = 'fully_connected'  # 'fully_connected' or 'sequential' or 'grid'
algorithm_type = 'gpc' # 'gpc' or 'sfc'
m = 20
k = 4 # Action horizon, must be <= m for GPC, and == h for SFC
# For SFC
h = 5
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
        'std': 0.5,
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

debug=False