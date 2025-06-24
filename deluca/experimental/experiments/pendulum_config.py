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
k = 10 # action horizon 

# Model parameters
m_gpc = 10  # history length for GPC

m_sfc = 15  # history length for SFC
h_sfc = 10   # filter horizon for SFC
gamma = 0.9
hidden_dims = [16]

# Disturbance parameters
disturbance_params = {
    'sinusoidal': {
        'noise_amplitude': 0.1,
        'noise_frequency': 0.5,
    },
    'gaussian': {
        'dist_std': 0.1,
    },
    'zero': {}
}

model_type = 'fully_connected'

# Training parameters
learning_rate = 1e-3
R_M = 1.0
num_steps = 500
gradient_updates_per_step = 10

# Experiment parameters
num_trials = 10
seed = 42 