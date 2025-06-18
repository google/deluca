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

# Model parameters
m_gpc = 5  # history length for GPC
k_gpc = 5   # action horizon for GPC

m_sfc = 10  # history length for SFC
h_sfc = 5   # filter horizon for SFC
k_sfc = 5   # action horizon for SFC
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

# Training parameters
learning_rate = 1e-3
R_M = 1.0
num_steps = 1000

# Experiment parameters
num_trials = 10
seed = 42 