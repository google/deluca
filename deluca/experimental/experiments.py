"""Experiment script for comparing GPC and SFC models."""

import argparse
import functools
from typing import Callable, Tuple, Any, Optional, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn

from deluca.experimental.agents.agent import policy_loss, update_agentstate, AgentState
from deluca.experimental.agents.gpc import GPCModel
from deluca.experimental.agents.sfc import SFCModel
from deluca.experimental.enviornments.disturbances.sinusoidal import sinusoidal_disturbance
from deluca.experimental.enviornments.disturbances.gaussian import gaussian_disturbance
from deluca.experimental.enviornments.disturbances.zero import zero_disturbance
from deluca.experimental.enviornments.sim_and_output.lds import lds_sim, lds_output
from deluca.experimental.enviornments.cost_functions.quadratic_cost import quadratic_cost_evaluate
from deluca.experimental.enviornments.env import step

def create_random_system(
    key: jax.random.PRNGKey,
    d: int,
    n: int,
    min_eig: float,
    max_eig: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create random system matrices A and B.
    
    Args:
        key: Random key
        d: State dimension
        n: Input dimension
        min_eig: Minimum eigenvalue of A
        max_eig: Maximum eigenvalue of A
        
    Returns:
        A: State matrix of shape (d, d)
        B: Input matrix of shape (d, n)
    """
    key1, key2 = jax.random.split(key)
    
    # Create diagonal A matrix with eigenvalues between min_eig and max_eig
    A = jnp.diag(jax.random.uniform(key1, (d,), minval=min_eig, maxval=max_eig))
    
    # Create random B matrix
    B = jax.random.normal(key2, (d, n))
    
    return A, B

def create_random_cost_matrices(
    key: jax.random.PRNGKey,
    d: int,
    n: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create random positive semidefinite cost matrices Q and R.
    
    Args:
        key: Random key
        d: State dimension
        n: Input dimension
        
    Returns:
        Q: State cost matrix of shape (d, d)
        R: Input cost matrix of shape (n, n)
    """
    # Create random positive semidefinite matrices
    key1, key2 = jax.random.split(key)
    Q = jax.random.normal(key1, (d, d))
    R = jax.random.normal(key2, (n, n))
    
    # Make them positive semidefinite
    Q = Q.T @ Q
    R = R.T @ R
    
    return Q, R

def run_trial(
    key: jax.random.PRNGKey,
    model_type: str,
    d: int,
    n: int,
    m: int,
    h: Optional[int],
    gamma: Optional[float],
    hidden_dims: Optional[Sequence[int]],
    A: jnp.ndarray,
    B: jnp.ndarray,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    disturbance_type: str,
    disturbance_params: dict,
    learning_rate: float,
    R_M: float,
    num_steps: int,
    window_length: int
) -> jnp.ndarray:
    """Run a single trial."""
    # Create simulator and cost function
    sim = functools.partial(lds_sim, A=A, B=B)
    output_map = functools.partial(lds_output, C=jnp.eye(d))  # Identity output map
    cost_fn = functools.partial(quadratic_cost_evaluate, Q=Q, R=R)
    
    # Create disturbance
    if disturbance_type == 'sinusoidal':
        disturbance = lambda d, dist_std, t, key: sinusoidal_disturbance(key, **disturbance_params)
    elif disturbance_type == 'gaussian':
        disturbance = lambda d, dist_std, t, key: gaussian_disturbance(key, **disturbance_params)
    elif disturbance_type == 'zero':
        disturbance = lambda d, dist_std, t, key: zero_disturbance(key, **disturbance_params)
    else:
        raise ValueError(f"Unknown disturbance type: {disturbance_type}")
    
    # Create model
    if model_type == 'gpc':
        model = GPCModel(d=d, n=n, m=m, hidden_dims=hidden_dims)
    elif model_type == 'sfc':
        if h is None or gamma is None:
            raise ValueError("h and gamma must be provided for SFC")
        model = SFCModel(d=d, n=n, m=m, h=h, gamma=gamma, hidden_dims=hidden_dims)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(R_M),
        optax.adam(learning_rate)
    )
    
    # Initialize model parameters and agent state
    key1, key2 = jax.random.split(key)
    params = model.init(key1, jnp.zeros((m, n, 1)))
    opt_state = optimizer.init(params)
    agentstate = AgentState(
        controller_t=0,
        z=jnp.zeros((d, 1)),
        ynat_history=jnp.zeros((m, d, 1)),
        params=params,
        opt_state=opt_state
    )
    
    # Run simulation
    losses = []
    state = jnp.zeros((d, 1))
    for t in range(num_steps):
        # Get disturbance and update state
        key1, key2 = jax.random.split(key2)
        state, output = step(
            x=state,
            u=jnp.zeros((n, 1)),  # Initial control input
            sim=sim,
            output_map=output_map,
            dist_std=1.0,  # Standard deviation for disturbance
            t_sim_step=t,
            disturbance=disturbance,
            key=key1
        )
        
        # Update agent state and compute loss
        agentstate = update_agentstate(
            agentstate=agentstate,
            w=output,
            sim=sim,
            cost_fn=cost_fn,
            model=model,
            optimizer=optimizer
        )
        
        loss = policy_loss(
            agentstate=agentstate,
            w=output,
            sim=sim,
            cost_fn=cost_fn,
            model=model
        )
        losses.append(loss)
    
    # Compute windowed losses
    windowed_losses = []
    for i in range(len(losses) - window_length + 1):
        windowed_losses.append(jnp.mean(jnp.array(losses[i:i + window_length])))
    
    return jnp.array(windowed_losses)

def main():
    parser = argparse.ArgumentParser(description='Run GPC/SFC experiments')
    
    # System parameters
    parser.add_argument('--d', type=int, required=True, help='State dimension')
    parser.add_argument('--n', type=int, required=True, help='Input dimension')
    parser.add_argument('--min_eig', type=float, required=True, help='Minimum eigenvalue of A')
    parser.add_argument('--max_eig', type=float, required=True, help='Maximum eigenvalue of A')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, choices=['gpc', 'sfc'], required=True, help='Model type')
    parser.add_argument('--m', type=int, required=True, help='History length')
    parser.add_argument('--h', type=int, help='Number of eigenvectors (for SFC)')
    parser.add_argument('--gamma', type=float, help='Decay rate (for SFC)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', help='Hidden dimensions for networks')
    
    # Disturbance parameters
    parser.add_argument('--disturbance_type', type=str, choices=['sinusoidal', 'gaussian', 'zero'], required=True, help='Disturbance type')
    parser.add_argument('--disturbance_params', type=float, nargs='+', help='Disturbance parameters')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--R_M', type=float, required=True, help='Radius for parameter projection')
    parser.add_argument('--num_steps', type=int, required=True, help='Number of steps to run')
    parser.add_argument('--window_length', type=int, required=True, help='Length of window for loss computation')
    
    # Experiment parameters
    parser.add_argument('--num_trials', type=int, required=True, help='Number of trials')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    key = jax.random.PRNGKey(args.seed)
    
    # Create random system
    key1, key2 = jax.random.split(key)
    A, B = create_random_system(key1, args.d, args.n, args.min_eig, args.max_eig)
    
    # Create random cost matrices
    key1, key2 = jax.random.split(key2)
    Q, R = create_random_cost_matrices(key1, args.d, args.n)
    
    # Parse disturbance parameters
    disturbance_params = {}
    if args.disturbance_type == 'sinusoidal':
        disturbance_params['amplitude'] = args.disturbance_params[0]
        disturbance_params['frequency'] = args.disturbance_params[1]
        disturbance_params['phase'] = args.disturbance_params[2]
    elif args.disturbance_type == 'gaussian':
        disturbance_params['mean'] = args.disturbance_params[0]
        disturbance_params['std'] = args.disturbance_params[1]
    elif args.disturbance_type == 'zero':
        pass  # No parameters needed
    
    # Run trials
    all_losses = []
    for i in range(args.num_trials):
        key1, key2 = jax.random.split(key2)
        losses = run_trial(
            key=key1,
            model_type=args.model_type,
            d=args.d,
            n=args.n,
            m=args.m,
            h=args.h,
            gamma=args.gamma,
            hidden_dims=args.hidden_dims,
            A=A,
            B=B,
            Q=Q,
            R=R,
            disturbance_type=args.disturbance_type,
            disturbance_params=disturbance_params,
            learning_rate=args.learning_rate,
            R_M=args.R_M,
            num_steps=args.num_steps,
            window_length=args.window_length
        )
        all_losses.append(losses)
    
    # Compute mean and standard error
    mean_losses = jnp.mean(jnp.array(all_losses), axis=0)
    std_losses = jnp.std(jnp.array(all_losses), axis=0) / jnp.sqrt(args.num_trials)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(mean_losses, label='Mean Loss')
    plt.fill_between(
        range(len(mean_losses)),
        mean_losses - std_losses,
        mean_losses + std_losses,
        alpha=0.2
    )
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'{args.model_type.upper()} Performance')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main() 