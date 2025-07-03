from deluca.agents._random import SimpleRandom
from deluca.core import Env

class Learner:
  env: Env

  def __init__(self, env: Env):
    self.env = env

  
  def generate_trajectories(self, N, T):
    """Generate N trajectories of length T. Use random policy for now."""
    agent = SimpleRandom(self.env.action_size)
    
    for i in range(N):
      state = self.env.init()

      for t in range(T):
        action = agent(state)
        state, obs = self.env(state, action[0])
        yield state, obs
        
  