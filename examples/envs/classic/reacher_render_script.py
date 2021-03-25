import deluca
import gym
import jax
from deluca.envs import Reacher
from deluca.agents import ILQR
from deluca.agents._ilqr import rollout
import matplotlib.pyplot as plt
from IPython import display
import gym
from gym.wrappers import Monitor

def show_state(env, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    # plt.title("%s | Step: %d %s" % (env._spec.id,step, info))
    plt.title("%s | Step: %d %s" % (type(env).__name__,step, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())

def loop(context, i):
    env, agent = context
    control = agent(env.state)
    _, reward, _, _ = env.step(control)
    show_state(env, step=i)
    return (env, agent), reward


# ILQR
ALPHA = 1.0
env_true, env_sim = Reacher(-10.), Reacher()
print('-------------- ilqr_sim ----------------')
ilqr_sim = ILQR()
ilqr_sim.train(env_sim, 5, alpha=ALPHA)
print('----------- compute zero_cost -----------')
env_ZEROCOST = Reacher(-10.)
_,_,ZEROCOST = rollout(env_ZEROCOST, ilqr_sim.U, ilqr_sim.k, ilqr_sim.K, ilqr_sim.X)
print('ZEROCOST:' + str(ZEROCOST))
print('-------------- ilqr_true ----------------')
agent = ILQR()
agent.train(env_true, 10, ilqr_sim.U, alpha=ALPHA)

# for loop version
T = 350
env = Reacher(-10.)
env = Monitor(env, './video', video_callable=lambda episode_id: True, force=True)
print(env.reset())
reward = 0
for i in range(T):
    (env, agent), r = loop((env, agent), i)
    reward += r
reward_forloop = reward
print('reward_forloop = ' + str(reward_forloop))
env.close()