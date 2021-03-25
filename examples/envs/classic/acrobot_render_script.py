
import deluca
import gym
import jax
from deluca.envs import Acrobot
from deluca.agents import ILQR
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
agent = ILQR()
agent.train(Acrobot(horizon=10), 10)


# for loop version
T = 75
env = Acrobot()
env = Monitor(env, './video', video_callable=lambda episode_id: True, force=True)
print(env.reset())
reward = 0
for i in range(T):
    (env, agent), r = loop((env, agent), i)
    reward += r

reward_forloop = reward
print('reward_forloop = ' + str(reward_forloop))
env.close()