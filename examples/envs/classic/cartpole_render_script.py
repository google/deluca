import deluca
import gym
# import gnwrapper
import jax.numpy as jnp
from deluca.envs import CartPole
from deluca.agents import Zero
from pyvirtualdisplay import Display
import matplotlib.pyplot as plt
from IPython import display

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
    print('control:' + str(control))
    print('env.state:' + str(env.state))
    _, reward, _, _ = env.step(control)
    show_state(env, step=i)
    return (env, agent), reward

env = CartPole()
agent = Zero(())
# display_dummy = Display(visible=False, size=(1400, 900))
# display_dummy.start()
T = 75
print(env.reset())
reward = 0
for i in range(T):
    (env, agent), r = loop((env, agent), i)
    reward += r
# env.reset()
# env.display()
reward_forloop = reward
print('reward_forloop = ' + str(reward_forloop))