import deluca
import gym
# import gnwrapper
import jax.numpy as jnp
from deluca.envs import BalloonLung
from deluca.agents import PID
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

'''
def loop(context, i):
    env, agent = context
    control = agent(env.state)
    print('control:' + str(control))
    print('env.state:' + str(env.state))
    _, reward, _, _ = env.step(control)
    show_state(env, step=i)
    return (env, agent), reward'''

def loop(context, i):
    env, agent = context
    agent_in, agent_out = agent
    error = env.observation['target'] - env.observation['measured']
    control_in = agent_in(error)
    control_out = agent_out(error)
    _, reward, _, _ = env.step((control_in, control_out))
    if i > 10:
        show_state(env, step=i)
    return (env, (agent_in, agent_out)), reward


# DelayLung env
lung = BalloonLung(leak=False,
                   peep_valve=5.0,
                   PC=40.0,
                   P0=0.0,
                   C=10.0,
                   R=15.0,
                   dt=0.03,
                   waveform=None,
                   reward_fn=None)
lung = Monitor(lung, './video', force=True)

T = 100
xs = jnp.array(jnp.arange(T))
agent_in = PID([3.0, 4.0, 0.0])
agent_out = PID([3.0, 4.0, 0.0])
print(lung.reset())
reward = 0
for i in range(T):
    (lung, (agent_in, agent_out)), r = loop((lung, (agent_in, agent_out)), i)
    reward += r
reward_forloop = reward
lung.close()