import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import lax
from deluca.envs import LearnedLung
from deluca.agents import PID
import matplotlib.pyplot as plt
from IPython import display

def show_state(env, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array', scale=2.0))
    # plt.title("%s | Step: %d %s" % (env._spec.id,step, info))
    plt.title("%s | Step: %d %s" % (type(env).__name__,step, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())

def loop(context, i):
    env, agent = context
    agent_in, agent_out = agent
    error = env.observation['target'] - env.observation['measured']
    control_in = agent_in(error)
    control_out = agent_out(error)
    _, reward, _, _ = env.step((control_in, control_out))
    show_state(env, step=i)
    return (env, (agent_in, agent_out)), reward


# LearnedLung env
lung = LearnedLung.from_torch("learned_lung_C20_R20_PEEP10.pkl")

# for loop version
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