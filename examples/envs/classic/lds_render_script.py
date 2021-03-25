import deluca
import gym
# import gnwrapper
import jax.numpy as jnp
from deluca.envs import LDS
from deluca.agents import GPC
import matplotlib.pyplot as plt
from IPython import display
import gym
from gym.wrappers import Monitor
from tqdm import tqdm

def disable_view_window():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor

def show_state(env, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    # plt.title("%s | Step: %d %s" % (env._spec.id,step, info))
    plt.title("%s | Step: %d %s" % (type(env).__name__,step, info))
    plt.axis('off')

    display.clear_output(wait=True)
    # time.sleep(0.1)
    display.display(plt.gcf())

def smooth_show_state(env, step=0, prev_state=None, info=""):
    new_state = env.state
    dist = ((new_state[1] - prev_state[1])**2 + (new_state[0] - prev_state[0])**2) ** 0.5
    num_steps = max(1, int(dist/0.1)) # enforce num_steps >= 1
    for i in range(1, num_steps + 1):
        intermediate_state = prev_state * float(1 - i/(num_steps)) + new_state * float(i/(num_steps))
        env.state = intermediate_state
        plt.figure(3)
        plt.clf()
        plt.imshow(env.render(mode='rgb_array'))
        # plt.title("%s | Step: %d %s" % (env._spec.id,step, info))
        plt.title("%s | Step: %d %s" % (type(env).__name__,step, info))
        plt.axis('off')

        display.clear_output(wait=True)
        display.display(plt.gcf())

def loop(context, i):
    env, controller = context
    A, B, state = env.A, env.B, env.state
    try:
        action = controller(state, A, B)
    except:
        action = controller(state)
    prev_state = env.state
    env.step(action)
    smooth_show_state(env, step=i, prev_state=prev_state)
    error = jnp.linalg.norm(state)+jnp.linalg.norm(action)
    return (env, controller), error

def get_errs(T, controller, A, B):
    env = LDS(state_size=3, action_size=1, A=A, B=B)
    errs = [0.0]
    disable_view_window()
    
    for i in tqdm(range(1, T)):
        (env, controller), error = loop((env, controller), i)
        errs.append(float(error))
    
    return errs



# for loop version
T = 20
A,B = jnp.array([[1.,.5,1.], [0,1.,0.], [0.,0.1,0.]]), jnp.array([[0],[1.2],[0.]])
gpc = GPC(A, B)
gpc_errs = get_errs(T, gpc, A, B)
print("GPC incurs ", jnp.mean(jnp.asarray(gpc_errs)), " loss under gaussian iid noise")