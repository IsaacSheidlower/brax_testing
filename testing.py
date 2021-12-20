import functools
import time
import gym
import brax 
from IPython.display import HTML, Image 
from brax import envs
from brax import jumpy as jp
from brax.envs import to_torch
from brax.io import html
from brax.io import image
import jax
from jax import numpy as jnp
import torch
import numpy as np
import matplotlib.pyplot as plt

environment = "ant"  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'reacher', 'walker2d', 'fetch', 'grasp', 'ur5e']
env = envs.create(env_name=environment)
state = env.reset(rng=jp.random_prngkey(seed=0))

#print(image.render_array(env.sys, state.qp, 100, 100))

for i in range(100):
  # wiggle sinusoidally
  action = jp.ones((env.action_size,)) * jp.sin(i * jp.pi / 15)
  #state = jax.jit(env.step)(state, action)
  state = env.step(state, action)
  #plt.imshow(image.render_array(env.sys, state.qp, 100, 100), interpolation='none')
  #plt.pause(0.001)
#plt.show()

entry_point = functools.partial(envs.create_gym_env, env_name='ant')
if 'brax-ant-v0' not in gym.envs.registry.env_specs:
  gym.register('brax-ant-v0', entry_point=entry_point)

# create a gym environment that contains 4096 parallel ant environments
gym_env = gym.make("brax-ant-v0", batch_size=1)

# wrap it to interoperate with torch data structures
gym_env = to_torch.JaxToTorchWrapper(gym_env, device='cpu')

# jit compile env.reset
obs = gym_env.reset()

# jit compile env.step
action = torch.rand(gym_env.action_space.shape, device='cpu') 
obs, reward, done, info = gym_env.step(action)

print(reward)