"""
note using multi-processing does not make training faster
as stated in the anser https://github.com/hill-a/stable-baselines/issues/1113

```
Is it true that having multiple envs even though running sequentially, 
will make the training stable?
```
At the end, both are synchronous, 
so it does not change anything for the agent if you use a DummyVecEnv with 4 envs or a SubprocVecEnv with 4 envs. 
What may change is the fps (cf. notebook for a comparison).
"""




import time

import gymnasium as gym
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from typing import Callable


def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


env_id = "CartPole-v1"
num_cpu = 16  # Number of processes to use
# # Create the vectorized environment
# env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

# model = A2C("MlpPolicy", env, verbose=0)

vec_env = make_vec_env(env_id, n_envs=num_cpu)

model = A2C("MlpPolicy", vec_env, verbose=0)

eval_env = gym.make(env_id)

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

n_timesteps = 400000

# Multiprocessed RL Training
start_time = time.time()
model.learn(n_timesteps)
total_time_multi = time.time() - start_time

print(
    f"Took {total_time_multi:.2f}s for multiprocessed version - {n_timesteps / total_time_multi:.2f} FPS"
)

# Single Process RL Training
single_process_model = A2C("MlpPolicy", env_id, verbose=0)

start_time = time.time()
single_process_model.learn(n_timesteps)
total_time_single = time.time() - start_time

print(
    f"Took {total_time_single:.2f}s for single process version - {n_timesteps / total_time_single:.2f} FPS"
)

print(
    "Multiprocessed training is {:.2f}x faster!".format(
        total_time_single / total_time_multi
    )
)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")
# import gymnasium as gym
# from stable_baselines3 import SAC

# from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.utils import set_random_seed

# # below is an example
# def make_env(env_id: str, rank: int, seed: int = 0):
# 	"""
# 	Utility function for multiprocessed env.
# 	:param env_id: the environment ID
# 	:param num_env: the number of environments you wish to have in subprocesses
# 	:param seed: the inital seed for RNG
# 	:param rank: index of the subprocess
# 	"""
# 	def _init():
# 		env = gym.make(env_id, render_mode="human")
# 		env.reset(seed=seed + rank)
# 		return env
# 	set_random_seed(seed)
# 	return _init

# if __name__ == "__main__":
# 	env_id = "CartPole-v1"
# 	num_process = 8 # Number of processes to use
# 	# Create the vectorized environment
# 	vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_process)])
	
# 	# Stable Baselines provides you with make_vec_env() helper
# 	# which does exactly the previous steps for you.
# 	# You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
# 	# env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
# 	model = SAC("MlpPolicy", vec_env, verbose=1)
# 	model.learn(total_timesteps=25_000)
	
# 	obs = vec_env.reset()
# 	for _ in range(1000):
# 		action, _states = model.predict(obs)
# 		obs, rewards, dones, info = vec_env.step(action)
# 		vec_env.render()