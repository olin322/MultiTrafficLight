### again this code doesn't work


import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Define a function to create a single environment
def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    return _init

# Create a vectorized environment with 4 subprocesses
env_id = "CartPole-v1"
num_envs = 4
env = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])

# Create and train a PPO2 agent on the vectorized environment
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Save the trained model
model.save("my_model")
