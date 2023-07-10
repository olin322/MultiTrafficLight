###############################
# The following code runs in  #
# stable-baselines3[extra]    #
###############################

import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3 import SAC
# from gymnasium.envs.registration import register

def trainHumanoid(i: int):
    env = gym.make('HumanoidStandup-v4', render_mode="human")
    load_model_name = f"./savedModels/HumanoidStandup-v4_{i}M"
    model = SAC.load(load_model_name)
    model.set_env(env)
    model.learn(1000_000, progress_bar=True)
    model_name = f"./savedModels/HumanoidStandup-v4_{i+1}M"
    model.save(model_name)
    
# def multiHumanoidStandup():
#     env_id = "CartPole-v1"
#     num_cpu = 8  # Number of processes to use
#     # Create the vectorized environment
#     env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
#     model = A2C("MlpPolicy", env, verbose=0)

for i in range(4,21):
    trainHumanoid(i)


##########################################################################

# env = gym.make("HumanoidStandup-v4", render_mode="rgb_array")

# model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1_000)
# model = SAC.load("./savedModels/HumanoidStandup-v4_4M")
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")



##############################################################################

# env = gym.make("HumanoidStandup-v4", render_mode="human")
# model = SAC("MlpPolicy", env, verbose=1)
# # model.learn(total_timesteps=10_000)
# model = SAC("MlpPolicy",env,verbose=1)
# # model = DQN("MultiInputPolicy",env=env,verbose=1,tensorboard_log='./logs',learning_rate=8e-3,)
# model = SAC.load("./savedModels/HumanoidStandup-v4_500k")


# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")



# env = gym.make('HumanoidStandup-v4', render_mode="human")
# model = SAC("MlpPolicy", env=env, verbose=1)
# model = SAC.load("HumanoidStandup-v4_3M")

# model.set_env(env)
# model.learn(5000)
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(2000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

####################################################
# The following code should run in                 #
# stable-baselines[mpi]==2.10.0 and TensorFlow 1.x #
# create a virtual env called TF1 to do so         #
####################################################


# import time
# import numpy as np
# import matplotlib.pyplot as plt
# # %matplotlib inline

# import gym

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines.common import set_global_seeds
# from stable_baselines import PPO2

# from stable_baselines.common.evaluation import evaluate_policy

# from stable_baselines.common.cmd_util import make_vec_env


# def make_env(env_id, rank, seed=0):
#     """
#     Utility function for multiprocessed env.

#     :param env_id: (str) the environment ID
#     :param seed: (int) the inital seed for RNG
#     :param rank: (int) index of the subprocess
#     """
#     def _init():
#         env = gym.make(env_id)
#         # Important: use a different seed for each environment
#         env.seed(seed + rank)
#         return env
#     set_global_seeds(seed)
#     return _init

# env_id = 'CartPole-v1'
# # The different number of processes that will be used
# PROCESSES_TO_TEST = [1, 2, 4, 8, 16]
# NUM_EXPERIMENTS = 3 # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
# TRAIN_STEPS = 5000
# # Number of episodes for evaluation
# EVAL_EPS = 20
# ALGO = PPO2

# # We will create one environment to evaluate the agent on
# eval_env = gym.make(env_id)

# reward_averages = []
# reward_std = []
# training_times = []
# total_procs = 0
# for n_procs in PROCESSES_TO_TEST:
#     total_procs += n_procs
#     print('Running for n_procs = {}'.format(n_procs))
#     if n_procs == 1:
#         # if there is only one process, there is no need to use multiprocessing
#         train_env = DummyVecEnv([lambda: gym.make(env_id)])
#     else:
#         # Here we use the "spawn" method for launching the processes, more information is available in the doc
#         # This is equivalent to make_vec_env(env_id, n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='spawn'))
#         train_env = SubprocVecEnv([make_env(env_id, i+total_procs) for i in range(n_procs)], start_method='spawn')

#     rewards = []
#     times = []

#     for experiment in range(NUM_EXPERIMENTS):
#         # it is recommended to run several experiments due to variability in results
#         train_env.reset()
#         model = ALGO('MlpPolicy', train_env, verbose=0)
#         start = time.time()
#         model.learn(total_timesteps=TRAIN_STEPS)
#         times.append(time.time() - start)
#         mean_reward, _  = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
#         rewards.append(mean_reward)
#     # Important: when using subprocess, don't forget to close them
#     # otherwise, you may have memory issues when running a lot of experiments
#     train_env.close()
#     reward_averages.append(np.mean(rewards))
#     reward_std.append(np.std(rewards))
#     training_times.append(np.mean(times))

# training_steps_per_second = [TRAIN_STEPS / t for t in training_times]

# plt.figure(figsize=(9, 4))
# plt.subplots_adjust(wspace=0.5)
# plt.subplot(1, 2, 1)
# plt.errorbar(PROCESSES_TO_TEST, reward_averages, yerr=reward_std, capsize=2)
# plt.xlabel('Processes')
# plt.ylabel('Average return')
# plt.subplot(1, 2, 2)
# plt.bar(range(len(PROCESSES_TO_TEST)), training_steps_per_second)
# plt.xticks(range(len(PROCESSES_TO_TEST)), PROCESSES_TO_TEST)
# plt.xlabel('Processes')
# _ = plt.ylabel('Training steps per second')