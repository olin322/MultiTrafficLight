###############################
# The following code runs in  #
# stable-baselines3[extra]    #
###############################


import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from typing import Callable
# from stable_baselines3.common.utils import set_random_seed

def trainHumanoid(it: int,t: float, g: float, a:float):
    env_id = "HumanoidStandup-v4"
    num_process = 4096
    vec_env_train = make_vec_env(env_id, n_envs=num_process)
    model = SAC(
        "MlpPolicy", 
        env=vec_env_train, 
        batch_size=2048,
        tau=t,
        gamma=g,
        optimize_memory_usage=False,
        learning_rate=a, 
        action_noise=NormalActionNoise(mean=np.zeros(vec_env_train.action_space.shape[-1]), 
            sigma=0.1*np.ones(vec_env_train.action_space.shape[-1])),
        tensorboard_log=None,
        verbose=1, 
        device='cuda'
        )
    model = SAC.load(f"./savedModels/second/HumanoidStandup-v4_tau{model.tau}"+\
            f"gamma{model.gamma}alpha{model.learning_rate}_{it}M", vec_env_train)    
    # model.set_env(vec_env_train)
    model.learn(5000_000, progress_bar=True)
    trained = f"./savedModels/second/HumanoidStandup-v4_tau{model.tau}gamma"+\
                f"{model.gamma}alpha{model.learning_rate}_{it+5}M"
    model.save(trained)
    print("saved model:\t", trained.split('/')[-1])


def demo(i: int): # i-th saved model 
    # env_id = "HumanoidStandup-v4"
    # num_process = 1
    # vec_env_demo = make_vec_env(env_id, n_envs=num_process)
    env_demo = gym.make('HumanoidStandup-v4', render_mode="human")
    model = SAC(
        "MlpPolicy", 
        env_demo, 
        batch_size=1024,
        tau=0.01,
        gamma=0.9,
        optimize_memory_usage=False,
        learning_rate=0.001, 
        action_noise=None,
        tensorboard_log="./logs",
        verbose=1, 
        device='cpu'
        )
    model = SAC.load(f"./savedModels/second/HumanoidStandup-v4_tau{model.tau}"+\
            f"gamma{model.gamma}alpha{model.learning_rate}_{i}M", env_demo)
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
    vec_env.close()


# for it in range(640,700, 5):
#     trainHumanoid(it, t=0.01, g=0.9, a=0.001)

demo(470)






import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3 import SAC
from typing import Callable
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

# from gymnasium.envs.registration import register

def trainHumanoid(i: int):
    env = gym.make('HumanoidStandup-v4', render_mode="human")
    # tried multi-processing not working
    # env_id = "HumanoidStandup-v4"
    # num_process = 4  # Number of processes to use
    # # Create the vectorized environment
    # env = SubprocVecEnv([make_env(env_id, j) for j in range(num_process)])
    env_id = "HumanoidStandup-v4"
    num_process = 32
    vec_env_train = make_vec_env(env_id, n_envs=num_process)
    model = SAC("MlpPolicy", vec_env_train, verbose=0)
    model = SAC.load(f"./savedModels/HumanoidStandup-v4_{i}M", vec_env_train)
    # model.set_env(vec_env_train)
    model.learn(1000_000, progress_bar=True)
    model.save(f"./savedModels/HumanoidStandup-v4_{i+1}M_t")
    # load_model_name = f"./savedModels/HumanoidStandup-v4_{i}M"
    # model = SAC("MlpPolicy", env=env, learning_rate=linear_schedule(0.001), verbose=1)
    # model = SAC.load(load_model_name, device='cuda')
    # model.set_env(env)
    # model.learn(1000_000, progress_bar=True)
    # model_name = f"./savedModels/HumanoidStandup-v4_{i+1}M"
    # model.save(model_name)
    


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes
    current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func
# Initial learning rate of 0.001
# model = PPO("MlpPolicy", "CartPole-v1", learning_rate=linear_schedule(0.001), verbose=1)


def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:

    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

#########################################################################################

def demo(i: int): # i-th saved model 
    env_id = "HumanoidStandup-v4"
    num_process = 1
    vec_env_demo = make_vec_env(env_id, n_envs=num_process)
    model = SAC(
        "MlpPolicy", 
        vec_env_demo, 
        batch_size=1024,
        tau=0.01,
        gamma=0.9,
        optimize_memory_usage=False,
        learning_rate=0.001, 
        action_noise=None,
        tensorboard_log="./logs",
        verbose=1, 
        device='cpu'
        )
    model = SAC.load(f"./savedModels/second/HumanoidStandup-v4_tau{model.tau}gamma{model.gamma}alpha{model.learning_rate}_{i}M", vec_env_demo)
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(2000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
    vec_env_demo.close()

# demo(120)

#########################################################################################
# for it in range(22, 23):
#     trainHumanoid(it)

# env_id = "HumanoidStandup-v4"
# num_process = 16
# vec_env_train = make_vec_env(env_id, n_envs=num_process)
# model = SAC("MlpPolicy", vec_env_train, verbose=0)
# model = SAC.load("./savedModels/HumanoidStandup-v4_13M", vec_env_train)
# # model.set_env(vec_env_train)
# model.learn(1000_000, progress_bar=True)
# model.save("./savedModels/HumanoidStandup-v4_14M")

### uncomment the following code to check result

# env = gym.make('HumanoidStandup-v4', render_mode="human")
# load_model_name = f"./savedModels/HumanoidStandup-v4_19M"
# model = SAC("MlpPolicy", env=env, learning_rate=linear_schedule(0.001), verbose=1)
# model = SAC.load(load_model_name)
# model.set_env(env)
# # model.learn(1000_000, progress_bar=True)
# # # model_name = f"./savedModels/HumanoidStandup-v4_100M"
# # # model.save(model_name)
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")

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






#############################################################################################33

# from typing import Callable
# from stable_baselines3 import PPO

# def linear_schedule(initial_value: float) -> Callable[[float], float]:
#     """
#     Linear learning rate schedule.
#     :param initial_value: Initial learning rate.
#     :return: schedule that computes
#     current learning rate depending on remaining progress
#     """
#     def func(progress_remaining: float) -> float:
#         """
#         Progress will decrease from 1 (beginning) to 0.
#         :param progress_remaining:
#         :return: current learning rate
#         """
#         return progress_remaining * initial_value
#     return func
# # Initial learning rate of 0.001
# model = PPO("MlpPolicy", "CartPole-v1", learning_rate=linear_schedule(0.001), verbose=1)
# model.learn(total_timesteps=20_000)
# # By default, `reset_num_timesteps` is True, in which case the learning rate schedule resets.
# # progress_remaining = 1.0 - (num_timesteps / total_timesteps)
# model.learn(total_timesteps=10_000, reset_num_timesteps=True)
