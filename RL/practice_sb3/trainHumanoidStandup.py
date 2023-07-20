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
        batch_size=4096,
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

# def trainHumanoid_fast(it: int,t: float, g: float, a:float):
    
#     def _func(i: int, t: float, g: float, a: float, it: int):
#         # env = gym.make('HumanoidStandup-v4', render_mode="human")
#         env_id = "HumanoidStandup-v4"
#         num_process = 1024
#         vec_env_train = make_vec_env(env_id, n_envs=num_process)
#         model = SAC(
#             "MlpPolicy", 
#             vec_env_train, 
#             # learning_rate = 3e-3,
#             batch_size=1024,
#             learning_starts=1000,
#             tau=t,
#             gamma=g,
#             optimize_memory_usage=False,
#             learning_rate=a, 
#             action_noise=NormalActionNoise,
#             tensorboard_log=None,
#             verbose=1, 
#             device='cpu'
#             )
#         # model = SAC("MlpPolicy", vec_env_train, learning_rate=linear_schedule(0.01), verbose=1, device='cpu')
#         pre = f"./savedModels/temp/HumanoidStandup-v4_tau{model.tau}gamma\
#                 {model.gamma}alpha{model.learning_rate}_{i}"
#         model = SAC.load(pre, vec_env_train)    
#         model.learn(50_000, progress_bar=True)
#         pre = pre = f"./savedModels/temp/HumanoidStandup-v4_tau{model.tau}gamma\
#                     {model.gamma}alpha{model.learning_rate}_{i+1}"
#         model.save(pre)
#         if (i % 19 == 0):
#             model.save(f"./savedModels/second/HumanoidStandup-v4_tau{model.tau}\
#                         gamma{model.gamma}alpha{model.learning_rate}_{it+1}M")
#             print("saved model:\t", pre)
        
#         vec_env_train.close()

#     env_id = "HumanoidStandup-v4"
#     num_process = 1024
#     vec_env_train = make_vec_env(env_id, n_envs=num_process)
#     model = SAC(
#         "MlpPolicy", 
#         vec_env_train, 
#         batch_size=1024,
#         tau=t,
#         gamma=g,
#         optimize_memory_usage=False,
#         learning_rate=a, 
#         action_noise=NormalActionNoise,
#         verbose=1, 
#         device='cpu'
#         )
#     # model = SAC("MlpPolicy", vec_env_train, learning_rate=linear_schedule(0.01), verbose=1, device='cpu')
#     model = SAC.load(f"./savedModels/second/HumanoidStandup-v4_tau{model.tau}"+\
#             f"gamma{model.gamma}alpha{model.learning_rate}_{it}M", vec_env_train)    
#     model.learn(50_000, progress_bar=True)
#     pre = f"./savedModels/temp/HumanoidStandup-v4_tau{model.tau}"+\
#             f"gamma{model.gamma}alpha{model.learning_rate}_{1}"
#     model.save(pre)
#     print("saved model:\t", pre)
#     for i in range(1, 20):
#         _func(i, t, g, a, it)

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


for it in range(925, 995, 5):
    trainHumanoid(it, t=0.01, g=0.9, a=0.001)

# demo(900)


###############################################################################

# env_id = "HumanoidStandup-v4"
# num_process = 1
# vec_env_demo = make_vec_env(env_id, n_envs=num_process)
# model = SAC(
#     "MlpPolicy", 
#     vec_env_demo, 
#     batch_size=1024,
#     tau=0.01,
#     gamma=0.9,
#     optimize_memory_usage=False,
#     learning_rate=0.001, 
#     action_noise=None,
#     tensorboard_log="./logs",
#     verbose=1, 
#     device='cpu'
#     )
# model = SAC.load(f"./savedModels/second/HumanoidStandup-v4_tau{model.tau}gamma{model.gamma}alpha{model.learning_rate}_{50}M", vec_env_demo)
# model.learn(1000_000, progress_bar=True)
# model_name=f"./savedModels/second/HumanoidStandup-v4_tau{model.tau}gamma{model.gamma}alpha{model.learning_rate}_M"
# model.save(model_name)
# vec_env = make_vec_env(env_id, 16)
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")
# vec_env_demo.close()

# tensorboard --logdir ./logs