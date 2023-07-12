import gymnasium as gym

from stable_baselines3 import SAC
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from typing import Callable
# from stable_baselines3.common.utils import set_random_seed



def trainHumanoid(i: int):
    # env = gym.make('HumanoidStandup-v4', render_mode="human")
    env_id = "HumanoidStandup-v4"
    num_process = 256
    vec_env_train = make_vec_env(env_id, n_envs=num_process)
    model = SAC(
        "MlpPolicy", 
        vec_env_train, 
        batch_size=1024,
        tau=0.005,
        gamma=0.9,
        optimize_memory_usage=False,
        learning_rate=0.001, 
        action_noise=NormalActionNoise,
        tensorboard_log="./logs",
        verbose=1, 
        device='cpu'
        )
    # model = SAC("MlpPolicy", vec_env_train, learning_rate=linear_schedule(0.01), verbose=1, device='cpu')
    model = SAC.load(f"./savedModels/HumanoidStandup-v4_{i}M", vec_env_train)    
    model.learn(1000_000, progress_bar=True)
    model.save(f"./savedModels/HumanoidStandup-v4_{i+1}M")
    vec_env_train.close()
    

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



for it in range(30, 90):
    trainHumanoid(it)



env_id = "HumanoidStandup-v4"
num_process = 32
vec_env_train = make_vec_env(env_id, n_envs=num_process)
model = SAC("MlpPolicy", vec_env_train, verbose=0)
model = SAC.load(f"./savedModels/HumanoidStandup-v4_22M", vec_env_train)
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")


# tensorboard --logdir ./logs