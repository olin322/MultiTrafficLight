import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3 import SAC
from gymnasium.envs.registration import register

def trainHumanoid(i: int):
    env = gym.make('HumanoidStandup-v4', render_mode="human")
    load_model_name = f"./savedModels/HumanoidStandup-v4_{i}M"
    model = SAC.load(load_model_name)
    model.set_env(env)
    model.learn(500_000, progress_bar=True)
    model_name = f"./savedModels/HumanoidStandup-v4_4M500k"
    model.save(model_name)
    
trainHumanoid(4)
