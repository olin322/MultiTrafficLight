import gymnasium as gym

from stable_baselines3 import SAC
# from gymnasium.envs.registration import register



def trainHumanoid(i: int):
    env = gym.make('HumanoidStandup-v4', render_mode="human")
    load_model_name = "../savedModels/HumanoidStandup-v4_" + str(i+1) + "M"
    model = SAC.load(load_model_name, env=env)
    model.set_env(env)
    model.learn(1_000, progress_bar=True)
    model_name = "HumanoidStandup-v4_" + str(i+2) + "M"
    model.save(model_name)
    
# for i in range(7):
#     trainHumanoid(i)

env = gym.make("HumanoidStandup-v4", render_mode="human")
model = SAC.load("../savedModels/HumanoidStandup-v4_4M")
model.set_env(env)
vec_env = model.get_env()
obs = vec_env.reset()


for i in range(2000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
