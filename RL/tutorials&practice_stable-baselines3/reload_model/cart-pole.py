import gymnasium as gym

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
# import gym
from datetime import date
from datetime import datetime


model_name = "cart-pole16:46:52"

def train():
    global model_name
    loaded_model = A2C.load(model_name)
    new_env = gym.make('CartPole-v1')
    new_env = DummyVecEnv([lambda: new_env])
    loaded_model.set_env(new_env)
    loaded_model.learn(total_timesteps=10000)
    model_name = "cart-pole" + str(datetime.now().strftime("%H:%M:%S"))
    loaded_model.save(model_name)
    vec_env = loaded_model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = loaded_model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        # VecEnv resets automatically
        # if done:
        #   obs = vec_env.reset()



# if __name__ == "__main__":
# env = gym.make("CartPole-v1", render_mode="rgb_array")
# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000)
# model_name = "cart-pole" + str(datetime.now().strftime("%H:%M:%S"))
# model.save(model_name)
# for i in range(10):
#     train()

loaded_model = A2C.load("cart-pole16:50:54")
env = gym.make('CartPole-v1', render_mode='rgb_array')
loaded_model.set_env(env)
vec_env = loaded_model.get_env()
obs = vec_env.reset()   
for i in range(1000):
    action, _state = loaded_model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")


##########################################################
# the following example was given by LLM but not working #
# ValueError raised was likely caused by different envs  #
# but the code above tried to reload and train model     #
# in the same env kinda worked                           #
##########################################################


# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.env_util import make_vec_env
# # import gym

# # Create the CartPole-v1 environment
# env = gym.make('CartPole-v1')
# env = DummyVecEnv([lambda: env])

# # Create the PPO model and train on the CartPole-v1 environment
# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=10000)

# # Save the trained model
# model.save('my_model')

# # Load the trained model and continue training on the Acrobot-v1 environment
# loaded_model = PPO.load('my_model', policy_kwargs={'net_arch': [dict(pi=[64, 64], vf=[64, 64])]})
# new_env = make_vec_env('Acrobot-v1', n_envs=1, seed=0)
# loaded_model.set_env(new_env)
# loaded_model.learn(total_timesteps=10000)

# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")
#     # VecEnv resets automatically
#     # if done:
#     #   obs = vec_env.reset()