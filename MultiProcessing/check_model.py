# import sys
# sys.path.append('~/Owen/MultiTrafficLight/MultiTrafficLight/Simulation')


from Game import Game
from Actor import Actor
from Vehicle import Vehicle
from rewards import RewardMap
from TrafficLight import TrafficLight
from envs.straightRoad import StraightRoadEnv
from envs.SimpleEnvs import SingleTrafficLightEnvMultiProc, TwoTrafficLightEnvMultiProc

import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
import math
from datetime import date
from datetime import datetime
from datetime import timedelta

import gymnasium as gym

from stable_baselines3 import SAC, TD3, A2C, DDPG, PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

# import carla
# from carla import Actor
# from carla import Vector3D
# from carla import Transform, Location, Rotation


from stable_baselines3.common.env_checker import check_env
### Currently the simulation runs in 1-D space/x-axis
### CONSTANTS
# speed_limit = 60km/h


# hyper parameter
# frame = 0
# INITIAL_REWARD = 0
# MAP_SIZE = 1000 # m
# DESTINATION = MAP_SIZE
# HZ = 10
# DELTA_T = 1/HZ
# NUMBR_OF_LIGHTS = 16	


def checkModel_MultiPPO():
	# stl_vec_env = make_vec_env("SingleTrafficLightMultiProc-v1", 1024)
	env = gym.make("SeventeenTrafficLights")
	# model = PPO(
	# 	"MlpPolicy", 
	# 	env=env, 
	# 	batch_size=1024,
	# 	learning_rate=3e-5, 
	# 	# action_noise=NormalActionNoise(mean=np.zeros(vec_env_train.action_space.shape[-1]), 
	# 	tensorboard_log='./tb_log',
	# 	verbose=1, 
	# 	device='cuda'
	# )
	model = PPO.load(f"./models/0925/PPO_SeventeenTrafficLights_2048_3e-5_deltat_0.1_{140}e8[-2,2]")
	eposides = 1

	file_name = f"./check_result_log/0915_PPO_SeventeenTrafficLights_2048_3e-5_deltat_0.1_{140}e8[-2,2]"
	f = open(file_name, "a")
	f.write("step, action, location, speed, observation\n")
	data = ''
	for ep in range(eposides):
		obs = env.reset()[0]
		done = False
		rewards = 0
		step = 0
		while not done:
			step += 1 
			action, _states = model.predict(obs, deterministic=True)
			obs, reward, done, info, t = env.step(action)

			data += str(step) + ",\t" + str(float(f'{action[0]:.6f}')) + ",\t" \
					+ str(float(f'{obs[0]:.6f}')) + "," + str(float(f'{obs[1]:.6f}')) + "," \
					+ str([float(f'{i:.6f}') for i in obs]) + "\n"
			print(data)
			rewards += reward
	f.write(data)
	f.close()
checkModel_MultiPPO()