# import sys
# sys.path.append('~/Owen/MultiTrafficLight/MultiTrafficLight/Simulation')
import subprocess

from Game import Game
from Actor import Actor
from Vehicle import Vehicle
from rewards import RewardMap
from TrafficLight import TrafficLight
from envs.SingleTrafficLightEnv import SingleTrafficLightEnv

import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
import math
from datetime import date
from datetime import datetime
from datetime import timedelta

import gymnasium as gym

from stable_baselines3 import SAC, TD3, A2C, PPO
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
INITIAL_REWARD = 0
MAP_SIZE = 1000 # m
DESTINATION = MAP_SIZE
HZ = 50
DELTA_T = 1/HZ
NUMBR_OF_LIGHTS = 16
# num = 1

# game = game(DELTA_T)
# reward_map = RewardMap(MAP_SIZE, INITIAL_REWARD)
# ego_vehicle = Vehicle("ego_vehicle", 0.0, 1500.0, 2, 2, game.get_delta_t())





# model = SAC()
# model.learn()

# TO-DO
# 1. implement seed for random generator so experiment can be replicated
# 2. try multi-pro cessing # checkout conventions need to follow
def rl_vec_straighRoad(seed: int):
	vec_envs = gym.make("StraightRoad-v1", number_of_lights, DELTA_T, reward_map)
	game.spawn_vehicle(ego_vehicle)
	lights = creatTrafficLightList(number_of_lights=NUMBR_OF_LIGHTS, 
									min_distance=100,
									max_distance=500,
									min_countDown=30,
									max_countDown=180)
	for light in lights:
		game.add_traffic_light(light)

def SingleTrafficLight():	
	stl = gym.make("SingleTrafficLight-v1")
	
	# trafficLight_1  = TrafficLight("1",  100,  "green", 10, game.get_delta_t())
	
	# game.add_traffic_light(trafficLight_1)
	

	# trafficLights = [
	# 				trafficLight_1
	# 				]
	
	# reward_map.updateMapInfo(200, 0.02, trafficLights)
	
	# env = VecNormalize(game, norm_obs=True, norm_reward=True, clip_obs=10.)
	# env1 = gym.make("StraightRoad-v1", totalTrafficLights=16, delta_t=DELTA_T, rewardMap=reward_map)
	# check_env(env1)
	# check_env(game)
	# vec_env_train = make_vec_env(env, n_envs=1024)
	# env = SubprocVecEnv([env for _ in range(1024)])
	# env = game
	model = SAC(
			"MlpPolicy",
			# "MultiInputPolicy", 
			# env=vec_env_train, 
			env = stl,
			batch_size=256, 
			verbose=1, 
			learning_rate=5e-7, 
			device='cuda')

	def _func(i: int):
		# model = SAC.load(f"./081423/SingleTrafficLight-v1_256_1e-6_cuda_{i}e6[-2,2]_newreward")
		# model.set_env(gym.make("StraightRoad-v1", 16, DELTA_T, reward_map))
		# model.set_env(env)
		model.learn(1e6, progress_bar=True)
		"""
		naming convention: <ENV_NAME>_<BATCH_SIZE>_<LEARNING_RATE>_<EPISODES>
		"""
		model.save(f"./081423/SingleTrafficLight-v1_256_5e-7_cuda_{i+1}e6[-2,2]_newreward")
		print(f"finished training SingleTrafficLight-v1_256_5e-7_cuda-v1_256_1e-6_cuda_{i+1}e6[-2, 2]")
	
	for i in range(0, 1, 1):
		_func(i)
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	print(current_time)

# SingleTrafficLight()

def SingleTrafficLightMultiEnv():
	stl_vec_env = make_vec_env("SingleTrafficLightMultiProc-v1", 1024)
	model = PPO(
		"MlpPolicy", 
		env=stl_vec_env, 
		batch_size=1024,
		learning_rate=3e-7, 
		# action_noise=NormalActionNoise(mean=np.zeros(vec_env_train.action_space.shape[-1]), 
		tensorboard_log='./tb_log/0821',
		verbose=1, 
		device='cuda'
		)
	for i in range(0, 1):
		model = PPO.load(f"./models/0824/PPO_MultiProcSingleTrafficLight-v1_1024_3e-7_cuda_3.3e9[-2,2]_0917")
		model.set_env(stl_vec_env)
		model.learn(1.2e9, progress_bar=True)
		trained = f"./models/0824/PPO_MultiProcSingleTrafficLight-v1_1024_3e-7_cuda_4.5e9[-2,2]_1630"
		model.save(trained)
		print("Finished Training:", trained)
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		print(current_time)

# subprocess.run(["tensorboard", "--logdir", "./tb_log"])
# tensorboard --logdir ./tb_log
SingleTrafficLightMultiEnv()