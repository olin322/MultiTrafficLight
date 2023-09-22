# import sys
# sys.path.append('~/Owen/MultiTrafficLight/MultiTrafficLight/Simulation')


from Game import Game
from Actor import Actor
from Vehicle import Vehicle
from rewards import RewardMap
from TrafficLight import TrafficLight
from envs.straightRoad import StraightRoadEnv

import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
import math
from datetime import date
from datetime import datetime
from datetime import timedelta

import gymnasium as gym

from stable_baselines3 import SAC, TD3, A2C, DDPG
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
	# global frame, num
	ego_vehicle = Vehicle("ego_vehicle", 0.0, 1500.0, 2, 2, DELTA_T)
	reward_map = RewardMap(ego_vehicle)
	# reward_map.updateMapInfo(200, DELTA_T)
	game = StraightRoadEnv(1, DELTA_T, reward_map,)
	game.spawn_vehicle(ego_vehicle)
	"""
	trafficLight = TrafficLight(ID:str, location:float, initialPhase:str, countDown:int, DELTA_T:float)
	"""
	trafficLight_1  = TrafficLight("1",  100,  "green", 10, game.get_delta_t())
	
	game.add_traffic_light(trafficLight_1)
	

	trafficLights = [
					trafficLight_1
					]
	# reward_map.setTrafficLights(trafficLights)
	reward_map.updateMapInfo(200, 0.02, trafficLights)
	# reward_map.updateMapInfo(MAP_SIZE, DELTA_T, trafficLights)
	# env = make_vec_env(lambda: game, n_envs=1)
	# env = VecNormalize(game, norm_obs=True, norm_reward=True, clip_obs=10.)
	# env1 = gym.make("StraightRoad-v1", totalTrafficLights=16, delta_t=DELTA_T, rewardMap=reward_map)
	# check_env(env1)
	# check_env(game)
	# vec_env_train = make_vec_env(env, n_envs=1024)
	# env = SubprocVecEnv([env for _ in range(1024)])
	env = game
	model = SAC(
			"MlpPolicy",
			# "MultiInputPolicy", 
			# env=vec_env_train, 
			env = env,
			batch_size=256, 
			verbose=1, 
			learning_rate=1e-6, 
			device='cuda')

	def _func(i: int):
		# model = SAC.load(f"./straightRoadModels/081123/StraightRoad-v1_256_1e-6_cuda_{i}e6")
		# model.set_env(gym.make("StraightRoad-v1", 16, DELTA_T, reward_map))
		# model.set_env(env)
		model.learn(1e6, progress_bar=True)
		"""
		naming convention: <ENV_NAME>_<BATCH_SIZE>_<LEARNING_RATE>_<EPISODES>
		"""
		model.save(f"./straightRoadModels/081123/StraightRoad-v1_256_1e-6_cuda_{i+1}e6[-2,2]_newreward")
		print(f"finished training StraightRoad-v1_256_1e-6_cuda_{i+1}e6[-2, 2]")
	for i in range(0, 1, 1):
		_func(i)

SingleTrafficLight()