# import sys
# sys.path.append('~/Owen/MultiTrafficLight/MultiTrafficLight/Simulation')


from Game import Game
from Actor import Actor
from Vehicle import Vehicle
from rewards import RewardMap
from TrafficLight import TrafficLight
from envs.SimpleEnvs import SingleTrafficLightEnvMultiProc, TwoTrafficLightEnvMultiProc

import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
import math
from datetime import date
from datetime import datetime
from datetime import timedelta
from typing import Callable

import gymnasium as gym

from stable_baselines3 import SAC, TD3, A2C, DDPG, PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback

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

# TO-DO
# 1. implement seed for random generator so experiment can be replicated
# 2. try multi-pro cessing # checkout conventions need to follow

def someFunc():
	pass
	return None

def nextThreeLights():
	return None

# # Callback
# eval_callback = EvalCallback(env, best_model_save_path="./logs/BestModel0419_02/",
#                              log_path="./logs/BestModel0419_02/", eval_freq=500,
#                              deterministic=True, render=False)
#
# # Train the agent and display a progress bar
# model.learn(
#     total_timesteps=int(7e6),
#     tb_log_name='Sumo_pattern1_straight_DQN_alpha_7e-3_7M_call_1',
#     progress_bar=True,
#     callback=eval_callback
# )
#
# # Save the agent
# model.save("Sumo_pattern1_straight_DQN_alpha_7e-3_7M_call_1")
# del model  # delete trained model to demonstrate loading
#
# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("Sumo_pattern1_straight_DQN_alpha_7e-3_7M_call_1", env=env)


def seventeenTrafficLightsSettings(env_id, load_model_name, trained_model_name, training_iterations):
	stl_vec_env = make_vec_env(env_id, 2048)
	model = PPO(
		"MlpPolicy", 
		env=stl_vec_env, 
		batch_size=2048,
		learning_rate= 3e-5, 
		verbose=1, 
		device='cuda'
		)
	
	for i in range(0, 1):
		model = PPO.load(load_model_name)
		model.set_env(stl_vec_env)
		model.tensorboard_log = './tb_log/1020'
		model.learn(training_iterations, progress_bar=True)#, callback=eval_callback)
		model.save(trained_model_name)
		print("Finished Training:", trained_model_name)
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		print(current_time)



def seventeenTrafficLights(load_model_name, trained_model_name, training_iterations):
	stl_vec_env = make_vec_env("SeventeenTrafficLights", 2048)
	model = PPO(
		"MlpPolicy", 
		env=stl_vec_env, 
		batch_size=2048,
		learning_rate= 3e-5, 
		# action_noise=NormalActionNoise(mean=np.zeros(vec_env_train.action_space.shape[-1]), 
		# tensorboard_log='./tb_log/1020',
		verbose=1, 
		device='cuda'
		)
	# eval_callback = EvalCallback(stl_vec_env, best_model_save_path="./models/0911/best_models/",
    #                          log_path="./models/0911/best_models_log/", eval_freq=1e5,
    #                          deterministic=True, render=False)
	for i in range(0, 1):
		# model_name = f"./models/PPO_SeventeenTrafficLights_2048_3e-5_deltat_0.1_{200}e8[-2,2]"
		model = PPO.load(load_model_name)
		# model = PPO.load("./models/0828/"+model_name, custom_objects={'learning_rate':7.77e-7})
		# print("loaded", model_name)
		model.set_env(stl_vec_env)
		model.tensorboard_log = './tb_log/1020'
		model.learn(training_iterations, progress_bar=True)#, callback=eval_callback)
		model.save(trained_model_name)
		print("Finished Training:", trained_model_name)
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		print(current_time)

def threeTrafficLights():
	stl_vec_env = make_vec_env("ThreeTrafficLights", 1024)
	model = PPO(
		"MlpPolicy", 
		env=stl_vec_env, 
		batch_size=1024,
		learning_rate= 3e-5, 
		# action_noise=NormalActionNoise(mean=np.zeros(vec_env_train.action_space.shape[-1]), 
		# tensorboard_log='./tb_log/0901',
		verbose=1, 
		device='cuda'
		)
	for i in range(30, 32, 2):
		model_name = f"PPO_TwoTrafficLights_1024_3e-6_deltat_0.1_27e8[-2,2]"
		model = PPO.load("./models/0828/" + model_name)
		# model = PPO.load("./models/0828/"+model_name, custom_objects={'learning_rate':7.77e-7})
		print("loaded", model_name)
		model.set_env(stl_vec_env)
		model.learn(1e8, progress_bar=True)
		trained = f"./models/0901/PPO_ThreeTrafficLights_2048_3e-5_deltat_0.1_{1}e8[-2,2]"
		model.save(trained)
		print("Finished Training:", trained)
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		print(current_time)


def twoTrafficLights():	
	stl_vec_env = make_vec_env("TwoTrafficLightMultiProc-v1", 1024)
	model = PPO(
		"MlpPolicy", 
		env=stl_vec_env, 
		batch_size=1024,
		learning_rate= 3e-6, 
		# action_noise=NormalActionNoise(mean=np.zeros(vec_env_train.action_space.shape[-1]), 
		tensorboard_log='./tb_log/0825',
		verbose=1, 
		device='cuda'
		)
	for i in range(30, 32, 2):
		model_name = f"PPO_TwoTrafficLights_1024_3e-6_deltat_0.1_{27}e8[-2,2]"
		model = PPO.load("./models/0828/" + model_name)
		# model = PPO.load("./models/0828/"+model_name, custom_objects={'learning_rate':7.77e-7})
		# model.set_parameters({"param_groups":[
		# 	{	
		# 		'lr': 7.77e-9, 
		# 		'betas': (0.9, 0.999), 
		# 		'eps': 1e-05, 
		# 		'weight_decay': 0, 
		# 		'amsgrad': False, 
		# 		'maximize': False, 
		# 		'foreach': None, 
		# 		'capturable': False, 
		# 		'differentiable': False, 
		# 		'fused': None, 
		# 		'params': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
		# 	}]}, True, "cuda")
		# print(model.get_parameters())
		print("loaded", model_name)
		model.set_env(stl_vec_env)
		model.learn(13e8, progress_bar=True)
		trained = f"./models/0831/PPO_TwoTrafficLights_1024_3e-6_deltat_0.1_{40}e8[-2,2]"
		model.save(trained)
		print("Finished Training:", trained)
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		print(current_time)

def oneTrafficLight():	
	stl_vec_env = make_vec_env("SingleTrafficLightMultiProc-v1", 1024)
	model = PPO(
		"MlpPolicy", 
		env=stl_vec_env, 
		batch_size=1024,
		learning_rate=3e-6, 
		# action_noise=NormalActionNoise(mean=np.zeros(vec_env_train.action_space.shape[-1]), 
		tensorboard_log='./tb_log/0825',
		verbose=1, 
		device='cuda'
		)
	for i in range(0, 1):
		# model = PPO.load(f"./models/0825/PPO_MultiProcSingleTrafficLight-v1_1024_3e-6_cuda_3.3e9[-2,2]_0917")
		model.set_env(stl_vec_env)
		model.learn(1e8, progress_bar=True)
		trained = f"./models/0825/PPO_MultiProcSingleTrafficLight-v1_1024_3e-6_deltat_0.1_1e8[-2,2]_1505"
		model.save(trained)
		print("Finished Training:", trained)
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		print(current_time)


def linear_decreasing_learning_rate(initial_learning_rate: float, \
					target_learning_rate: float) -> Callable[[float], float]:
	def _func(progress_remaining: float) -> float:
		return (initial_learning_rate - target_learning_rate) \
					* progress_remaining + target_learning_rate
	return _func

def piecewise_learning_rate(initial_learning_rate: float) -> Callable[[float], float]:
	def _func(progress_remaining: float) -> float:
		if (progress_remaining > 0.5):
			return initial_learning_rate
		elif (progress_remaining > 0.2):
			return initial_learning_rate * 0.1
		else:
			return initial_learning_rate * 0.01
	return _func

	


"""
The following does not work

# Get the parameters that where saved with the model
params = model.get_parameters()

# Print the initial learning rate
print("INITIAL LR: {}".format(params['policy.optimizer']['param_groups'][0]['lr']))

# Change the learning rate
params['policy.optimizer']['param_groups'][0]['lr'] = 0.000005

# Set the parameters on the model
model.set_parameters(params, exact_match=True)

new_params = model.get_parameters()
# Print the initial learning rate
print("NEW LR: {}".format(new_params['policy.optimizer']['param_groups'][0]['lr']))

# Start training
model.learn(total_timesteps=1000)
"""




def _notes():
	# global frame, num
	ego_vehicle = Vehicle("ego_vehicle", 0.0, 1500.0, 2, 2, DELTA_T)
	reward_map = RewardMap(ego_vehicle)
	game = StraightRoadEnv(16, DELTA_T, reward_map)
	game.spawn_vehicle(ego_vehicle)
	"""
	trafficLight = TrafficLight(ID:str, location:float, initialPhase:str, countDown:int, DELTA_T:float)
	"""
	trafficLight_1  = TrafficLight("1",  100,  "green", 10, game.get_delta_t())
	# trafficLight_2  = TrafficLight("2",  200,  "green", 47, game.get_delta_t())
	# trafficLight_3  = TrafficLight("3",  500,  "green", 61, game.get_delta_t())
	# trafficLight_4  = TrafficLight("4",  2000, "green", 53, game.get_delta_t())
	# trafficLight_5  = TrafficLight("5",  2500, "green", 53, game.get_delta_t())
	# trafficLight_6  = TrafficLight("6",  3200, "green", 61, game.get_delta_t())
	# trafficLight_7  = TrafficLight("7",  3400, "green", 67, game.get_delta_t())
	# trafficLight_8  = TrafficLight("8",  3600, "green", 67, game.get_delta_t())
	# trafficLight_9  = TrafficLight("9",  3800, "green", 67, game.get_delta_t())
	# trafficLight_10 = TrafficLight("10", 4000, "green", 57, game.get_delta_t())
	# trafficLight_11 = TrafficLight("11", 5000, "green", 57, game.get_delta_t())
	# trafficLight_12 = TrafficLight("12", 5100, "green", 67, game.get_delta_t())
	# trafficLight_13 = TrafficLight("13", 6000, "green", 61, game.get_delta_t())
	# trafficLight_14 = TrafficLight("14", 7000, "green", 61, game.get_delta_t())
	# trafficLight_15 = TrafficLight("15", 8000, "green", 61, game.get_delta_t())
	# trafficLight_16 = TrafficLight("16", 9900, "green", 61, game.get_delta_t())
	game.add_traffic_light(trafficLight_1)
	# game.add_traffic_light(trafficLight_2)
	# game.add_traffic_light(trafficLight_3)
	# game.add_traffic_light(trafficLight_4)
	# game.add_traffic_light(trafficLight_5)
	# game.add_traffic_light(trafficLight_6)
	# game.add_traffic_light(trafficLight_7)
	# game.add_traffic_light(trafficLight_8)
	# game.add_traffic_light(trafficLight_9)
	# game.add_traffic_light(trafficLight_10)
	# game.add_traffic_light(trafficLight_11)
	# game.add_traffic_light(trafficLight_12)
	# game.add_traffic_light(trafficLight_13)
	# game.add_traffic_light(trafficLight_14)
	# game.add_traffic_light(trafficLight_15)
	# game.add_traffic_light(trafficLight_16)

	trafficLights = [
					trafficLight_1, 
					# trafficLight_2, trafficLight_3,trafficLight_4,
					# trafficLight_5, trafficLight_6, trafficLight_7,trafficLight_8,
					# trafficLight_9, trafficLight_10, trafficLight_11,trafficLight_12,
					# trafficLight_13, trafficLight_14, trafficLight_15,trafficLight_16,
					]
	reward_map.setTrafficLights(trafficLights)
	# reward_map.updateMapInfo(MAP_SIZE, DELTA_T, trafficLights)
	# env = make_vec_env(lambda: game, n_envs=1)
	# env = VecNormalize(game, norm_obs=True, norm_reward=True, clip_obs=10.)
	# env1 = gym.make("StraightRoad-v1", totalTrafficLights=16, delta_t=DELTA_T, rewardMap=reward_map)
	# check_env(env1)
	# check_env(game)
	# vec_env_train = make_vec_env(env, n_envs=1024)
	# env = SubprocVecEnv([env for _ in range(1024)])
	def __func():
		T = 1000  
		time = torch.arange(1, T + 1, dtype=torch.float32)
		x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
		d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
		tau = 4
		features = torch.zeros((T - tau, tau))
		for i in range(tau):
		    features[:, i] = x[i: T - tau + i]
		labels = x[tau:].reshape((-1, 1))

		batch_size, n_train = 16, 600
		# 只有前n_train个样本用于训练
		train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
		                            batch_size, is_train=True)
	

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
		model = SAC.load(f"./straightRoadModels/081023/StraightRoad-v1_256_1e-6_cuda_{i}e6[-2,2]_newreward")
		print(f"loaded StraightRoad-v1_256_1e-6_cuda_{i}e6[-2,2]_newreward")
		# model.set_env(gym.make("StraightRoad-v1", 16, DELTA_T, reward_map))
		model.set_env(env)
		model.learn(1e6, progress_bar=True)
		"""
		naming convention: <ENV_NAME>_<BATCH_SIZE>_<LEARNING_RATE>_<EPISODES>
		"""
		model.save(f"./straightRoadModels/081023/StraightRoad-v1_256_1e-6_cuda_{i+1}e6[-2,2]_newreward")
		print(f"finished training StraightRoad-v1_256_1e-6_cuda_{i+1}e6[-2, 2]")
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		print(current_time)
	for i in range(12, 14, 1):
		_func(i)

