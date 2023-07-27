# import sys
# sys.path.append('~/Owen/MultiTrafficLight/MultiTrafficLight/Simulation')


from World import World
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

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise



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

# world = World(DELTA_T)
# reward_map = RewardMap(MAP_SIZE, INITIAL_REWARD)
# ego_vehicle = Vehicle("ego_vehicle", 0.0, 1500.0, 2, 2, world.get_delta_t())





# model = SAC()
# model.learn()

# TO-DO
# 1. implement seed for random generator so experiment can be replicated
# 2. try multi-pro cessing
def rl_vec_straighRoad(seed: int):
	vec_envs = gym.make("StraightRoad-v1", number_of_lights, DELTA_T, reward_map)
	world.spawn_vehicle(ego_vehicle)
	lights = creatTrafficLightList(number_of_lights=NUMBR_OF_LIGHTS, 
									min_distance=100,
									max_distance=500,
									min_countDown=30,
									max_countDown=180)
	for light in lights:
		world.add_traffic_light(light)

def main():	
	# global frame, num
	ego_vehicle = Vehicle("ego_vehicle", 0.0, 1500.0, 2, 2, DELTA_T)
	reward_map = RewardMap(ego_vehicle)
	world = StraightRoadEnv(16, DELTA_T, reward_map)
	world.spawn_vehicle(ego_vehicle)
	"""
	trafficLight = TrafficLight(ID:str, location:float, initialPhase:str, countDown:int, DELTA_T:float)
	"""
	trafficLight_1  = TrafficLight("1",  100,  "green", 10, world.get_delta_t())
	trafficLight_2  = TrafficLight("2",  200,  "green", 47, world.get_delta_t())
	trafficLight_3  = TrafficLight("3",  500,  "green", 61, world.get_delta_t())
	trafficLight_4  = TrafficLight("4",  2000, "green", 53, world.get_delta_t())
	trafficLight_5  = TrafficLight("5",  2500, "green", 53, world.get_delta_t())
	trafficLight_6  = TrafficLight("6",  3200, "green", 61, world.get_delta_t())
	trafficLight_7  = TrafficLight("7",  3400, "green", 67, world.get_delta_t())
	trafficLight_8  = TrafficLight("8",  3600, "green", 67, world.get_delta_t())
	trafficLight_9  = TrafficLight("9",  3800, "green", 67, world.get_delta_t())
	trafficLight_10 = TrafficLight("10", 4000, "green", 57, world.get_delta_t())
	trafficLight_11 = TrafficLight("11", 5000, "green", 57, world.get_delta_t())
	trafficLight_12 = TrafficLight("12", 5100, "green", 67, world.get_delta_t())
	trafficLight_13 = TrafficLight("13", 6000, "green", 61, world.get_delta_t())
	trafficLight_14 = TrafficLight("14", 7000, "green", 61, world.get_delta_t())
	trafficLight_15 = TrafficLight("15", 8000, "green", 61, world.get_delta_t())
	trafficLight_16 = TrafficLight("16", 9900, "green", 61, world.get_delta_t())
	world.add_traffic_light(trafficLight_1)
	world.add_traffic_light(trafficLight_2)
	world.add_traffic_light(trafficLight_3)
	world.add_traffic_light(trafficLight_4)
	world.add_traffic_light(trafficLight_5)
	world.add_traffic_light(trafficLight_6)
	world.add_traffic_light(trafficLight_7)
	world.add_traffic_light(trafficLight_8)
	world.add_traffic_light(trafficLight_9)
	world.add_traffic_light(trafficLight_10)
	world.add_traffic_light(trafficLight_11)
	world.add_traffic_light(trafficLight_12)
	world.add_traffic_light(trafficLight_13)
	world.add_traffic_light(trafficLight_14)
	world.add_traffic_light(trafficLight_15)
	world.add_traffic_light(trafficLight_16)

	trafficLights = [
					trafficLight_1, trafficLight_2, trafficLight_3,trafficLight_4,
					trafficLight_5, trafficLight_6, trafficLight_7,trafficLight_8,
					trafficLight_9, trafficLight_10, trafficLight_11,trafficLight_12,
					trafficLight_13, trafficLight_14, trafficLight_15,trafficLight_16,
					]
	reward_map.updateMapInfo(MAP_SIZE, DELTA_T, trafficLights)
	# env = make_vec_env(lambda: world, n_envs=1)
	# env = VecNormalize(world, norm_obs=True, norm_reward=True, clip_obs=10.)
	model = SAC("MultiInputPolicy", 
			env=world, 
			batch_size=256, 
			verbose=1, 
			learning_rate=1e-5, 
			device='cuda')
	# model = SAC.load("./straightRoadModels/temps/StraightRoad-v1_256_1e-5_cuda_2e6")
	# model.set_env(gym.make("StraightRoad-v1", 16, DELTA_T, reward_map))
	# model.set_env(world)
	model.learn(1e5, progress_bar=True)
	"""
	naming convention: <ENV_NAME>_<BATCH_SIZE>_<LEARNING_RATE>_<EPISODES>
	"""
	model.save("./straightRoadModels/temps/StraightRoad-v1_256_1e-5_cuda_1e5")

	# debug
	# log_data = ""
	# log_name = get_debug_log_name()
	# while ((ego_vehicle.getLocation() >= 0) & 
	# 		(ego_vehicle.getLocation() < DESTINATION)):
	# 	print(world.getFrame())
	# 	# frame += 1
	# 	log_debug_data(log_name)
	# 	world.tick()

		# print("simulation time = ", world.get_simulation_time())

		# print("speed = ", ego_vehicle.getSpeed())
		# print("light count down: ", trafficLight_1.getCountdown(), " ", trafficLight_1.getPhase())

def creatTrafficLightList(
							number_of_lights: int, 
							min_distance: float, # 最小间距
							max_distance: float, # 最大间距
							min_countDown: int, 
							max_countDown: int) -> list:
	lights = []
	trafficLight_0  = TrafficLight("0", 0, "green", 10, DELTA_T)
	lights.append(trafficLight_0)
	phases["green", "red"]
	for i in range(number_of_lights):
		lights.append(
			TrafficLight(
				str(i + 1),
				lights[i+1].getLocation() + random.randint(min_distance, max_distance),
				phases[random.randint(0,1)],
				random.randint(min_countDown, max_countDown),
				DELTA_T
			)
		)
	return lights


def log_debug_data(log_name: str) -> None:
	log_data = get_debug_log_data()
	f = open(log_name, 'a')
	f.write(log_data)
	f.close
	return None

def get_debug_log_name() -> str:
	today = date.today()
	current_time = datetime.now().strftime("%H:%M:%S")
	return("./debug_log/"+str(today)+"-"+str(current_time)+".txt")

def get_debug_log_data() -> str:
	log_data = "frame = " + str(world.getFrame()) + "\t"\
		    + " sim time = " + _roundup(world.get_simulation_time(), 6) + "\t"\
		    + " ev speed = " + _roundup(ego_vehicle.getSpeed(), 6) + "\t"\
		    + " ev location = " + _roundup(ego_vehicle.getLocation(), 6) + "\t"\
		    
	if(world.find_next_light()):
	    log_data += " countdown = " + str(world.find_next_light().getCountdown())\
	    		+ " " + world.find_next_light().getPhase() \
	    		+ " \tnext light location: " + _roundup(world.find_next_light().getLocation(), 6) \
	    		+ " \treal world time stamp: " + str(datetime.utcnow() + timedelta(hours=8)) + "\n"
	else:
		log_data += " vehicle passed all traffic lights"\
				+ "\treal world time stamp: " + str(datetime.utcnow() + timedelta(hours=8))\
				+ "\n"	
	return log_data

def _roundup(d: float, l: int) -> str:
	s = str(round(d, l))
	i = 0
	if (s.find(".") != -1):
		i = s.find(".")
	else:
		return str(d)
	while((len(s) - i) <= l):
		s += '0'
	return s


###############################################################################
###############################################################################

def check_result():
	eposides = 5
	for ep in range(eposides):
	    obs = env.reset()
	    done = False
	    rewards = 0
	    step = 0
	    while not done:
	        step += 1 
	        action, _states = model.predict(obs, deterministic=True)
	        obs, reward, done, info = env.step(action)
	        env.render()
	        rewards += reward
	print(rewards)


main()










############################################################################



### Animation
# ydata = []
# xdata = []
# fig, ax = plt.subplots()

# def animate(i):
# 	xdata.append(random.randint(0,9))
# 	ydata.append(random.randint(0,9))
# 	x_min = min(xdata)
# 	x_max = max(xdata)
# 	y_min = min(ydata)
# 	y_max = max(ydata)
# 	# fig, ax = plt.subplots()
# 	ax.set_xlim(x_min - 1, x_max + 1)
# 	ax.set_ylim(y_min - 1, y_max + 1)
# 	line, = ax.plot(xdata, ydata, 'ro')
# 	line.set_ydata(ydata)
# 	return line,

# ani = animation.FuncAnimation(fig, animate, interval=1000, blit=True, save_count = 50)
# plt.show()


# fig, ax = plt.subplots()
# ax.set_xlim(0, 10) 
# ax.set_ylim(0, 2)

# x_data, y_data = [], []

# def animate(i): 
#     x_data.append(i) 
#     y_data.append(math.sin(math.pi*i)) 

#     line, = ax.plot(x_data, y_data, 'r') 
#     line.set_ydata(y_data)
#     return line,

# ani = animation.FuncAnimation(fig, animate, interval=1000, blit=True, save_count = 50)
# plt.show()

















##############################################################################

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# fig, ax = plt.subplots()

# x = np.arange(0, 2*np.pi, 0.01)
# line, = ax.plot(x, np.sin(x))

# def animate(i):
#     line.set_ydata(np.sin(x + i / 50))  # update the data.
#     return line,

# ani = animation.FuncAnimation(
#     fig, animate, interval=20, blit=True, save_count=50) 

# # To save the animation, use e.g.
# #
# # ani.save("movie.mp4")
# #
# # or
# #
# # writer = animation.FFMpegWriter(
# #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# # ani.save("movie.mp4", writer=writer)

# plt.show()









# import numpy as np
# import matplotlib.pyplot as plt 
# import matplotlib.animation as animation

# # Fixing random state for reproducibility
# np.random.seed(19680801)

# fig, ax = plt.subplots()
# x = np.arange(0, 2*np.pi, 0.01)        # x-array

# def animate(i): 
#     line.set_ydata(np.sin(x + i/10.0))  # update the data
#     return line, 

# # Init only required for blitting to give a clean slate.
# def init():  
#     line.set_ydata(np.sin(x)) 
#     return line,

# line, = ax.plot(x, np.sin(x))
# ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,  
#                               interval=25, blit=True)
# plt.show()