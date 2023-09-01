
from Game import Game
from Actor import Actor
from Vehicle import Vehicle
from TrafficLight import TrafficLight

from math import floor, ceil
from typing import List

class RewardMap:
	
	def __init__(self, 
				mapSize: int,
				delta_t: float,
				ego_vehicle: Vehicle,
				trafficLights: List[TrafficLight],
				accumulatedReward= 1, 
				):
		# mapSize represents the length of the road
		# in this 1-dimension world
		self.accumulatedReward = accumulatedReward
		self.ego_vehicle = ego_vehicle
		# self.ego_vehicle_initial_state = ego_vehicle.getLocation()
		self.ego_vehicle_prev_location = ego_vehicle.getLocation()
		# default values needs to be updated
		self.ticks = 0
		self.mapSize = mapSize
		self.rewardMap = [1] * self.mapSize
		self.delta_t = delta_t
		self.trafficLights = trafficLights
		self.lightsPassed = 1
		self.nextLight = self._find_next_light()
		self.stepReward = 0
		self.initialState = {
							"accumulatedReward":self.accumulatedReward, 
							"ego_vehicle":self.ego_vehicle, 
							"mapSize":self.mapSize, 
							"rewardMap":self.rewardMap,
							"delta_t":self.delta_t, 
							"trafficLights":self.trafficLights
							}

	# serves as an additional constructor
	def updateMapInfo(self, 
					mapSize: int, 
					delta_t: float,
					trafficLights: list):
		self.mapSize = mapSize # technically should be a "Map" object
		self.delta_t = delta_t
		self.rewardMap = [1] * mapSize
		self.trafficLights = trafficLights
		self.nextLight = self._find_next_light()
		self.stepReward = 0
		self.initialState["mapSize"] = mapSize
		self.initialState["delta_t"] = delta_t
		self.initialState["trafficLights"] = trafficLights

	def tick(self, action) -> bool:
		self.ticks += 1
		terminated = self.calcReward(action)
		return terminated

	# need to add rewards for passing a traffic light?
	def calcReward(self, action) -> bool:
		"""
		reward coefficient
		1 - (time passed/total time needed)
		Can't get world.get_simultion_time, 
		reward_map is a parameter passed to World object,
		not super crucial, come back later
		"""
		# coef = 1 - (World.get_simulation_time() / ceil(self.mapSize / 16.67))
		coef = 1
		reward = 0
		"""
		reward consists of two parts:
		1. distance traveled during this step:
			1 reward for each integer point
		2. reward for passing a traffic light passed
			10 for first light, 20 for second light etc,.
		"""
		terminated = False
		# for i in range(floor(self.ego_vehicle_prev_location), 
		# 				min(ceil(self.ego_vehicle.getLocation()), self.mapSize)):
		# 	reward += self.rewardMap[i] * coef
		# 	self.rewardMap[i] = 0
		reward += (self.ego_vehicle.getLocation() - self.ego_vehicle_prev_location) * coef
		if (self.ego_vehicle.getLocation() >= (self.mapSize-2)):
			terminated = True
			reward += 1000
		# print("next light location", self.nextLight.getLocation())
		# if (not self.nextLight): print("!!!!!!!!!!!!!!!!!!!!")
		if (self.nextLight):
			if (self.ego_vehicle_prev_location < self.nextLight.getLocation()) & \
					(self.nextLight.getLocation() < self.ego_vehicle.getLocation()):
				if (self.nextLight.getPhase() == "red"):
					terminated = True
					reward -= 1000
					# print("passed red light")
				else:
					reward += 100 * self.lightsPassed
					if (self.nextLight.getPhase() == "green"):
						reward += 200
						# print("passed green light")
					elif (self.nextLight.getPhase() == "yellow"):
						reward -= 1000
						# print("passed yellow light")
					# if (self.ticks < 600):
					# 	reward += 100
					# elif (self.ticks < 1000):
					# 	reward += 100
					self.lightsPassed += 1
					self.nextLight = self._find_next_light()
		
		reward = reward - (self.delta_t * 2)
		# print(reward)	
		# if ((self.ticks % 50) == 0):
		# 	reward -= 1
		
		if (self.ego_vehicle.isAtSpeedLimit()):
			if (action > 0):
				reward -= action * 3
		else:
			if ((self.ego_vehicle.getSpeed() == 0) & (action < 0)):
				reward += action * 3

		# print(self.ego_vehicle_prev_location, self.ego_vehicle.getLocation())
		self.ego_vehicle_prev_location = self.ego_vehicle.getLocation()
		self.accumulatedReward += reward
		self.stepReward = reward
		return terminated

	def getStepReward(self) -> float:
		return self.stepReward

	def getAccumulatedReward(self) -> int:
		return self.accumulatedReward

	def reset(self, seed=None) -> None:
		self.accumulatedReward = self.initialState["accumulatedReward"]
		self.ego_vehicle = self.initialState["ego_vehicle"]
		self.ego_vehicle_prev_location = self.ego_vehicle.getLocation()
		self.ticks = 0
		self.mapSize = self.initialState["mapSize"]
		self.rewardMap = self.initialState["rewardMap"]	
		self.delta_t = self.initialState["delta_t"]
		self.trafficLights = self.initialState["trafficLights"]
		self.nextLight = self._find_next_light()
		self.lightsPassed = 1 	
		self.stepReward = 0
		return None

	def setMapSize():
		return None

	def setTrafficLights(self, trafficLights: List[TrafficLight]) -> List[TrafficLight]:
		self.trafficLights = trafficLights
		return self.trafficLights




############################## PRIVATE METHODS ################################

	def _find_next_light(self) -> TrafficLight:
		# print(self.trafficLights)
		nextLight = None
		if (not self.trafficLights):
			return None
		for light in self.trafficLights:
			if (light.getLocation() > self.ego_vehicle.getLocation()):
				if (not nextLight):
					nextLight = light
				else:
					if (nextLight.getLocation() < nextLight.getLocation()):
						nextLight = light
		return nextLight