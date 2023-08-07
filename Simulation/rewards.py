
from World import World
from Actor import Actor
from Vehicle import Vehicle
from TrafficLight import TrafficLight

from math import floor, ceil
from typing import List

class RewardMap:
	
	def __init__(self, 
				ego_vehicle: Vehicle,
				accumulatedReward: int = 1, 
				):
		# mapSize represents the length of the road
		# in this 1-dimension world
		self.accumulatedReward = accumulatedReward
		self.ego_vehicle = ego_vehicle
		# self.ego_vehicle_initial_state = ego_vehicle.getLocation()
		self.ego_vehicle_prev_location = ego_vehicle.getLocation()
		# default values needs to be updated
		self.ticks = 0
		self.mapSize = 10000
		self.rewardMap = [1] * self.mapSize
		self.delta_t = 0.02
		self.trafficLights = []
		self.lightsPassed = 1
		self.nextLight = self._find_next_light()
		self.stepReward = 0
		self.initialState = [
							self.accumulatedReward, 
							self.ego_vehicle, 
							self.mapSize, 
							self.rewardMap,
							self.delta_t, 
							self.trafficLights
							]

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
		self.initialState = [
							self.accumulatedReward, 
							self.ego_vehicle, 
							self.mapSize, 
							self.rewardMap,
							self.delta_t, 
							self.trafficLights]

	def tick(self) -> bool:
		self.ticks += 1
		terminated = self.calcReward()
		return terminated

	# need to add rewards for passing a traffic light?
	def calcReward(self) -> bool:
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
			100 for first light, 200 for second light etc,.
		"""
		terminated = False
		for i in range(floor(self.ego_vehicle_prev_location), 
						ceil(self.ego_vehicle.getLocation())):
			reward += self.rewardMap[i] * coef
			self.rewardMap[i] = 0
		if (self.nextLight):
			if (self.ego_vehicle_prev_location < self.nextLight.getLocation()) & \
				 (self.nextLight.getLocation() < self.ego_vehicle.getLocation()):
				if (self.nextLight.getPhase() == "red"):
					terminated = True
					reward -= self.accumulatedReward
				else:
					reward += 10 * self.lightsPassed
					self.lightsPassed += 1
					self.nextLight = self._find_next_light()
		reward -= self.delta_t
		self.ego_vehicle_prev_location = self.ego_vehicle.getLocation()
		self.accumulatedReward += reward
		self.stepReward = reward
		return terminated

	def getStepReward(self) -> float:
		return self.stepReward

	def getAccumulatedReward(self) -> int:
		return self.accumulatedReward

	def reset(self, seed=None) -> None:
		self.accumulatedReward = self.initialState[0]
		self.ego_vehicle = self.initialState[1]
		self.ego_vehicle_prev_location = self.ego_vehicle.getLocation()
		self.ticks = 0
		self.mapSize = self.initialState[2]
		self.rewardMap = self.initialState[3]	
		self.delta_t = self.initialState[4]
		self.trafficLights = self.initialState[5]
		self.nextLight = self._find_next_light()
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
		if (not self.trafficLights):
			return None
			nextLight = None
		for light in self.trafficLights:
			if (light.getLocation() < self.ego_vehicle.getLocation()):
				if (not nextLight):
					nextLight = light
				else:
					if (nextLight.getLocation() < nextLight.getLocation()):
						nextLight = light
		return nextLight