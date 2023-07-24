
from World import World
from Actor import Actor
from Vehicle import Vehicle
from math import floor, ceil

class RewardMap:
	
	def __init__(self, mapSize: int, 
				ego_vehicle: Vehicle,
				initialReward: int = 0, 
				):
		# mapSize represents the length of the road
		# in this 1-dimension world
		self.mapSize = mapSize
		self.reward = initialReward
		self.ego_vehicle_initial_state = ego_vehicle.getLocation()
		self.ego_vehicle_prev_location = ego_vehicle.getLocation()
		self.rewardMap = [1] * mapSize

	def tick(self):
		self.calcReward()

	def calcReward(self) -> None:
		# reward coefficient
		# 1 - (time passed/total time needed)
		coef = 1 - (World.get_simulation_time() / ceil(self.mapSize / 16.67))
		for i in range(floor(ego_vehicle_prev_location), 
						ceil(ego_vehicle.getLocation())):
			self.reward += self.rewardMap[i] * coef
			self.rewardMap[i] = 0
		return None

	def getReward(self) -> int:
		return self.reward