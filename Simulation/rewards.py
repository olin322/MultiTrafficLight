
from World import World
from Actor import Actor
from Vehicle import Vehicle


class RewardMap:
	
	def __init__(self, mapSize: int, 
				initialReward = 0: int, 
				ego_vehicle: Vehicle):
		# mapSize represents the length of the road
		# in this 1-dimension world
		self.mapSize = mapSize
		self.reward = initialReward

	def tick():
		pass

	def updateRewardMap():
		pass

	def calcReward() -> int:
		pass
		return self.reward

	def getReward() -> int:
		return self.reward