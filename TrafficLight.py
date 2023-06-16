import random
import math
from Actor import Actor

class TrafficLight(Actor):

	# Constructor
	# location: location of the traffic light, also is distance from origin
	# phase: "red" == red light, "green" == green light, "yellow" == yellow light
	# timer: amount of time current phase remaining, decrement every tick() so it is float type
	# 
	def __init__(self, location: float, phase: str, \
				timer: float, delta_t: float):
		Actor.__init__(location)
		self.phase = phase
		self.timer = timer
		self.delta_t = delta_t
		self.countdown = int(math.ceil(timer))

	def setLocation(self, location: float) -> None:
		self.location = location
		return None

	def setSPAT(self, phase: str, timer: float) -> None:
		self.phase = phase
		self.timer = timer
		self.countdown = math.ceil(timer)
		return None

	def tick(self):
		if(timer > 0):
			timer -= delta_t
		self.countdown = math.ceil(timer)
		updatePhase()
	
	def updatePhase() -> None:
		if(countdown == 0):
			if(self.phase == "green"):
				self.phase = "yello"
				self.timer = 3 # 黄灯固定3秒
			elif(self.phase == "yello"):
				self.phase = "red"
				self.timer = random.randint(60, 90)
			else:
				self.phase = "green"
				self.timer = random.randint(60, 90)
		return None

	def getLocation(self) -> float:
		return self.location

	def getPhase(self) -> str:
		return self.phase

	def getCountdown(self) -> int:
		return self.countdown