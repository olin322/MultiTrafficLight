import random
import math
from Actor import Actor

from overrides import override

class TrafficLight(Actor):

	# Constructor
	# location: location of the traffic light, also is distance from origin
	# phase: "red" == red light, "green" == green light, "yellow" == yellow light
	# timer: amount of time current phase remaining, decrement every tick() so it is float type
	# 
	def __init__(self, id: str, location: float, phase: str, \
				timer: float, delta_t: float):
		super().__init__(id, location)
		self.phase = phase
		self.timer = timer
		self.delta_t = delta_t
		self.countdown = int(math.ceil(timer))
		self.timer_repeat = timer

	

	def tick(self):
		if (self.timer > 0):
			self.timer -= self.delta_t
		self.countdown = math.ceil(self.timer)
		self.updatePhase()
	

	def reset(self):
		self.timer = self.timer_repeat
		self.countdown = int(math.ceil(self.timer))
		self.phase = "green"

	## TO-DO:
	## resolve the issue during phase transition
	## next light status = 1yellow
	## next light status = 1red
	## next light status = 70red
	def updatePhase(self) -> None:
		if ((self.timer - self.delta_t) <= 0.0001):
			if (self.phase == "green"):
				self.phase = "yellow"
				self.timer = 3 # 黄灯固定3秒
			elif (self.phase == "yellow"):
				self.phase = "red"
				self.timer = self.timer_repeat
			else:
				self.phase = "green"
				self.timer = self.timer_repeat
		return None
	

	# def setLocation(self, location: float) -> None:
	# 	self.location = location
	# 	return None

	def setSPAT(self, phase: str, timer: float) -> None:
		self.phase = phase
		self.timer = timer
		self.countdown = math.ceil(timer)
		return None


	# def getLocation(self) -> float:
	# 	return self.location

	def getPhase(self) -> str:
		return self.phase

	def getPhaseInFloat(self) -> float:
		t = {
			"green": 0,
			"red": 1,
			"yellow": 2
		}
		return float(t[self.getPhase()])

	def getCountdown(self) -> int:
		return self.countdown

