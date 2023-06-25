# import Util.Location
from Actor import Actor
from TrafficLight import TrafficLight
from math import ceil

### Units
# velocity: m/s
# location: distance from origin, 1-D space/x-axis

### 参考GB/T 33577，我国驾驶员的平均反应时间在0.3 s～2 s之间，
### 驾驶员制动平均减速度为3.6m/s^2～7.9 m/s^2

###
DISTANCE_STOPLINE_TO_TRAFFIC_LIGHT = 2 # m

class Vehicle(Actor):
	def __init__(self, location: float, mass: float, \
				max_acceleration: float, max_deacceleration: float, \
				delta_t: float, speed = 0.0, \
				speedLimit = 16.6667):
		super().__init__(location)
		self.mass = mass
		self.max_acceleration = max_acceleration
		self.max_deacceleration = max_deacceleration
		self.speed = speed
		# self.acceleration = acceleration
		self.speedLimit = speedLimit
		self.delta_t = delta_t


	def tick(self, nextLight: TrafficLight) -> None:
		if (self.location < (nextLight.getLocation() - self.min_distance_to_brake())):
			self.accelerate()
		else:
			if (nextLight.getPhase() == "green"):
				if (ceil(self.speed) == 0):
					self.accelerate()
				elif (ceil((nextLight.getLocation() - self.location) / self.speed) 
					>= (nextLight.getCountdown())):
					self.deaccelerate()
				else:
					self.accelerate()
			elif (nextLight.getPhase() == "red"):
				if (ceil(self.speed) == 0):
					pass
				elif (ceil((nextLight.getLocation() - self.location) / self.speed) 
					> (nextLight.getCountdown())):
					self.moveAtCurrentSpeed()
				else:
					self.deaccelerate()
			else:
				self.deaccelerate()
		# self.accelerate()
		return None

	def moveAtCurrentSpeed(self) -> None:
		self.location += self.speed * self.delta_t
		return None

	def accelerate(self) -> None:
		self.speed = min(self.speed + self.max_acceleration * self.delta_t, self.speedLimit)
		self.location += self.speed * self.delta_t
		return None

	def deaccelerate(self) -> None:
		if (self.speed - self.max_deacceleration * self.delta_t > 0):
			self.speed = self.speed - self.max_deacceleration * self.delta_t
		else:
			self.speed = 0
		self.location += self.speed * self.delta_t
		return None


	def willPassGreenLight(self, nextLight: TrafficLight) -> bool:
		if (nextLight.getPhase() == "green"):
			t = (self.speedLimit - self.speed) / self.max_acceleration
			d = self.speed * t + (self.max_acceleration * t ** 2) / 2
			if (((nextLight.getLocation() - self.location - d) 
				/ self.speedLimit) < nextLight.getCountdown()):
				return True
		return False

	# return true if continue driving at current speed will pass red light
	#
	def willPassRedLight(self, nextLight: TrafficLight) -> bool:
		if (nextLight.getPhase() == "red"):
			if ((nextLight.getLocation() - self.location) < 
				(self.speed * (nextLight.getCountdown() - 1))):
				return True
		elif (nextLight.getPhase() == "green"):
			if ((nextLight.getLocation() - self.location) > 
				(self.speed * (nextLight.getCountdown() + 1))):
				return True
		elif (nextLight.getPhase() == "yellow"):
			return True
		else:
			return False

	def min_distance_to_brake(self) -> float:
		n = ceil(self.speed / self.max_deacceleration / self.delta_t)
		print("n = ", n)
		d = 0
		s = self.speed
		for _ in range(n + 1):
			s -= self.max_deacceleration * self.delta_t
			d += s * self.delta_t
		return d + DISTANCE_STOPLINE_TO_TRAFFIC_LIGHT
		# return((self.speed ** 2) / (2 * self.max_deacceleration) 
		# 	+ DISTANCE_STOPLINE_TO_TRAFFIC_LIGHT)


	def getLocation(self) -> float:
		# print(self.location)
		return self.location

	def getSpeed(self) -> float:
		return self.speed