# import Util.Location
from Actor import Actor
from TrafficLight import TrafficLight
from math import ceil

from overrides import override 

### Units
# velocity: m/s
# location: distance from origin, 1-D space/x-axis

### 参考GB/T 33577，我国驾驶员的平均反应时间在0.3 s～2 s之间，
### 驾驶员制动平均减速度为3.6m/s^2～7.9 m/s^2

###
DISTANCE_STOPLINE_TO_TRAFFIC_LIGHT = 10 # m


"""
@max_acceleration and @max_deacceleration both are initialized or given in positive value
initialization and modification should follow accordingly
"""
class Vehicle(Actor):
	def __init__(self, ID: str, location: float, mass: float, \
				max_acceleration: float, max_deacceleration: float, \
				delta_t: float, speed = 0.0, \
				speedLimit = 16.6667):
		self.INITIAL_STATE = [location, mass, 
							max_acceleration, max_deacceleration,
							delta_t, speed, speedLimit]
		super().__init__(ID, location)
		self.mass = mass
		self.max_acceleration = max_acceleration
		self.max_deacceleration = max_deacceleration
		self.speed = speed
		# self.acceleration = acceleration
		self.speedLimit = speedLimit
		self.delta_t = delta_t
		self.deaccelerate_mode = False


	def tick(self, nextLight: TrafficLight) -> None:
		if (self.deaccelerate_mode):
			if (self.speed == 0) & (nextLight.getPhase() == "green"):
				self.deaccelerate_mode = False
			else:
				self.deaccelerate()
		elif (self.location < (nextLight.getLocation() - self.min_distance_to_brake())):
			self.accelerate()
		else:
			if (nextLight.getPhase() == "green"):
				if (self.willPassGreenLight(nextLight)):
					self.accelerate()
				else:
					self.deaccelerate_mode = True
					self.deaccelerate()
				# if (ceil(self.speed) == 0):
				# 	self.accelerate()
				# elif (ceil((nextLight.getLocation() - self.location) / self.speed) 
				# 		>= (nextLight.getCountdown())):
				# 	self.deaccelerate()
				# 	self.deaccelerate_mode = True
				# else:
				# 	self.accelerate()
			elif (nextLight.getPhase() == "red"):
				if (ceil(self.speed) == 0):
					pass
				elif (ceil((nextLight.getLocation() - self.location) / self.speed) 
					> (nextLight.getCountdown())):
					self.moveAtCurrentSpeed()
				else:
					self.deaccelerate()
					self.deaccelerate_mode = True
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

	def accelerateAt(self, acceleration: float) -> None:
		if (acceleration >= 0):
			# speed = acceleration * self.delta_t + self.speed
			# if (speed < self.speedLimit):
			# 	self.speed += speed
			# else:
			# 	self.speed = self.speedLimit
			# print(self.speedLimit)
			self.speed = min(self.speedLimit, (self.speed + acceleration * self.delta_t))
		else:
			# print(self.speedLimit)
			self.speed = max(0, self.speed + acceleration * self.delta_t)
		"""
		the following line was WRONG
		and probably was why training terminated early after 65K episodes
		# self.speed = self.speed + (acceleration * self.delta_t)
		"""
		self.location = self.location + (self.speed * self.delta_t)
		return None

	def willPassGreenLight(self, nextLight: TrafficLight) -> bool:
		if (nextLight.getPhase() == "green"):
			n1 = ceil((self.speedLimit - self.speed) / self.max_deacceleration / self.delta_t)
			d1 = 0
			s = self.speed
			for _ in range(n1 + 1):
				s += self.max_acceleration * self.delta_t
				d1 += s * self.delta_t
			if (ceil((nextLight.getLocation() - self.location - d1) / self.speedLimit) < 
					nextLight.getCountdown()):
				return True
			# t = (self.speedLimit - self.speed) / self.max_acceleration
			# d = self.speed * t + (self.max_acceleration * t ** 2) / 2
			# if (((nextLight.getLocation() - self.location - d) 
			# 	/ self.speedLimit) < nextLight.getCountdown()):
			# 	return True
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
		for _ in range(n):
			d += s * self.delta_t
			s -= self.max_deacceleration * self.delta_t
			
		return d + DISTANCE_STOPLINE_TO_TRAFFIC_LIGHT
		# return((self.speed ** 2) / (2 * self.max_deacceleration) 
		# 	+ DISTANCE_STOPLINE_TO_TRAFFIC_LIGHT)


	# def getLocation(self) -> float:
	# 	# print(self.location)
	# 	return self.location

	def getSpeed(self) -> float:
		return self.speed

	@override
	def reset(self, seed=None) -> None:
		self.location = self.INITIAL_STATE[0]
		self.mass = self.INITIAL_STATE[1]
		self.max_acceleration = self.INITIAL_STATE[2]
		self.max_deacceleration = self.INITIAL_STATE[3]
		self.delta_t = self.INITIAL_STATE[4]
		self.speed = self.INITIAL_STATE[5]
		self.speedLimit = self.INITIAL_STATE[6]
		
		self.deaccelerate_mode = False
		# print("vehicle", seed)
		return None