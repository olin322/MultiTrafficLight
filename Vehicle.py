# import Util.Location
from Actor import Actor
from TrafficLight import TrafficLight

### Units
# velocity: m/s
# location: distance from origin, 1-D space/x-axis

### 参考GB/T 33577，我国驾驶员的平均反应时间在0.3 s～2 s之间，
### 驾驶员制动平均减速度为3.6m/s^2～7.9 m/s^2
class Vehicle(Actor):
	def __init__(self, location: float, mass: float, \
				max_acceleration: float, max_deacceleration: float, \
				delta_t: float, speed = 0.0, \
				speedLimit = 16.6667):
		Actor.__init__(self, location)
		self.mass = mass
		self.max_acceleration = max_acceleration
		self.max_deacceleration = max_deacceleration
		self.speed = speed
		# self.acceleration = acceleration
		self.speedLimit = speedLimit
		self.delta_t = delta_t


	def tick(self, nextLight: TrafficLight) -> None:
		if(RLVW(nextLight)):
			self.deaccelerate()
		elif (self.speed < self.speedLimit):
			self.accelerate()
		return None

	def accelerate(self) -> None:
		self.speed += min(self.max_acceleration * self.delta_t, self.speedLimit)
		self.location += self.speed * self.delta_t
		return None

	def deaccelerate(self) -> None:
		self.speed -= max(self.max_deacceleration * delta_t, 0)
		self.location += speed * delta_t
		return None

	# return true if continue driving at current speed will pass red light
	# 
	def RLVW(nextLight: TrafficLight) -> bool:
		if(nextLight.getPhase() == "red"):
			if(nextLight.getLocation() - self.location < self.speed * nextLight.getCountdown()):
				return True
		elif(nextLight.getPhase() == "green"):
			if(nextLight.getLocation() - self.location > self.speed * nextLight.getCountdown()):
				return True
		else:
			return False

	def getLocation(self) -> float:
		print(self.location)
		return self.location