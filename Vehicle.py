# import Util.Location

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
		this.mass = mass
		this.max_acceleration = max_acceleration
		this.max_deacceleration = max_deacceleration
		this.speed = speed
		# this.acceleration = acceleration
		this.speedLimit = speedLimit
		this.delta_t = delta_t


	def tick(nextLight: TrafficLight) -> None:
		if(RLVW(nextLight)):
			deaccelerate()
		elif (this.speed < this.speedLimit):
			speed += min(max_acceleration * delta_t, this.speedLimit)
			location += speed * delta_t
		return None

	def deaccelerate() -> None:
		this.speed -= max_deacceleration * delta_t
		this.location += speed * delta_t
		return None

	# return true if red light warning should be issued
	# 
	def RLVW(nextLight: TrafficLight) -> bool:
		if(nextLight.getPhase() == "red"):
			if(nextLight.getLocation - this.location < this.speed * nextLight.getCountdown):
				return True
		elif(nextLight.getPhase == "green"):
			if(nextLight.getLocation - this.location > this.speed * nextLight.getCountdown):
				return True
		else:
			return False

	def getLocation() -> float:
		return this.location