import random
import math
import Actor

class TrafficLight(Actor):

	# Constructor
	# location: location of the traffic light, also is distance from origin
	# phase: "red" == red light, "green" == green light, "yellow" == yellow light
	# timer: amount of time current phase remaining, decrement every tick() so it is float type
	# 
	def __init__(self, location: float, phase: str, \
				timer: float, delta_t: float):
		Actor.__init__(location)
		this.phase = phase
		this.timer = timer
		this.delta_t = delta_t
		this.countdown = int(math.ceil(timer))

	def setLocation(location: float) -> None:
		this.location = location
		return None

	def setSPAT(phase: str, timer: float) -> None:
		this.phase = phase
		this.timer = timer
		this.countdown = math.ceil(timer)
		return None

	def tick():
		if(timer > 0):
			timer -= delta_t
		this.countdown = math.ceil(timer)
		updatePhase()
	
	def updatePhase() -> None:
		if(countdown == 0):
			if(this.phase == "green"):
				this.phase = "yello"
				this.timer = 3 # 黄灯固定3秒
			elif(this.phase == "yello"):
				this.phase = "red"
				this.timer = random.randint(60, 90)
			else:
				this.phase = "green"
				this.timer = random.randint(60, 90)
		return None

	def getLocation() -> float:
		return this.location

	def getPhase() -> str:
		return this.phase

	def getCountdown() -> int:
		return this.countdown