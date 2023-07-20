# from Util.Location import Location


# class Actor:
# 	def __init__(self, spawn_point:Location):
# 		self.location = spawn_point

class Actor:
	def __init__(self, id: str, location: float):
		self.id = id
		self.location = location

	def getID(self) -> str:
		return self.id

	def getLocation(self) -> float
		return self.location

	def setLocation(self, location: float) -> None:
		self.location = location
		return None

	def tick(self) -> None:
		return None

	def reset(self):
		return None