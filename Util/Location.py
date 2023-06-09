class Location:

	def __init__(x: float, y: float):
		this.x = x;
		this.y = y;

	# Parameters: 
	# location - The other point to compute the distance with
	def distance(self, location: Location) -> float:
		# returns Euclidean distance from this location to another one
		return sqrt(sqr(location.x - this.x) + sqr(location.y - this.y))