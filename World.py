# import Util.Location
from Actor import Actor
from Vehicle import Vehicle

class World:
    def __init__(self, delta_t: float):
        self.delta_t = delta_t
        self.simulation_time = 0
        self.actors = []

    def tick(self) -> None:
        for actor in self.actors:
            if (typeof(actor) == "Vehicle"):
                
        self.simulation_time += self.delta_t
        return None

    def spawn_vehicle(self, vehicle: Actor) -> None:
        self.actors.append(vehicle)
        return None

    def set_traffic_lights(self):
        return None

    def get_delta_t(self) -> float:
        return self.delta_t

    def get_simulation_time(self) -> float:
        return self.simulation_time