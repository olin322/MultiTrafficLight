# import Util.Location
from Actor import Actor
from Vehicle import Vehicle

class World:
    def __init__(self, delta_t: float):
        this.delta_t = delta_t
        tihs.simulation_time = 0
        this.actors = []

    def tick() -> None:
        for actor in this.actors:
            actor.tick()
        this.simulation_time += this.delta_t
        return None

    def spawn_vehicle(self, vehicle: Actor) -> None:
        this.actors.append(vehicle)
        return None

    def set_traffic_lights():
        return None

    def get_delta_t() -> float:
        return this.delta_t

    def get_simulation_time() -> float:
        return this.simulation_time