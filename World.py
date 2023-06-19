import Util.Location
import Actor
import Vehicle
import TrafficLight
from Util.Utils import getClassName

class World:
    def __init__(self, delta_t: float):
        self.delta_t = delta_t
        self.simulation_time = 0
        self.actors = []

    def tick(self) -> None:
        for actor in self.actors:
            print(type(actor))
            if(getClassName(str(type(actor)))  == "Vehicle"):
                nextLight = self.find_next_light()
                if(nextLight):
                    actor.tick(find_next_light())
                else:
                    actor.accelerate()
            else:
                pass
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

    def find_next_light(self) -> TrafficLight:
        if(len(self.actors) < 2):
            return None
        vehicle = None
        closest = None
        for actor in self.actors:
            if(type(actor) == "Vehicle"):
                vehicle = actor
                break
        if(type(self.actors[0]) == "TrafficLight"):
            closest = self.actors[0]
        else:
            closest = self.actors[1]
        for actor in self.actors:
            if(type(actor) == "TrafficLight"):
                if((closest.location - vehicle.location) < 0):
                    closest = actor
                else:
                    if((actor.location - vehicle.location) < (closest.location - vehicle.location)):
                        closest = actor
        return closest