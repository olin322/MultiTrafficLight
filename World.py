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
            if (getClassName(str(type(actor)))  == "Vehicle"):
                nextLight = self.find_next_light()
                if(nextLight):
                    actor.tick(self.find_next_light())
                else:
                    actor.accelerate()
            else:
                actor.tick()
        self.simulation_time += self.delta_t
        return None

    def spawn_vehicle(self, vehicle: Actor) -> None:
        self.actors.append(vehicle)
        return None

    def add_traffic_light(self, trafficLight: TrafficLight) -> bool:
        if (getClassName(str(type(trafficLight))) != "TrafficLight"):
            return False
        self.actors.append(trafficLight)
        return True

    def get_delta_t(self) -> float:
        return self.delta_t

    def get_simulation_time(self) -> float:
        return self.simulation_time

    def find_next_light(self) -> TrafficLight:
        if (len(self.actors) < 2):
            return None
        vehicle = None
        closest = None
        for actor in self.actors:
            if (getClassName(str(type(actor))) == "Vehicle"):
                vehicle = actor
            if (getClassName(str(type(actor))) == "TrafficLight"):
                if (actor.getLocation() > vehicle.getLocation()):
                    if (closest):
                        if (actor.getLocation() < vehicle.getLocation()):
                            closest = actor
                    else:
                        closest = actor
        return closest