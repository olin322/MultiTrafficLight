# import Util.Location
# from overrides import override

import Actor
import Vehicle
import TrafficLight

from Util.Utils import getClassName

class World:
    def __init__(self, delta_t: float):
        self.delta_t = delta_t
        self.simulation_time = 0
        self.actors = []
        self.frame = 0

    def tick(self) -> None:
        self.frame += 1
        for actor in self.actors:
            if (getClassName(str(type(actor)))  == "Vehicle"):
                nextLight = self._find_next_light()
                if(nextLight):
                    actor.tick(self._find_next_light())
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

    @staticmethod
    def get_simulation_time(self) -> float:
        return self.simulation_time

    def get_ego_vehicle(self) -> Vehicle:
        for actor in self.actors:
            if (getClassName(str(type(actor)))  == "Vehicle"):
                ego_vehicle = actor
                return ego_vehicle
        # raise error "ego vehicle not found"
        return None

    def getFrame() -> int:
        return self.frame

    def numTrafficLightAhead(self, actor: Actor) -> int:
        n = 0
        for a in actors:
            n += 1
            if (a.getID() == actor.getID()):
                return len(actors) - n
        # should raise actor not found error
        return None

    def reset(self) -> None:
        for a in self.actors:
            a.reset()
        return None




###############################################################################

"""
    private methods
"""

def _find_next_light(self) -> TrafficLight:
        if (len(self.actors) < 2):
            return None
        vehicle = None
        closest = None
        for actor in self.actors:
            if (getClassName(str(type(actor))) == "Vehicle"):
                vehicle = actor
            if (getClassName(str(type(actor))) == "TrafficLight"):
                if (actor.getLocation() > vehicle.getLocation()):
                    if (closest != None):
                        if (actor.getLocation() < closest.getLocation()):
                            closest = actor
                    else:
                        closest = actor
        return closest

def _find_Actor_Type(self, actor: Actor) -> str:
    return getClassName(str(type(actor)))